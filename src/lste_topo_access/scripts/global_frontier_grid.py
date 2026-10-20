#!/usr/bin/env python3
"""Pure occupancy-grid primitives for online frontier exploration.

This module deliberately knows nothing about ROS nodes, route ownership, or
place memory.  It contains the deterministic raster operations shared by the
frontier selector, structural-place lifecycle, and costmap validation.
Keeping these operations here makes their safety and coordinate contracts easy
to test without constructing a running navigation stack.
"""

import collections
import heapq
import math

import numpy as np


def conservative_clearance_cells(clearance_m, resolution):
    """Convert a physical centre-clearance contract to grid cells.

    Occupancy cells represent an area, while the route endpoint is published
    at a cell centre.  Rounding down (or subtracting one cell after a ceil)
    can admit a centre whose footprint plus the controller's obstacle margin
    overlaps the occupied cell.  Keep this conversion in one place so the
    exploration route mask uses the same conservative contract as the
    Navfn/TEB endpoint checks.
    """
    try:
        clearance_m = max(0.0, float(clearance_m))
        resolution = float(resolution)
    except (TypeError, ValueError):
        return 1
    if resolution <= 0.0:
        return 1
    return max(1, int(math.ceil(clearance_m / resolution)))


def inflate(occupied, cells):
    """Return ``occupied`` dilated by a circular radius in grid cells."""
    inflated = occupied.copy()
    rows, cols = occupied.shape
    for delta_row in range(-cells, cells + 1):
        for delta_col in range(-cells, cells + 1):
            if delta_row * delta_row + delta_col * delta_col > cells * cells:
                continue
            source_row_start = max(0, -delta_row)
            source_row_stop = min(rows, rows - delta_row)
            source_col_start = max(0, -delta_col)
            source_col_stop = min(cols, cols - delta_col)
            target_row_start = max(0, delta_row)
            target_row_stop = min(rows, rows + delta_row)
            target_col_start = max(0, delta_col)
            target_col_stop = min(cols, cols + delta_col)
            inflated[
                target_row_start:target_row_stop,
                target_col_start:target_col_stop,
            ] |= occupied[
                source_row_start:source_row_stop,
                source_col_start:source_col_stop,
            ]
    return inflated


def nearest_seed(free, row, col, limit):
    """Find the closest free cell around a requested grid coordinate."""
    rows, cols = free.shape
    if 0 <= row < rows and 0 <= col < cols and free[row, col]:
        return row, col
    for radius in range(1, limit + 1):
        row_start = max(0, row - radius)
        row_stop = min(rows, row + radius + 1)
        col_start = max(0, col - radius)
        col_stop = min(cols, col + radius + 1)
        candidates = np.argwhere(free[row_start:row_stop, col_start:col_stop])
        if candidates.size:
            candidates[:, 0] += row_start
            candidates[:, 1] += col_start
            distances = (
                (candidates[:, 0] - row) ** 2
                + (candidates[:, 1] - col) ** 2
            )
            result = candidates[np.argmin(distances)]
            return int(result[0]), int(result[1])
    return None


def bfs(free, seed, should_abort=None):
    """Return a four-connected shortest-path field, or ``None`` if invalidated."""
    steps = np.full(free.shape, -1, dtype=np.int32)
    queue = collections.deque([seed])
    steps[seed] = 0
    rows, cols = free.shape
    visited = 0
    while queue:
        if should_abort is not None and visited % 2048 == 0 and should_abort():
            return None
        row, col = queue.popleft()
        visited += 1
        next_step = steps[row, col] + 1
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row, next_col = row + delta_row, col + delta_col
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and free[next_row, next_col]
                and steps[next_row, next_col] < 0
            ):
                steps[next_row, next_col] = next_step
                queue.append((next_row, next_col))
    return steps


def frontier_mask(free, unknown):
    """Return free cells with an orthogonally adjacent unknown cell."""
    adjacent_unknown = np.zeros_like(unknown, dtype=bool)
    adjacent_unknown[1:] |= unknown[:-1]
    adjacent_unknown[:-1] |= unknown[1:]
    adjacent_unknown[:, 1:] |= unknown[:, :-1]
    adjacent_unknown[:, :-1] |= unknown[:, 1:]
    return free & adjacent_unknown


def cell_xy(message, row, col):
    """Return the map-frame centre of an occupancy-grid cell."""
    return (
        message.info.origin.position.x + (col + 0.5) * message.info.resolution,
        message.info.origin.position.y + (row + 0.5) * message.info.resolution,
    )


def xy_to_grid_cell(message, x, y):
    """Return a grid cell for one map-frame coordinate, or ``None`` out of map."""
    resolution = float(message.info.resolution)
    if resolution <= 0.0:
        return None
    col = int(math.floor(
        (float(x) - float(message.info.origin.position.x)) / resolution
    ))
    row = int(math.floor(
        (float(y) - float(message.info.origin.position.y)) / resolution
    ))
    if not (0 <= row < int(message.info.height) and 0 <= col < int(message.info.width)):
        return None
    return row, col


def frontier_information(unknown, row, col, radius):
    """Count unknown cells in a square observation neighbourhood."""
    row_start = max(0, row - radius)
    row_stop = min(unknown.shape[0], row + radius + 1)
    col_start = max(0, col - radius)
    col_stop = min(unknown.shape[1], col + radius + 1)
    return float(np.count_nonzero(unknown[row_start:row_stop, col_start:col_stop]))


def waypoint_on_path(steps, row, col, lookahead_steps):
    """Walk a BFS field backward to the requested lookahead distance."""
    current = (row, col)
    while steps[current] > lookahead_steps:
        candidates = []
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = current[0] + delta_row, current[1] + delta_col
            if (
                0 <= neighbour[0] < steps.shape[0]
                and 0 <= neighbour[1] < steps.shape[1]
                and 0 <= steps[neighbour] < steps[current]
            ):
                candidates.append(neighbour)
        if not candidates:
            return None
        current = min(candidates, key=lambda item: steps[item])
    return current


def route_path(steps, seed, target):
    """Recover a deterministic shortest route from a BFS distance field."""
    if steps is None or seed is None or target is None:
        return []
    if not (
        0 <= seed[0] < steps.shape[0]
        and 0 <= seed[1] < steps.shape[1]
        and 0 <= target[0] < steps.shape[0]
        and 0 <= target[1] < steps.shape[1]
    ):
        return []
    if steps[seed] < 0 or steps[target] < 0:
        return []
    current = (int(target[0]), int(target[1]))
    reverse_path = [current]
    while current != (int(seed[0]), int(seed[1])):
        current_step = int(steps[current])
        candidates = []
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = current[0] + delta_row, current[1] + delta_col
            if (
                0 <= neighbour[0] < steps.shape[0]
                and 0 <= neighbour[1] < steps.shape[1]
                and 0 <= steps[neighbour] < current_step
            ):
                candidates.append(neighbour)
        if not candidates:
            return []
        # Lower path cost is the primary criterion; row/column makes equal
        # BFS ties deterministic across planning cycles.
        current = min(
            candidates,
            key=lambda item: (int(steps[item]), item[0], item[1]),
        )
        reverse_path.append(current)
    return list(reversed(reverse_path))


def geodesic_dijkstra(free, seed, resolution=1.0, should_abort=None):
    """Return an 8-connected weighted shortest-path geodesic distance field.

    Orthogonal edges are weighted by 1.0 * resolution.
    Diagonal edges are weighted by sqrt(2) * resolution.
    Unreachable or obstacle cells remain float('inf').
    """
    rows, cols = free.shape
    distances = np.full((rows, cols), np.inf, dtype=np.float64)
    if seed is None:
        return distances
    try:
        s_row, s_col = int(seed[0]), int(seed[1])
    except (IndexError, TypeError, ValueError):
        return distances

    if not (0 <= s_row < rows and 0 <= s_col < cols):
        return distances
    if not free[s_row, s_col]:
        return distances

    res = float(resolution)
    if res <= 0.0:
        res = 1.0
    diag_cost = math.sqrt(2.0) * res
    ortho_cost = 1.0 * res

    distances[s_row, s_col] = 0.0
    heap = [(0.0, s_row, s_col)]

    ortho_dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    diag_dirs = ((1, 1), (1, -1), (-1, 1), (-1, -1))

    visited = 0
    while heap:
        if should_abort is not None and visited % 2048 == 0 and should_abort():
            return None
        d, r, c = heapq.heappop(heap)
        visited += 1
        if d > distances[r, c]:
            continue

        for dr, dc in ortho_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and free[nr, nc]:
                cand = d + ortho_cost
                if cand < distances[nr, nc]:
                    distances[nr, nc] = cand
                    heapq.heappush(heap, (cand, nr, nc))

        for dr, dc in diag_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and free[nr, nc]:
                # Corner-cutting prevention: require orthogonal neighbors to be free
                if free[r, nc] and free[nr, c]:
                    cand = d + diag_cost
                    if cand < distances[nr, nc]:
                        distances[nr, nc] = cand
                        heapq.heappush(heap, (cand, nr, nc))

    return distances


def _extract_xy_from_object(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        if obj.get("anchor_xy") is not None:
            return obj["anchor_xy"]
        if obj.get("viewpoint_xy") is not None:
            return obj["viewpoint_xy"]
        if obj.get("x") is not None and obj.get("y") is not None:
            return (float(obj["x"]), float(obj["y"]))
        if obj.get("xy") is not None:
            return obj["xy"]
    elif hasattr(obj, "anchor_xy") and getattr(obj, "anchor_xy", None) is not None:
        return getattr(obj, "anchor_xy")
    elif hasattr(obj, "viewpoint_xy") and getattr(obj, "viewpoint_xy", None) is not None:
        return getattr(obj, "viewpoint_xy")
    elif hasattr(obj, "x") and hasattr(obj, "y"):
        return (float(getattr(obj, "x")), float(getattr(obj, "y")))
    elif isinstance(obj, (tuple, list)) and len(obj) >= 2:
        return (float(obj[0]), float(obj[1]))
    return None


def point_to_grid_cell(
    xy,
    message=None,
    resolution=None,
    origin_xy=None,
    physical_to_map=None,
    shape=None,
):
    """Convert an arbitrary (x, y) or object to integer grid cell (row, col)."""
    if xy is None:
        return None
    raw_xy = _extract_xy_from_object(xy)
    if raw_xy is None:
        return None
    px, py = float(raw_xy[0]), float(raw_xy[1])

    if physical_to_map is not None and callable(physical_to_map):
        try:
            m_xy = physical_to_map(px, py)
            if m_xy is not None:
                px, py = float(m_xy[0]), float(m_xy[1])
        except Exception:
            pass

    if message is not None:
        return xy_to_grid_cell(message, px, py)

    if resolution is not None and origin_xy is not None:
        res = float(resolution)
        if res <= 0:
            return None
        col = int(math.floor((px - float(origin_xy[0])) / res))
        row = int(math.floor((py - float(origin_xy[1])) / res))
        if shape is not None:
            if not (0 <= row < shape[0] and 0 <= col < shape[1]):
                return None
        return row, col

    # Fallback when coordinates are already grid indices
    row, col = int(round(px)), int(round(py))
    if shape is not None:
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            return None
    return row, col


def get_or_compute_spawn_geodesic_field(
    free,
    spawn_xy,
    message=None,
    resolution=None,
    origin_xy=None,
    physical_to_map=None,
    search_limit=25,
    should_abort=None,
):
    """Compute the 8-connected Dijkstra distance field from the spawn point."""
    if free is None or spawn_xy is None:
        return None
    res = float(message.info.resolution if message is not None else (resolution or 1.0))
    shape = free.shape
    seed_cell = point_to_grid_cell(
        spawn_xy,
        message=message,
        resolution=res,
        origin_xy=origin_xy,
        physical_to_map=physical_to_map,
        shape=shape,
    )
    if seed_cell is None:
        return None
    r, c = seed_cell
    if not (0 <= r < shape[0] and 0 <= c < shape[1] and free[r, c]):
        snapped = nearest_seed(free, r, c, limit=search_limit)
        if snapped is None:
            return None
        seed_cell = snapped

    return geodesic_dijkstra(
        free, seed_cell, resolution=res, should_abort=should_abort
    )


def compute_geodesic_distance_to_work_item(
    free,
    spawn_xy,
    target_item_or_xy,
    message=None,
    resolution=None,
    origin_xy=None,
    physical_to_map=None,
    search_limit=25,
    geodesic_field=None,
):
    """Compute shortest obstacle-avoiding geodesic distance from spawn_xy to target."""
    if free is None or spawn_xy is None or target_item_or_xy is None:
        return None

    if geodesic_field is None:
        geodesic_field = get_or_compute_spawn_geodesic_field(
            free=free,
            spawn_xy=spawn_xy,
            message=message,
            resolution=resolution,
            origin_xy=origin_xy,
            physical_to_map=physical_to_map,
            search_limit=search_limit,
        )
    if geodesic_field is None:
        return None

    shape = free.shape
    res = float(message.info.resolution if message is not None else (resolution or 1.0))

    target_cell = point_to_grid_cell(
        target_item_or_xy,
        message=message,
        resolution=res,
        origin_xy=origin_xy,
        physical_to_map=physical_to_map,
        shape=shape,
    )
    if target_cell is None:
        return None

    tr, tc = target_cell
    if not (0 <= tr < shape[0] and 0 <= tc < shape[1] and free[tr, tc]):
        snapped = nearest_seed(free, tr, tc, limit=search_limit)
        if snapped is not None:
            tr, tc = snapped
        elif isinstance(target_item_or_xy, dict) and target_item_or_xy.get("viewpoint_xy") is not None:
            vp_xy = target_item_or_xy["viewpoint_xy"]
            vp_cell = point_to_grid_cell(
                vp_xy,
                message=message,
                resolution=res,
                origin_xy=origin_xy,
                physical_to_map=physical_to_map,
                shape=shape,
            )
            if vp_cell is not None and free[vp_cell[0], vp_cell[1]]:
                tr, tc = vp_cell
            elif vp_cell is not None:
                vp_snapped = nearest_seed(free, vp_cell[0], vp_cell[1], limit=search_limit)
                if vp_snapped is not None:
                    tr, tc = vp_snapped
                else:
                    return None
            else:
                return None
        else:
            return None

    dist = geodesic_field[tr, tc]
    if math.isfinite(dist):
        return float(dist)
    return None
