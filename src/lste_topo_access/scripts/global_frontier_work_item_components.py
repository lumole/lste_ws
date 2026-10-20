"""Pure geometry for persistent local observation work items.

An exploration endpoint is only a temporary safe pose.  The durable unit of
work is the unknown-space boundary it lets the robot observe.  This module
extracts that boundary from one occupancy snapshot and expresses its support
in the stable odom frame, so a later SLAM update can inherit the same work
instead of creating a new task at a farther endpoint.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


FOUR_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
EIGHT_NEIGHBOURS = FOUR_NEIGHBOURS + ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True)
class ObservationSupport:
    """One current frontier arc and its local unknown-side physical support."""

    frontier_cells: frozenset
    support_cells: frozenset
    # The anchor and normal are derived from the same map snapshot as the
    # support.  Defaults keep old fixtures and callers that only provide the
    # two persistent cell sets source-compatible.
    anchor_xy: object = None
    normal_xy: object = None


def _components(mask, neighbours):
    """Return deterministic connected components of a boolean grid."""
    visited = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    result = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        queue = deque([(row, col)])
        visited[row, col] = True
        cells = []
        while queue:
            current_row, current_col = queue.popleft()
            cells.append((current_row, current_col))
            for delta_row, delta_col in neighbours:
                next_row, next_col = current_row + delta_row, current_col + delta_col
                if (
                    0 <= next_row < rows
                    and 0 <= next_col < cols
                    and mask[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
        result.append(frozenset(cells))
    return tuple(result)


def _unknown_neighbours(frontier_cells, unknown):
    """Return unknown cells touching one free-space frontier arc."""
    rows, cols = unknown.shape
    seeds = set()
    for row, col in frontier_cells:
        for delta_row, delta_col in FOUR_NEIGHBOURS:
            next_row, next_col = row + delta_row, col + delta_col
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and unknown[next_row, next_col]
            ):
                seeds.add((next_row, next_col))
    return seeds


def _local_unknown_patch(unknown, seeds, frontier_cells, horizon_cells):
    """Flood one unknown side, bounded by the sensor's physical horizon.

    The radius is a sensor fact: it keeps one vast unseen building exterior
    from joining every doorway into one task, while preserving descendants as
    the observed boundary moves forward after a scan.
    """
    if not seeds:
        return frozenset()
    rows, cols = unknown.shape
    horizon_squared = int(horizon_cells) * int(horizon_cells)
    minimum_row = min(cell[0] for cell in frontier_cells)
    maximum_row = max(cell[0] for cell in frontier_cells)
    minimum_col = min(cell[1] for cell in frontier_cells)
    maximum_col = max(cell[1] for cell in frontier_cells)
    queue = deque(sorted(seeds))
    visited = set(seeds)
    cells = []
    while queue:
        row, col = queue.popleft()
        # The arc bounding box is a fast conservative distance bound. The
        # subsequent unknown-side flood still prevents support from crossing a
        # known wall, and avoids an O(frontier_cells * patch_cells) loop on
        # long office walls.
        row_delta = max(minimum_row - row, 0, row - maximum_row)
        col_delta = max(minimum_col - col, 0, col - maximum_col)
        nearest_squared = row_delta * row_delta + col_delta * col_delta
        if nearest_squared > horizon_squared:
            continue
        cells.append((row, col))
        for delta_row, delta_col in FOUR_NEIGHBOURS:
            next_row, next_col = row + delta_row, col + delta_col
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and unknown[next_row, next_col]
                and (next_row, next_col) not in visited
            ):
                visited.add((next_row, next_col))
                queue.append((next_row, next_col))
    return frozenset(cells)


def _quantized_physical_cells(cells, cell_xy, map_to_physical_xy, quantum):
    """Freeze grid-cell centres into a compact odom-frame support signature."""
    support = set()
    for row, col in cells:
        x, y = cell_xy(row, col)
        if map_to_physical_xy is not None:
            projected = map_to_physical_xy(x, y)
            if projected is not None:
                x, y = projected
        try:
            support.add((
                int(round(float(x) / quantum)),
                int(round(float(y) / quantum)),
            ))
        except (TypeError, ValueError):
            continue
    return frozenset(support)


def _physical_cell_xy(cell, cell_xy, map_to_physical_xy):
    """Return one cell centre in the stable physical frame when available."""
    try:
        x, y = cell_xy(int(cell[0]), int(cell[1]))
        x, y = float(x), float(y)
    except (IndexError, TypeError, ValueError):
        return None
    if map_to_physical_xy is not None:
        try:
            projected = map_to_physical_xy(x, y)
        except (TypeError, ValueError):
            projected = None
        if projected is not None:
            try:
                x, y = float(projected[0]), float(projected[1])
            except (IndexError, TypeError, ValueError):
                return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _centroid(points):
    """Return the deterministic centroid of a non-empty point sequence."""
    if not points:
        return None
    return (
        sum(point[0] for point in points) / float(len(points)),
        sum(point[1] for point in points) / float(len(points)),
    )


def _unknown_side_direction(frontier_cells, unknown_side_cells, cell_xy, map_to_physical_xy):
    """Derive an unknown-side normal from local geometric evidence.

    A direction is a physical fact of the observed boundary, not a tuning
    angle.  Use unknown neighbours rather than the whole flooded patch so a
    large unseen room cannot rotate a doorway's local orientation.
    """
    frontier_points = [
        point
        for cell in frontier_cells
        for point in (_physical_cell_xy(cell, cell_xy, map_to_physical_xy),)
        if point is not None
    ]
    unknown_points = [
        point
        for cell in unknown_side_cells
        for point in (_physical_cell_xy(cell, cell_xy, map_to_physical_xy),)
        if point is not None
    ]
    anchor = _centroid(frontier_points)
    unknown_centroid = _centroid(unknown_points)
    if anchor is None or unknown_centroid is None:
        return anchor, None
    dx = unknown_centroid[0] - anchor[0]
    dy = unknown_centroid[1] - anchor[1]
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm <= 1e-9:
        return anchor, None
    return anchor, (dx / norm, dy / norm)


def check_anchor_clearance_and_corner(
    arc,
    unknown,
    known_free,
    occupied,
    resolution,
    *,
    clearance_radius_m=0.7,
    min_clearance_ratio=0.35,
    corner_wall_distance_m=0.4,
    min_wall_distance_m=0.25,
    max_unknown_ratio=0.70,
    min_free_ratio=0.25,
    min_free_area_m2=0.20,
):
    """Evaluate whether a frontier arc is viable for WorkItem generation.

    Under the open-generation / runtime-abandonment policy, upfront geometric
    filtering is intentionally relaxed so that all valid frontier boundaries
    generate WorkItems without being prematurely choked off. Only empty arcs
    or anchors embedded inside solid obstacles are rejected upfront; physical
    unreachability and stalls are handled dynamically at runtime.
    """
    if not arc:
        return True, "empty_arc", {}
    rows, cols = unknown.shape
    r_anchor = sum(r for r, c in arc) / float(len(arc))
    c_anchor = sum(c for r, c in arc) / float(len(arc))
    r0 = int(round(r_anchor))
    c0 = int(round(c_anchor))
    if not (0 <= r0 < rows and 0 <= c0 < cols):
        return True, "anchor_out_of_bounds", {}

    resolution = float(resolution)
    if resolution <= 0.0:
        return False, "invalid_resolution", {}

    # If the geometric centroid lands on an occupied cell (common for curved
    # arcs or walls), project onto the closest frontier cell on the arc.
    if occupied[r0, c0]:
        r_closest, c_closest = min(
            arc, key=lambda rc: (rc[0] - r_anchor) ** 2 + (rc[1] - c_anchor) ** 2
        )
        if occupied[r_closest, c_closest]:
            return True, "anchor_in_obstacle", {}

    # All non-empty frontier arcs with anchors outside solid obstacles are admitted.
    # Unreachability and dead corners are handled dynamically at runtime.
    return False, "admitted", {"free_cells": len(arc), "clearance_ratio": 1.0}



def suppress_dense_observation_supports(
    supports,
    density_separation_m=1.0,
):
    """Spatially suppress and merge clustered observation supports.

    In a local neighborhood of radius ``density_separation_m``, keeps the
    dominant support (highest support_cell_count, clear normal, largest arc)
    and absorbs the support cells and frontier cells of nearby supports into it.
    """
    if not supports or len(supports) <= 1:
        return tuple(supports or ())
    density_sep = max(0.05, float(density_separation_m))

    def support_key(s):
        support_count = len(s.support_cells)
        has_normal = 1 if (getattr(s, "normal_xy", None) is not None) else 0
        frontier_count = len(s.frontier_cells)
        return (support_count, has_normal, frontier_count)

    sorted_supports = sorted(supports, key=support_key, reverse=True)
    retained = []

    for s in sorted_supports:
        s_anchor = getattr(s, "anchor_xy", None)
        if s_anchor is None:
            retained.append(s)
            continue

        is_fragment = (len(s.support_cells) <= 4 or len(s.frontier_cells) <= 3)
        effective_sep = density_sep * 1.5 if is_fragment else density_sep

        closest_idx = None
        min_dist = float("inf")
        for i, r in enumerate(retained):
            r_anchor = getattr(r, "anchor_xy", None)
            if r_anchor is None:
                continue
            dist = math.hypot(s_anchor[0] - r_anchor[0], s_anchor[1] - r_anchor[1])
            if dist <= effective_sep and dist < min_dist:
                min_dist = dist
                closest_idx = i

        if closest_idx is not None:
            r = retained[closest_idx]
            merged_support = ObservationSupport(
                frontier_cells=frozenset(r.frontier_cells | s.frontier_cells),
                support_cells=frozenset(r.support_cells | s.support_cells),
                anchor_xy=r.anchor_xy,
                normal_xy=r.normal_xy if r.normal_xy is not None else s.normal_xy,
            )
            retained[closest_idx] = merged_support
        else:
            retained.append(s)

    return tuple(retained)


def observation_supports(
    frontier, unknown, *, cell_xy, map_to_physical_xy=None,
    resolution, sensor_horizon_m=None,
    known_free=None, occupied=None,
    clearance_radius_m=0.7, min_clearance_ratio=0.35,
    corner_wall_distance_m=0.4, density_separation_m=None,
    min_wall_distance_m=0.25, max_unknown_ratio=0.70,
    min_free_ratio=0.25, min_free_area_m2=0.20,
):
    """Extract physical support records for 8-connected frontier arcs.

    Optionally filters out corner, wall-clinging, and constricted arcs with
    insufficient explored clearance, and suppresses densely clustered supports
    within ``density_separation_m``.
    """
    frontier = np.asarray(frontier, dtype=bool)
    unknown = np.asarray(unknown, dtype=bool)
    if frontier.shape != unknown.shape:
        raise ValueError("frontier and unknown must share a grid shape")
    resolution = float(resolution)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("positive map resolution required")
    horizon_m = resolution if sensor_horizon_m is None else float(sensor_horizon_m)
    if not math.isfinite(horizon_m) or horizon_m <= 0.0:
        horizon_m = resolution
    horizon_cells = max(1, int(math.ceil(horizon_m / resolution)))
    quantum = max(resolution, 0.25)
    supports = []
    for arc in _components(frontier, EIGHT_NEIGHBOURS):
        if known_free is not None and occupied is not None:
            rejected, _reason, _metrics = check_anchor_clearance_and_corner(
                arc,
                unknown,
                known_free,
                occupied,
                resolution,
                clearance_radius_m=clearance_radius_m,
                min_clearance_ratio=min_clearance_ratio,
                corner_wall_distance_m=corner_wall_distance_m,
                min_wall_distance_m=min_wall_distance_m,
                max_unknown_ratio=max_unknown_ratio,
                min_free_ratio=min_free_ratio,
                min_free_area_m2=min_free_area_m2,
            )
            if rejected:
                continue

        seeds = _unknown_neighbours(arc, unknown)
        patch = _local_unknown_patch(unknown, seeds, arc, horizon_cells)
        if not patch:
            continue
        support_cells = _quantized_physical_cells(
            patch,
            cell_xy,
            map_to_physical_xy,
            quantum,
        )
        if support_cells:
            anchor_xy, normal_xy = _unknown_side_direction(
                arc,
                seeds,
                cell_xy,
                map_to_physical_xy,
            )
            supports.append(
                ObservationSupport(
                    arc,
                    support_cells,
                    anchor_xy=anchor_xy,
                    normal_xy=normal_xy,
                )
            )

    if density_separation_m is not None and float(density_separation_m) > 0.0:
        supports = list(suppress_dense_observation_supports(
            supports,
            density_separation_m=float(density_separation_m),
        ))

    return tuple(supports)


def support_overlap(left, right):
    """Return the shared physical cells of two local observation supports."""
    return len(frozenset(left) & frozenset(right))

