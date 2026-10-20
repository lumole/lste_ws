"""Persistent, place-owned frontier work for online exploration.

A structural component is a transient description of free space in one SLAM
snapshot.  A work item is instead an observation boundary owned by the
physical place where it was discovered.  That distinction prevents furniture
from turning one office into a stream of newly minted exploration places.
"""

import math

import numpy as np

from global_frontier_topology_grid import grid_line_reaches_cell
from global_frontier_work_item_components import support_overlap


# A WorkItem answers one semantic question: has this unknown-side boundary
# been observed sufficiently to resolve it?  It intentionally does not carry
# route failure state.  A route reaches a particular safe viewpoint, whereas
# the WorkItem owns the physical observation boundary and can require another
# viewpoint after that route fails.
WORK_UNRESOLVED = "unresolved"
WORK_RESOLVED = "resolved"

# Kept as imports for older callers/tests. They are *not* WorkItem states:
# ``dispatched`` and ``deferred`` are now Attempt lifecycle values.
WORK_UNSEEN = WORK_UNRESOLVED
WORK_DISPATCHED = "dispatched"
WORK_DEFERRED = "deferred"

ATTEMPT_ACTIVE = "active"
ATTEMPT_FAILED = "failed"
ATTEMPT_BLOCKED = "blocked"
ATTEMPT_SUCCEEDED = "succeeded"


class PlaceWorkItemLedger:
    """Keep a small lifecycle ledger for frontier observations per Place."""

    def __init__(self, merge_radius, viewpoint_merge_radius=None, density_radius=None):
        self.merge_radius = max(0.05, float(merge_radius))
        # ``merge_radius`` identifies one physical unknown boundary across
        # SLAM snapshots.  It must not also collapse distinct recovery poses:
        # a failed corner view may need a lateral view well inside that
        # long-term identity footprint.  Keep the old behavior for injected
        # callers that omit the optional value, while production passes the
        # observation-lattice scale explicitly.
        if viewpoint_merge_radius is None:
            viewpoint_merge_radius = self.merge_radius
        self.viewpoint_merge_radius = max(
            0.05, min(self.merge_radius, float(viewpoint_merge_radius))
        )
        # Density suppression radius: prevents clustering multiple unresolved
        # WorkItems within a small physical neighborhood.
        if density_radius is None:
            self.density_radius = self.merge_radius
        else:
            self.density_radius = max(0.0, float(density_radius))
        self._dead_corners = []
        self._next_id = 1
        self._next_attempt_id = 1
        self._items = {}
        self.spawn_odom_xy = (0.0, 0.0)

    @staticmethod
    def _normalise_support(support_cells):
        """Make support signatures safe to retain across map snapshots."""
        if not support_cells:
            return frozenset()
        return frozenset((int(row), int(col)) for row, col in support_cells)

    @staticmethod
    def _normalise_xy(xy):
        if xy is None:
            return None
        try:
            return float(xy[0]), float(xy[1])
        except (IndexError, TypeError, ValueError):
            return None

    @classmethod
    def _normalise_direction(cls, direction):
        """Return a unit direction, or ``None`` for legacy/malformed data."""
        direction = cls._normalise_xy(direction)
        if direction is None:
            return None
        norm = math.hypot(direction[0], direction[1])
        if not math.isfinite(norm) or norm <= 1e-9:
            return None
        return direction[0] / norm, direction[1] / norm

    @classmethod
    def _directions_compatible(cls, left, right):
        """Require two known normals to point into the same unknown half-plane.

        Missing normals are accepted for compatibility with work created by
        older callers.  Once both snapshots provide geometry, an opposite (or
        orthogonal) direction cannot inherit the same physical observation
        task, even when their coarse support lattices overlap.
        """
        left = cls._normalise_direction(left)
        right = cls._normalise_direction(right)
        if left is None or right is None:
            return True
        return left[0] * right[0] + left[1] * right[1] > 0.0

    def _anchors_compatible(self, left, right):
        """Return whether two known physical anchors can share one item."""
        left = self._normalise_xy(left)
        right = self._normalise_xy(right)
        if left is None or right is None:
            # Older records have no anchor.  Their support overlap remains the
            # compatibility path until a physical anchor is reacquired.
            return True
        return math.hypot(left[0] - right[0], left[1] - right[1]) <= self.merge_radius

    def mark_dead_corner(self, anchor_xy):
        """Register an anchor position into the permanent dead corner blacklist."""
        xy = self._normalise_xy(anchor_xy)
        if xy is None:
            return
        if not self.is_dead_corner(xy, radius=0.3):
            self._dead_corners.append((float(xy[0]), float(xy[1])))
        # Also ensure any other unresolved item in this dead corner pocket is resolved
        for it in list(self._items.values()):
            if it.get("state") == WORK_UNRESOLVED:
                it_anchor = self._normalise_xy(it.get("anchor_xy"))
                if it_anchor is not None and math.hypot(xy[0] - it_anchor[0], xy[1] - it_anchor[1]) <= 0.6:
                    it["state"] = WORK_RESOLVED
                    it["last_reason"] = "dead_corner_proximity_abandoned"

    def is_dead_corner(self, anchor_xy, radius=0.6):
        """Return True if anchor_xy is within radius of any blacklisted dead corner."""
        xy = self._normalise_xy(anchor_xy)
        if xy is None:
            return False
        r = float(radius)
        for dead_x, dead_y in self._dead_corners:
            if math.hypot(xy[0] - dead_x, xy[1] - dead_y) <= r:
                return True
        return False

    def record_unreachable_viewpoint(self, item_id, now, threshold=2):
        """Record a tick where no viable viewpoint could be generated for item_id.

        If consecutive unreachable count reaches threshold, resolves and blacklists as dead corner.
        Returns True if item was abandoned as a dead corner.
        """
        item = self._items.get(item_id)
        if item is None or item["state"] != WORK_UNRESOLVED:
            return False
        count = int(item.get("unreachable_viewpoint_count", 0)) + 1
        item["unreachable_viewpoint_count"] = count
        if count >= int(threshold):
            self.resolve(item_id, now, "dead_corner_unreachable")
            anchor = item.get("anchor_xy")
            if anchor is not None:
                self.mark_dead_corner(anchor)
            return True
        return False

    def reset_unreachable_viewpoint(self, item_id):
        """Reset consecutive unreachable count when a viable viewpoint is found."""
        item = self._items.get(item_id)
        if item is not None and "unreachable_viewpoint_count" in item:
            item["unreachable_viewpoint_count"] = 0

    def prune_passed_through_work_items(
        self,
        robot_pose_odom,
        active_work_item_id,
        now,
        distance_threshold=0.4,
        angle_threshold_deg=30.0,
    ):
        """Opportunistically resolve non-active WorkItems that the robot has passed through.

        A WorkItem that is NOT the active goal is resolved if and only if:
        1. Robot is within distance_threshold (0.4m) of its anchor.
        2. Robot's heading matches the WorkItem's normal direction within angle_threshold_deg (30 deg).
        All other WorkItems are strictly preserved (no remote free-space deletion).
        """
        if robot_pose_odom is None:
            return []
        try:
            rx = float(robot_pose_odom[0])
            ry = float(robot_pose_odom[1])
            ryaw = float(robot_pose_odom[2])
        except (TypeError, ValueError, IndexError):
            return []

        dist_limit = float(distance_threshold)
        angle_limit_rad = math.radians(float(angle_threshold_deg))

        resolved_ids = []
        for item in list(self._items.values()):
            if item.get("state") != WORK_UNRESOLVED:
                continue
            # Do NOT apply to the active target being executed
            if active_work_item_id is not None and int(item.get("id", -1)) == int(active_work_item_id):
                continue

            anchor = self._normalise_xy(item.get("anchor_xy"))
            if anchor is None:
                continue

            dist = math.hypot(rx - anchor[0], ry - anchor[1])
            if dist > dist_limit:
                continue

            normal = self._normalise_direction(item.get("normal_xy"))
            if normal is not None:
                target_yaw = math.atan2(normal[1], normal[0])
                angle_diff = abs((ryaw - target_yaw + math.pi) % (2.0 * math.pi) - math.pi)
                if angle_diff > angle_limit_rad:
                    continue

            # Robot has physically passed through this non-active WorkItem with aligned heading!
            item["state"] = WORK_RESOLVED
            item["settled_at"] = float(now)
            item["last_reason"] = "opportunistic_pass_through"
            item["resolved_reason"] = "opportunistic_pass_through"
            if item.get("active_attempt_id") is not None:
                for att in item.get("attempts", []):
                    if att.get("state") == "active":
                        att["state"] = "succeeded"
                        att["reason"] = "opportunistic_pass_through"
                item["active_attempt_id"] = None
            resolved_ids.append(int(item["id"]))

        return resolved_ids

    def _bind_spawn_distance(
        self,
        item,
        spawn_odom_xy=None,
        free=None,
        map_msg=None,
        physical_to_map=None,
        geodesic_field=None,
    ):
        """Compute and permanently store spawn_distance on a WorkItem if not set."""
        if item.get("spawn_distance") is not None:
            return item["spawn_distance"]

        spawn_xy = (
            spawn_odom_xy
            if spawn_odom_xy is not None
            else getattr(self, "spawn_odom_xy", None)
        )
        if spawn_xy is None:
            spawn_xy = (0.0, 0.0)

        anchor = self._normalise_xy(item.get("anchor_xy"))
        if anchor is None:
            x_val = item.get("x")
            y_val = item.get("y")
            if x_val is not None and y_val is not None:
                anchor = (float(x_val), float(y_val))
            else:
                return None

        # Try geodesic distance calculation if free grid is available
        if free is not None:
            try:
                from global_frontier_grid import compute_geodesic_distance_to_work_item
                dist = compute_geodesic_distance_to_work_item(
                    free=free,
                    spawn_xy=spawn_xy,
                    target_item_or_xy=item,
                    message=map_msg,
                    physical_to_map=physical_to_map,
                    geodesic_field=geodesic_field,
                )
                if dist is not None and math.isfinite(dist):
                    item["spawn_distance"] = float(dist)
                    return item["spawn_distance"]
            except Exception:
                pass

        # Fallback to Euclidean distance if free grid is not supplied or geodesic path unreachable
        try:
            euc_dist = math.hypot(
                float(anchor[0]) - float(spawn_xy[0]),
                float(anchor[1]) - float(spawn_xy[1]),
            )
            item["spawn_distance"] = float(euc_dist)
            return item["spawn_distance"]
        except (TypeError, ValueError):
            return None

    def _new_item(
        self, place_id, support_cells, now, xy=None,
        reason="unknown_boundary_discovered", anchor_xy=None, normal_xy=None,
        spawn_distance=None,
    ):
        xy = self._normalise_xy(xy)
        item = {
            "id": self._next_id,
            "place_id": int(place_id),
            "x": None if xy is None else xy[0],
            "y": None if xy is None else xy[1],
            "anchor_xy": self._normalise_xy(anchor_xy),
            "normal_xy": self._normalise_direction(normal_xy),
            "support_cells": support_cells,
            # Semantic state: only observation evidence may resolve this.
            "state": WORK_UNRESOLVED,
            "last_reason": str(reason),
            "updated_at": float(now),
            "attempts": [],
            # A route can be rejected before a controller Attempt exists. Keep
            # that geometric admission fact separate from execution history so
            # the same physical boundary can be reprojected from another
            # viewpoint without pretending that the robot already tried it.
            "route_rejections": [],
            "active_attempt_id": None,
            "spawn_distance": float(spawn_distance) if spawn_distance is not None else None,
        }
        self._next_id += 1
        self._items[int(item["id"])] = item
        if item.get("spawn_distance") is None and getattr(self, "spawn_odom_xy", None) is not None:
            self._bind_spawn_distance(item)
        return item

    @staticmethod
    def _active_attempt(item):
        active_id = item.get("active_attempt_id")
        if active_id is None:
            return None
        for attempt in item.get("attempts", ()):
            if int(attempt["id"]) == int(active_id):
                return attempt
        # Ledger corruption must not leave a semantic item locked forever.
        item["active_attempt_id"] = None
        return None

    def _attempted_same_viewpoint(self, item, viewpoint_xy):
        """Return a failed attempt at this physical viewpoint, if any.

        The same odom-frame viewpoint after a SLAM correction is still the
        same physical route attempt.  The existing WorkItem merge footprint
        supplies the only tolerance: this is identity association, not a
        new controller tuning knob.
        """
        viewpoint_xy = self._normalise_xy(viewpoint_xy)
        if viewpoint_xy is None:
            return None
        for attempt in item.get("attempts", ()):
            if attempt.get("state") not in (ATTEMPT_FAILED, ATTEMPT_BLOCKED):
                continue
            attempt_xy = self._normalise_xy(attempt.get("viewpoint_xy"))
            if attempt_xy is None:
                continue
            if math.hypot(
                viewpoint_xy[0] - attempt_xy[0],
                viewpoint_xy[1] - attempt_xy[1],
            ) <= self.viewpoint_merge_radius:
                return attempt
        return None

    @staticmethod
    def _normalise_epoch(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _route_rejected_same_viewpoint(
        self, item, viewpoint_xy, map_epoch=None,
    ):
        """Return a pre-dispatch route rejection at this physical viewpoint."""
        viewpoint_xy = self._normalise_xy(viewpoint_xy)
        if viewpoint_xy is None:
            return None
        map_epoch = self._normalise_epoch(map_epoch)
        for rejection in item.get("route_rejections", ()):
            rejection_epoch = self._normalise_epoch(rejection.get("map_epoch"))
            # Route-admission facts belong to one short-lived map snapshot.
            # Keep the record for replay, but never let an older epoch block a
            # newly projected viewpoint after SLAM/costmap evidence changes.
            if (
                rejection_epoch is not None
                and map_epoch is not None
                and rejection_epoch != map_epoch
            ):
                continue
            rejection_xy = self._normalise_xy(rejection.get("viewpoint_xy"))
            if rejection_xy is None:
                continue
            if math.hypot(
                viewpoint_xy[0] - rejection_xy[0],
                viewpoint_xy[1] - rejection_xy[1],
            ) <= self.viewpoint_merge_radius:
                return rejection
        return None

    def reconcile(
        self,
        place_id,
        supports,
        now,
        spawn_odom_xy=None,
        free=None,
        map_msg=None,
        physical_to_map=None,
    ):
        """Match current frontier arcs to persistent physical work items.

        A support is an unknown-side region in odom coordinates, not a goal
        pose.  Any later arc overlapping a resolved support is its descendant
        and therefore cannot be dispatched again merely because the frontier
        moved farther into the same unobserved direction.
        """
        if place_id is None:
            return {}, {}
        if spawn_odom_xy is not None:
            self.spawn_odom_xy = self._normalise_xy(spawn_odom_xy)
        resolved_spawn_xy = getattr(self, "spawn_odom_xy", None)

        geodesic_field = None
        if free is not None and resolved_spawn_xy is not None:
            try:
                from global_frontier_grid import get_or_compute_spawn_geodesic_field
                geodesic_field = get_or_compute_spawn_geodesic_field(
                    free=free,
                    spawn_xy=resolved_spawn_xy,
                    message=map_msg,
                    physical_to_map=physical_to_map,
                )
            except Exception:
                geodesic_field = None

        cell_to_item = {}
        details = {}
        for support in supports:
            support_cells = self._normalise_support(support.support_cells)
            if not support_cells:
                continue
            support_anchor = self._normalise_xy(
                getattr(support, "anchor_xy", None)
            )
            if support_anchor is not None and self.is_dead_corner(support_anchor, radius=0.6):
                continue
            matches = []
            for item in self._items.values():
                prior_state = item["state"]
                overlap = support_overlap(support_cells, item.get("support_cells", ()))
                if not self._directions_compatible(
                    getattr(support, "normal_xy", None), item.get("normal_xy")
                ):
                    continue
                prior_anchor = self._normalise_xy(item.get("anchor_xy"))
                if prior_anchor is not None and support_anchor is not None:
                    # Physical anchors are the durable identity. Support
                    # cells belong to a transient SLAM snapshot and may be
                    # completely disjoint after map relabelling.
                    anchor_match = self._anchors_compatible(
                        prior_anchor, support_anchor
                    )
                    # For a resolved item, it can ONLY match if anchor_match is True
                    # (within merge_radius, preventing SLAM jitter from re-minting).
                    # An advancing frontier beyond merge_radius into unexplored space
                    # is a NEW exploration boundary and must NOT be swallowed.
                    if not anchor_match:
                        continue
                    identity_match = True
                    match_quality = 2
                else:
                    identity_match = bool(overlap)
                    match_quality = 0
                if identity_match:
                    matches.append((
                        match_quality,
                        overlap,
                        -int(item["id"]),
                        item,
                    ))
            if matches:
                _anchor_match, _overlap, _negative_id, item = max(
                    matches, key=lambda value: value[:3]
                )
                prior_state = item["state"]
                item["support_cells"] = frozenset(item["support_cells"] | support_cells)
                support_anchor = self._normalise_xy(
                    getattr(support, "anchor_xy", None)
                )
                # The first physical anchor is the WorkItem identity.
                if item.get("anchor_xy") is None and support_anchor is not None:
                    item["anchor_xy"] = support_anchor
                support_direction = self._normalise_direction(
                    getattr(support, "normal_xy", None)
                )
                if item.get("normal_xy") is None and support_direction is not None:
                    item["normal_xy"] = support_direction
                item["updated_at"] = float(now)
                self._bind_spawn_distance(
                    item,
                    resolved_spawn_xy,
                    free,
                    map_msg,
                    physical_to_map,
                    geodesic_field,
                )
                match = (
                    "resolved_descendant"
                    if prior_state == WORK_RESOLVED
                    else "inherited"
                )
            else:
                density_match = None
                if support_anchor is not None and self.density_radius > 0:
                    candidates_in_radius = []
                    is_fragment = len(support_cells) <= 4
                    effective_density = self.density_radius * 1.5 if is_fragment else self.density_radius
                    for cand_item in self._items.values():
                        if cand_item["state"] != WORK_UNRESOLVED:
                            continue
                        item_anchor = self._normalise_xy(cand_item.get("anchor_xy"))
                        if item_anchor is None:
                            continue
                        dist = math.hypot(
                            support_anchor[0] - item_anchor[0],
                            support_anchor[1] - item_anchor[1],
                        )
                        if dist <= effective_density:
                            candidates_in_radius.append((
                                dist,
                                len(cand_item.get("support_cells", ())),
                                -int(cand_item["id"]),
                                cand_item,
                            ))
                    if candidates_in_radius:
                        candidates_in_radius.sort(key=lambda c: (c[0], -c[1], c[2]))
                        density_match = candidates_in_radius[0][3]

                if density_match is not None:
                    item = density_match
                    item["support_cells"] = frozenset(item["support_cells"] | support_cells)
                    item["updated_at"] = float(now)
                    if support_anchor is not None:
                        item["anchor_xy"] = support_anchor
                    if getattr(support, "normal_xy", None) is not None:
                        item["normal_xy"] = self._normalise_direction(support.normal_xy)
                    self._bind_spawn_distance(
                        item,
                        resolved_spawn_xy,
                        free,
                        map_msg,
                        physical_to_map,
                        geodesic_field,
                    )
                    match = "density_absorbed"
                else:
                    item = self._new_item(
                        place_id or 1,
                        support_cells,
                        now,
                        anchor_xy=getattr(support, "anchor_xy", None),
                        normal_xy=getattr(support, "normal_xy", None),
                    )
                    self._bind_spawn_distance(
                        item,
                        resolved_spawn_xy,
                        free,
                        map_msg,
                        physical_to_map,
                        geodesic_field,
                    )
                    match = "created"
            item_id = int(item["id"])
            detail = {
                "id": item_id,
                "state": item["state"],
                "match": match,
                "support_cell_count": len(item.get("support_cells", support_cells)),
                "anchor_xy": item.get("anchor_xy"),
                "normal_xy": item.get("normal_xy"),
                "spawn_distance": item.get("spawn_distance"),
            }
            details[item_id] = detail
            for cell in support.frontier_cells:
                cell_to_item[(int(cell[0]), int(cell[1]))] = item_id

        if resolved_spawn_xy is not None:
            for it in self._items.values():
                if it.get("spawn_distance") is None:
                    self._bind_spawn_distance(
                        it,
                        resolved_spawn_xy,
                        free,
                        map_msg,
                        physical_to_map,
                        geodesic_field,
                    )

        return cell_to_item, details

    def suppress_dense_unresolved_items(self, place_id=None):
        """Merge multiple unresolved items in the same Place within density_radius.

        Keeps the item with the most support cells and settles or absorbs
        redundant nearby items into it.
        """
        if self.density_radius <= 0:
            return 0
        unresolved = [
            item for item in self._items.values()
            if (place_id is None or int(item["place_id"]) == int(place_id))
            and item["state"] == WORK_UNRESOLVED
            and self._normalise_xy(item.get("anchor_xy")) is not None
        ]
        unresolved.sort(
            key=lambda it: (len(it.get("support_cells", ())), -int(it["id"])),
            reverse=True,
        )
        suppressed_count = 0
        kept = []
        for it in unresolved:
            anchor = self._normalise_xy(it.get("anchor_xy"))
            merged_into = None
            for k in kept:
                k_anchor = self._normalise_xy(k.get("anchor_xy"))
                dist = math.hypot(anchor[0] - k_anchor[0], anchor[1] - k_anchor[1])
                if dist <= self.density_radius:
                    merged_into = k
                    break
            if merged_into is not None:
                merged_into["support_cells"] = frozenset(
                    merged_into["support_cells"] | it.get("support_cells", ())
                )
                it["state"] = WORK_RESOLVED
                it["last_reason"] = "density_suppressed_merged_into_%d" % int(merged_into["id"])
                suppressed_count += 1
            else:
                kept.append(it)
        return suppressed_count


    def work_item_is_available(self, item_id):
        """Return whether an unresolved boundary has no active Attempt."""
        item = self._items.get(item_id)
        return (
            item is not None
            and item["state"] == WORK_UNRESOLVED
            and self._active_attempt(item) is None
        )

    def viewpoint_is_available(self, item_id, viewpoint_xy, map_epoch=None):
        """Return whether this is a new safe viewpoint for an unresolved item."""
        item = self._items.get(item_id)
        if item is None or not self.work_item_is_available(item_id):
            return False
        return (
            self._attempted_same_viewpoint(item, viewpoint_xy) is None
            and self._route_rejected_same_viewpoint(
                item, viewpoint_xy, map_epoch=map_epoch,
            ) is None
        )

    def route_rejection_for_viewpoint(
        self, item_id, viewpoint_xy, map_epoch=None,
    ):
        """Return the current-epoch pre-dispatch rejection, if any."""
        item = self._items.get(item_id)
        if item is None:
            return None
        return self._route_rejected_same_viewpoint(
            item, viewpoint_xy, map_epoch=map_epoch,
        )

    def has_failed_viewpoint(self, item_id, map_epoch=None):
        """Return whether an unresolved item needs a different viewpoint.

        This is WorkItem state, not a retry counter. The selector may promote
        another map-safe viewpoint after either an executed viewpoint ended in
        ``failed``/``blocked`` or Navfn rejected the viewpoint before dispatch.
        """
        item = self._items.get(item_id)
        if item is None or item["state"] != WORK_UNRESOLVED:
            return False
        return any(
            attempt.get("state") in (ATTEMPT_FAILED, ATTEMPT_BLOCKED)
            for attempt in item.get("attempts", ())
        ) or any(
            self._route_rejected_same_viewpoint(
                item, rejection.get("viewpoint_xy"), map_epoch=map_epoch,
            ) is not None
            for rejection in item.get("route_rejections", ())
        )

    def record_route_rejection(
        self, item_id, now, viewpoint_xy, map_goal=None, reason="",
        map_epoch=None, plan_endpoint=None,
    ):
        """Record a route-admission failure without opening an Attempt.

        Navfn/costmap validation happens before route activation, so it is not
        an execution failure and must not consume an Attempt ID. The durable
        WorkItem still needs to remember that exact physical standoff until a
        different viewpoint on the same unknown boundary is compiled.
        """
        item = self._items.get(item_id)
        if item is None or item.get("state") != WORK_UNRESOLVED:
            return None, None
        if self._active_attempt(item) is not None:
            return item, None
        viewpoint_xy = self._normalise_xy(viewpoint_xy)
        if viewpoint_xy is None:
            return item, None
        existing = self._route_rejected_same_viewpoint(
            item, viewpoint_xy, map_epoch=map_epoch,
        )
        if existing is not None:
            return item, existing
        rejection = {
            "viewpoint_xy": viewpoint_xy,
            "map_goal": self._normalise_xy(map_goal),
            "map_epoch": self._normalise_epoch(map_epoch),
            "plan_endpoint": self._normalise_xy(plan_endpoint),
            "reason": str(reason),
            "recorded_at": float(now),
        }
        item.setdefault("route_rejections", []).append(rejection)
        item["last_reason"] = str(reason)
        item["updated_at"] = float(now)
        return item, rejection

    def active_attempt_id(self, item_id):
        item = self._items.get(item_id)
        if item is None:
            return None
        attempt = self._active_attempt(item)
        return None if attempt is None else int(attempt["id"])

    def active_attempt_viewpoint(self, item_id):
        """Return the active Attempt's immutable physical viewpoint."""
        item = self._items.get(item_id)
        if item is None:
            return None
        attempt = self._active_attempt(item)
        if attempt is None:
            return None
        return self._normalise_xy(attempt.get("viewpoint_xy"))

    def dispatch_existing(
        self, item_id, now, viewpoint_xy=None, map_goal=None, route_kind=None,
        map_epoch=None,
    ):
        """Start one Attempt without changing the WorkItem semantic state."""
        item = self._items.get(item_id)
        if item is None or not self.work_item_is_available(item_id):
            return None
        if viewpoint_xy is not None and not self.viewpoint_is_available(
            item_id, viewpoint_xy, map_epoch=map_epoch,
        ):
            return None
        attempt = {
            "id": self._next_attempt_id,
            "state": ATTEMPT_ACTIVE,
            "viewpoint_xy": self._normalise_xy(viewpoint_xy),
            "map_goal": self._normalise_xy(map_goal),
            "route_kind": None if route_kind is None else str(route_kind),
            "map_epoch": self._normalise_epoch(map_epoch),
            "started_at": float(now),
            "ended_at": None,
            "reason": "route_dispatched",
        }
        self._next_attempt_id += 1
        item["attempts"].append(attempt)
        item["active_attempt_id"] = int(attempt["id"])
        item["last_reason"] = "route_dispatched"
        item["updated_at"] = float(now)
        return int(item["id"])

    def _nearby(self, place_id, xy):
        if place_id is None or xy is None:
            return None
        try:
            x, y = float(xy[0]), float(xy[1])
        except (IndexError, TypeError, ValueError):
            return None
        best = None
        best_distance = float("inf")
        for item in self._items.values():
            if int(item["place_id"]) != int(place_id):
                continue
            if item.get("x") is None or item.get("y") is None:
                continue
            distance = math.hypot(x - item["x"], y - item["y"])
            if distance <= self.merge_radius and distance < best_distance:
                best = item
                best_distance = distance
        return best

    def candidate_is_available(self, place_id, xy):
        """Return whether a nearby WorkItem permits this physical viewpoint."""
        item = self._nearby(place_id, xy)
        return item is None or self.viewpoint_is_available(item["id"], xy)

    def dispatch(self, place_id, xy, now, map_epoch=None):
        """Create or reserve the one work item represented by ``xy``."""
        if place_id is None or xy is None:
            return None
        item = self._nearby(place_id, xy)
        if item is None:
            item = self._new_item(
                place_id, frozenset(), now, xy=xy, reason="discovered",
            )
        return self.dispatch_existing(
            int(item["id"]),
            now,
            viewpoint_xy=xy,
            map_goal=xy,
            route_kind="frontier_endpoint",
            map_epoch=map_epoch,
        )

    def _finish_active_attempt(self, item, state, now, reason):
        attempt = self._active_attempt(item)
        if attempt is None:
            return None
        attempt["state"] = str(state)
        attempt["ended_at"] = float(now)
        attempt["reason"] = str(reason)
        item["active_attempt_id"] = None
        item["last_reason"] = str(reason)
        item["updated_at"] = float(now)
        return attempt

    def resolve(self, item_id, now, reason, attempt_id=None):
        """Resolve work only from observation evidence or permanent dead corner abandonment."""
        item = self._items.get(item_id)
        if item is None:
            return None
        active = self._active_attempt(item)
        if active is not None:
            # A terminal without ownership metadata is ambiguous once a
            # replacement Attempt is active.  Leave that Attempt unresolved;
            # only its exact lease may settle the semantic WorkItem, unless
            # it is a dead corner abandonment where attempt_id may be omitted.
            is_dead_corner = "dead_corner" in str(reason)
            if not is_dead_corner and attempt_id is None:
                return None
            if attempt_id is not None and int(active["id"]) != int(attempt_id):
                return None
            attempt_state = ATTEMPT_FAILED if is_dead_corner else ATTEMPT_SUCCEEDED
            self._finish_active_attempt(item, attempt_state, now, reason)
        elif attempt_id is not None:
            return None
        item["state"] = WORK_RESOLVED
        item["last_reason"] = str(reason)
        item["updated_at"] = float(now)
        if "dead_corner" in str(reason):
            anchor = item.get("anchor_xy")
            if anchor is not None:
                self.mark_dead_corner(anchor)
        return item

    def fail_attempt(self, item_id, now, reason, state=ATTEMPT_FAILED, attempt_id=None):
        """End one bad route while keeping its observation task unresolved."""
        if state not in (ATTEMPT_FAILED, ATTEMPT_BLOCKED):
            raise ValueError("failed or blocked attempt state required")
        item = self._items.get(item_id)
        if item is None or item["state"] != WORK_UNRESOLVED:
            return None, None
        active = self._active_attempt(item)
        if active is None or (
            attempt_id is not None and int(active["id"]) != int(attempt_id)
        ):
            return item, None
        return item, self._finish_active_attempt(item, state, now, reason)

    def settle(self, item_id, state, now, reason):
        """Compatibility facade for old lifecycle callers.

        ``deferred`` used to retire the semantic task.  It now records only a
        failed viewpoint Attempt; callers must use ``resolved`` after real
        observation evidence.
        """
        if state == WORK_RESOLVED:
            return self.resolve(
                item_id,
                now,
                reason,
                attempt_id=self.active_attempt_id(item_id),
            )
        if state == WORK_DEFERRED:
            item, _attempt = self.fail_attempt(item_id, now, reason)
            return item
        raise ValueError("resolved work item or deferred attempt required")

    def unresolved_count(self, place_id):
        """Return work still eligible for this physical place."""
        return sum(
            1
            for item in self._items.values()
            if int(item["place_id"]) == int(place_id)
            and item["state"] == WORK_UNRESOLVED
        )

    def unresolved_observation_count(self, place_id, portal_probe_ledger=None):
        """Count ordinary local WorkItems, excluding doorway-owned items.

        A Portal probe may intentionally keep its associated WorkItem
        unresolved while it moves through ``source_arrived`` and
        ``destination_active`` evidence.  That is an outward edge obligation,
        not unfinished room coverage.  Keeping this distinction in the ledger
        prevents the Place phase from waiting forever for a probe to become a
        Portal certificate.
        """
        try:
            place_id = int(place_id)
        except (TypeError, ValueError):
            return 0
        probe_for_work = (
            None if portal_probe_ledger is None
            else getattr(portal_probe_ledger, "probe_for_work_item", None)
        )
        count = 0
        for item in self._items.values():
            if (
                int(item.get("place_id", -1)) != place_id
                or item.get("state") != WORK_UNRESOLVED
            ):
                continue
            if callable(probe_for_work) and probe_for_work(item.get("id")) is not None:
                continue
            count += 1
        return count

    def support_for(self, item_id):
        """Return one item's support lattice in its physical frame, read-only."""
        item = self._items.get(item_id)
        if item is None:
            return frozenset()
        return frozenset(item.get("support_cells") or ())

    def anchor_for(self, item_id):
        """Return one item's physical unknown-side anchor, read-only."""
        item = self._items.get(item_id)
        if item is None:
            return None
        anchor = item.get("anchor_xy")
        if anchor is None:
            return None
        try:
            return float(anchor[0]), float(anchor[1])
        except (IndexError, TypeError, ValueError):
            return None

    def normal_for(self, item_id):
        """Return one item's physical unknown-side normal, read-only."""
        item = self._items.get(item_id)
        if item is None:
            return None
        normal = item.get("normal_xy")
        if normal is None:
            return None
        try:
            return float(normal[0]), float(normal[1])
        except (IndexError, TypeError, ValueError):
            return None

    def restrict_support_to_observed(self, item_id, is_unknown):
        """Keep only the observed part of a support as resolution evidence.

        Resolving an item records its whole support as observation evidence,
        and the reconcile descendant rule later suppresses any support that
        overlaps a resolved item's cells.  If the resolve left part of the
        patch unseen (the endpoint's view could not cover it), storing that
        part as evidence would make the genuinely unobserved remainder
        permanently undispatchable.  Record only what the map proves was
        observed; report how much was still unknown.
        """
        item = self._items.get(item_id)
        if item is None:
            return 0
        cells = frozenset(item.get("support_cells") or ())
        observed = frozenset(cell for cell in cells if not is_unknown(cell))
        item["support_cells"] = observed
        item["unobserved_support_cells"] = cells - observed
        return len(item["unobserved_support_cells"])

    def reattribute_work_item(self, item_id, new_place_id, now=None):
        """Move one WorkItem ownership to a newly discovered or correct Place."""
        item = self._items.get(item_id)
        if item is None or new_place_id is None:
            return None
        item["place_id"] = int(new_place_id)
        if now is not None:
            item["updated_at"] = float(now)
        item["last_reason"] = "place_reattributed"
        return item

    def reattribute_items_near(
        self, old_place_id, new_place_id, physical_xy, radius=None, now=None,
    ):
        """Move WorkItems near physical_xy from old_place_id to new_place_id."""
        if old_place_id is None or new_place_id is None:
            return []
        try:
            old_place_id = int(old_place_id)
            new_place_id = int(new_place_id)
        except (TypeError, ValueError):
            return []
        if old_place_id == new_place_id:
            return []
        physical_xy = self._normalise_xy(physical_xy)
        if physical_xy is None:
            return []
        if radius is None:
            radius = self.merge_radius * 2.0
        else:
            radius = float(radius)
        reattributed = []
        for item in self._items.values():
            if int(item.get("place_id", -1)) != old_place_id:
                continue
            item_xy = self._normalise_xy(item.get("anchor_xy")) or (
                self._normalise_xy((item.get("x"), item.get("y")))
            )
            if item_xy is None:
                continue
            dist = math.hypot(physical_xy[0] - item_xy[0], physical_xy[1] - item_xy[1])
            if dist <= radius:
                item["place_id"] = new_place_id
                item["last_reason"] = "place_reattributed"
                if now is not None:
                    item["updated_at"] = float(now)
                reattributed.append(int(item["id"]))
        return reattributed

    def snapshot(self):
        """Return a read-only, JSON-safe view for graph-level planning.

        The graph planner must inspect durable WorkItems without reaching into
        this ledger's mutable records.  Copying the small lifecycle fields here
        also makes a planning decision replayable without exposing support
        geometry or allowing a caller to mutate the live attempt state.
        """
        result = []
        for item_id in sorted(self._items):
            item = self._items[item_id]
            attempts = []
            for attempt in item.get("attempts", ()):
                attempts.append(
                    {
                        "id": int(attempt["id"]),
                        "state": str(attempt.get("state", "")),
                        "viewpoint_xy": (
                            None
                            if attempt.get("viewpoint_xy") is None
                            else [
                                float(attempt["viewpoint_xy"][0]),
                                float(attempt["viewpoint_xy"][1]),
                            ]
                        ),
                        "map_goal": (
                            None
                            if attempt.get("map_goal") is None
                            else [
                                float(attempt["map_goal"][0]),
                                float(attempt["map_goal"][1]),
                            ]
                        ),
                        "route_kind": attempt.get("route_kind"),
                        "map_epoch": attempt.get("map_epoch"),
                        "reason": str(attempt.get("reason", "")),
                    }
                )
            route_rejections = []
            for rejection in item.get("route_rejections", ()):
                route_rejections.append(
                    {
                        "viewpoint_xy": (
                            None
                            if rejection.get("viewpoint_xy") is None
                            else [
                                float(rejection["viewpoint_xy"][0]),
                                float(rejection["viewpoint_xy"][1]),
                            ]
                        ),
                        "map_goal": (
                            None
                            if rejection.get("map_goal") is None
                            else [
                                float(rejection["map_goal"][0]),
                                float(rejection["map_goal"][1]),
                            ]
                        ),
                        "map_epoch": rejection.get("map_epoch"),
                        "plan_endpoint": (
                            None
                            if rejection.get("plan_endpoint") is None
                            else [
                                float(rejection["plan_endpoint"][0]),
                                float(rejection["plan_endpoint"][1]),
                            ]
                        ),
                        "reason": str(rejection.get("reason", "")),
                        "recorded_at": float(rejection.get("recorded_at", 0.0)),
                    }
                )
            result.append(
                {
                    "id": int(item["id"]),
                    "place_id": int(item["place_id"]),
                    "state": str(item.get("state", WORK_UNRESOLVED)),
                    "x": item.get("x"),
                    "y": item.get("y"),
                    "anchor_xy": item.get("anchor_xy"),
                    "normal_xy": item.get("normal_xy"),
                    "support_cell_count": len(item.get("support_cells", ())),
                    "active_attempt_id": (
                        None
                        if item.get("active_attempt_id") is None
                        else int(item["active_attempt_id"])
                    ),
                    "updated_at": float(item.get("updated_at", 0.0)),
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "route_rejection_count": len(route_rejections),
                    "route_rejections": route_rejections,
                    "spawn_distance": item.get("spawn_distance"),
                }
            )
        return result


class GlobalFrontierWorkItemLifecycleMixin:
    """Bind pure work-item transitions to one active frontier lease."""

    def work_item_physical_xy(self, message, x, y):
        """Project an endpoint once; map coordinates are only a TF fallback."""
        header = getattr(message, "header", None)
        map_frame = getattr(header, "frame_id", "map") or "map"
        transform = getattr(self, "transform_xy", None)
        if transform is not None:
            physical = transform("odom", map_frame, float(x), float(y))
            if physical is not None:
                return physical
        return float(x), float(y)

    def ensure_current_physical_place(
        self, message, components, known_free, robot_map, now,
    ):
        """Bootstrap the one Place occupied at process start.

        This is the sole non-portal place creation: before the first doorway
        crossing the robot is necessarily somewhere.  Every later physical
        Place is created by the certified portal-arrival transaction.
        """
        current_id = getattr(self, "current_physical_place_id", None)
        if current_id is not None and self.region_memory.by_id(current_id) is not None:
            bind_pending = getattr(self, "_bind_pending_semantic_observations", None)
            if bind_pending is not None:
                bind_pending(int(current_id))
            bind_target = getattr(
                self, "_bind_pending_target_belief_observations", None
            )
            if bind_target is not None:
                bind_target(int(current_id))
            return self.region_memory.by_id(current_id)
        component = self.component_at_map_position(
            message, components, known_free, robot_map[0], robot_map[1],
        )
        if component is None:
            return None
        region, _tier = self.region_memory.activate(
            robot_map[0],
            robot_map[1],
            0.0,
            now,
            component=component,
            physical_xy=(
                None
                if getattr(self, "pose_odom", None) is None
                else (float(self.pose_odom.x), float(self.pose_odom.y))
            ),
        )
        if region is not None:
            self.current_physical_place_id = int(region["id"])
            physical_source = (
                None
                if getattr(self, "pose_odom", None) is None
                else (
                    round(float(self.pose_odom.x), 3),
                    round(float(self.pose_odom.y), 3),
                )
            )
            self.physical_place_bootstrap_place_id = int(region["id"])
            self.physical_place_bootstrap_anchor = physical_source
            bind_pending = getattr(self, "_bind_pending_semantic_observations", None)
            if bind_pending is not None:
                bind_pending(int(region["id"]))
            bind_target = getattr(
                self, "_bind_pending_target_belief_observations", None
            )
            if bind_target is not None:
                bind_target(int(region["id"]))
            self.publish_status(
                "physical_place_bootstrapped",
                region_id=int(region["id"]),
                source=[round(float(robot_map[0]), 3), round(float(robot_map[1]), 3)],
                physical_source=(
                    None
                    if physical_source is None
                    else list(physical_source)
                ),
            )
        return region

    def dispatch_active_work_item(self, message, selection, region, now):
        """Reserve local work when a route is committed, before it is published.

        The ledger must close immediately so another planning pass cannot pick
        the same WorkItem.  Its *observable* dispatch event is intentionally
        delayed until the first route command has been published: selected
        candidates are not yet executed work, and a ROS observer may attach
        between selection and the first controller command.
        """
        self.active_work_item_id = None
        self.active_work_item_attempt_id = None
        self.active_work_item_place_id = None
        self.active_work_item_goal = None
        self.active_work_item_route_kind = None
        self.active_work_item_dispatch_announced = False
        if not getattr(self, "work_item_memory_enabled", True):
            return None
        if (
            region is None
            or selection.route_kind != "frontier_endpoint"
            or selection.place_hops != 0
        ):
            return None
        viewpoint_xy = self.work_item_physical_xy(
            message, selection.x, selection.y,
        )
        component = getattr(selection, "component", None)
        map_epoch = (
            component.get("epoch")
            if isinstance(component, dict) else getattr(component, "epoch", None)
        )
        item_id = getattr(selection, "work_item_id", None)
        if item_id is not None:
            item_id = self.place_work_items.dispatch_existing(
                item_id,
                now,
                viewpoint_xy=viewpoint_xy,
                map_goal=(selection.x, selection.y),
                route_kind=selection.route_kind,
                map_epoch=map_epoch,
            )
        else:
            # Compatibility fallback for callers that do not yet build a
            # frontier-arc snapshot (for example a narrow test fixture).
            item_id = self.place_work_items.dispatch(
                int(region["id"]),
                viewpoint_xy,
                now,
                map_epoch=map_epoch,
            )
        self.active_work_item_id = item_id
        if item_id is not None:
            self.active_work_item_attempt_id = self.place_work_items.active_attempt_id(
                item_id,
            )
            self.active_work_item_place_id = int(region["id"])
            self.active_work_item_goal = (float(selection.x), float(selection.y))
            self.active_work_item_route_kind = str(selection.route_kind)
        return item_id

    def record_work_item_route_rejection(
        self, message, candidate, now, reason, *, map_epoch=None,
        plan_endpoint=None,
    ):
        """Persist a pre-dispatch Navfn/costmap rejection for one WorkItem."""
        if not getattr(self, "work_item_memory_enabled", True):
            return None
        route = getattr(candidate, "route", candidate)
        try:
            item_id = int(route[12])
            map_goal = (float(route[2]), float(route[3]))
        except (IndexError, TypeError, ValueError):
            return None
        physical = self.work_item_physical_xy(message, map_goal[0], map_goal[1])
        item, rejection = self.place_work_items.record_route_rejection(
            item_id,
            now,
            physical,
            map_goal=map_goal,
            reason=reason,
            map_epoch=map_epoch,
            plan_endpoint=plan_endpoint,
        )
        if item is None or rejection is None:
            return rejection
        publish = getattr(self, "publish_status", None)
        if callable(publish):
            publish(
                "work_item_viewpoint_route_rejected",
                work_item_id=int(item_id),
                place_id=int(item["place_id"]),
                viewpoint_xy=[round(float(value), 3) for value in physical],
                map_goal=[round(float(value), 3) for value in map_goal],
                map_epoch=map_epoch,
                plan_endpoint=(
                    None
                    if plan_endpoint is None
                    else [round(float(value), 3) for value in plan_endpoint]
                ),
                reason=str(reason),
                attempt_state="not_started",
            )
        return rejection

    def reserve_prefetched_work_item(self, message, x, y, region, now):
        """Reserve a promoted same-Place item through the normal capability gate.

        Prefetch promotion is only a route-continuity optimization.  It must
        not create Place or WorkItem state for a geometry-only baseline just
        because the candidate happened to be cached before its predecessor
        completed.
        """
        self.active_work_item_id = None
        self.active_work_item_attempt_id = None
        self.active_work_item_place_id = None
        self.active_work_item_goal = None
        self.active_work_item_route_kind = None
        self.active_work_item_dispatch_announced = False
        if not getattr(self, "work_item_memory_enabled", True) or region is None:
            return None
        ledger = getattr(self, "place_work_items", None)
        if ledger is None:
            return None
        viewpoint_xy = self.work_item_physical_xy(message, x, y)
        prefetched_item_id = getattr(self, "prefetched_work_item_id", None)
        if prefetched_item_id is not None:
            item_id = ledger.dispatch_existing(
                prefetched_item_id,
                now,
                viewpoint_xy=viewpoint_xy,
                map_goal=(x, y),
                route_kind="frontier_endpoint",
            )
        else:
            item_id = ledger.dispatch(
                int(region["id"]),
                viewpoint_xy,
                now,
            )
        self.active_work_item_id = item_id
        if item_id is not None:
            self.active_work_item_attempt_id = ledger.active_attempt_id(item_id)
            self.active_work_item_place_id = int(region["id"])
            self.active_work_item_goal = (float(x), float(y))
            self.active_work_item_route_kind = "frontier_endpoint"
        return item_id

    def announce_active_work_item_dispatch(self):
        """Publish one lifecycle start after the command becomes executable.

        This is deliberately idempotent because a single mission endpoint may
        be represented by several connector commands while online SLAM
        refreshes.  The ledger itself was already reserved at selection time;
        this method only creates the auditable execution interval.
        """
        item_id = getattr(self, "active_work_item_id", None)
        if (
            item_id is None
            or getattr(self, "active_work_item_dispatch_announced", False)
        ):
            return False
        place_id = getattr(self, "active_work_item_place_id", None)
        goal = getattr(self, "active_work_item_goal", None)
        if place_id is None or goal is None:
            return False
        try:
            x, y = float(goal[0]), float(goal[1])
        except (IndexError, TypeError, ValueError):
            return False
        self.publish_work_item_dispatch(
            item_id,
            getattr(self, "active_work_item_attempt_id", None),
            {"id": int(place_id)},
            str(getattr(self, "active_work_item_route_kind", None) or "frontier_endpoint"),
            x,
            y,
        )
        self.active_work_item_dispatch_announced = True
        return True

    def publish_work_item_dispatch(
        self, item_id, attempt_id, region, route_kind, x, y,
    ):
        """Emit the one stable start event used by experiment reduction."""
        self.publish_status(
            "work_item_dispatched",
            work_item_id=int(item_id),
            attempt_id=(None if attempt_id is None else int(attempt_id)),
            place_id=(None if region is None else int(region["id"])),
            route_kind=str(route_kind),
            goal=[round(float(x), 3), round(float(y), 3)],
            work_item_state=WORK_UNRESOLVED,
            attempt_state=ATTEMPT_ACTIVE,
        )

    def _active_support_quantum(self):
        """Return the physical lattice quantum supports are frozen on."""
        message = getattr(self, "map_msg", None)
        resolution = 0.1
        if message is not None:
            try:
                resolution = float(message.info.resolution)
            except (AttributeError, TypeError, ValueError):
                resolution = 0.1
        return max(resolution, 0.25)

    def active_work_item_unknown_support_checker(self):
        """Return is_unknown(physical_xy) for the active item, or None.

        The support lattice stores integer keys of the physical (odom) frame
        on the same 0.25 m quantum ``observation_supports`` freezes them
        with; the current map decides, cell by cell, whether one of those
        cells is still unobserved.  None means the evidence (map, pose,
        projector, or scan range) is missing and no coverage claim can be
        checked.
        """
        item_id = getattr(self, "active_work_item_id", None)
        ledger = getattr(self, "place_work_items", None)
        support_for = getattr(ledger, "support_for", None)
        if item_id is None or not callable(support_for):
            return None
        support = support_for(int(item_id))
        if not support:
            return None
        message = getattr(self, "map_msg", None)
        pose = getattr(self, "pose_odom", None)
        factory = getattr(self, "planar_xy_projector", None)
        if message is None or pose is None or not callable(factory):
            return None
        info = getattr(message, "info", None)
        header = getattr(message, "header", None)
        if info is None or header is None:
            return None
        try:
            resolution = float(info.resolution)
            width, height = int(info.width), int(info.height)
            origin_x = float(info.origin.position.x)
            origin_y = float(info.origin.position.y)
        except (AttributeError, TypeError, ValueError):
            return None
        if resolution <= 0 or width <= 0 or height <= 0:
            return None
        map_frame = str(getattr(header, "frame_id", "") or "map")
        physical_to_map = factory(map_frame, "odom")
        if physical_to_map is None:
            return None
        data = message.data
        quantum = self._active_support_quantum()

        def is_unknown(cell):
            try:
                x = float(cell[0]) * quantum
                y = float(cell[1]) * quantum
                mx, my = physical_to_map(x, y)
                col = int((float(mx) - origin_x) / resolution)
                row = int((float(my) - origin_y) / resolution)
            except (IndexError, TypeError, ValueError):
                return False
            if not (0 <= row < height and 0 <= col < width):
                return False
            return int(data[row * width + col]) < 0

        return is_unknown

    def active_work_item_support_is_observed(self):
        """Return whether the active item's patch is unseen from this pose.

        Reaching an endpoint is an arrival fact, not coverage.  A dispatch
        goal can sit far from the item's unknown patch (a nearer arc of the
        same connected unknown region inherits the item's identity), and
        recording ``endpoint_coverage_recorded`` there marks never-seen space
        as observed; the durable descendant rule then suppresses it forever
        and the completion gate declares exploration finished.

        Coverage is what the map proves *from the pose claiming it*, so a cell
        blocks settlement only when it is still unknown **and** in the
        sensor's line of sight from that pose.  The support patch is the whole
        unknown flood out to the sensor horizon, and part of it is hidden
        behind walls that no standoff can see past; charging those cells to
        this observation would make every ordinary boundary defer forever.
        Missing evidence fails open to the historical arrival behaviour.
        """
        item_id = getattr(self, "active_work_item_id", None)
        ledger = getattr(self, "place_work_items", None)
        support_for = getattr(ledger, "support_for", None)
        if item_id is None or not callable(support_for):
            return True
        support = support_for(int(item_id))
        if not support:
            return True
        message = getattr(self, "map_msg", None)
        pose = getattr(self, "pose_odom", None)
        factory = getattr(self, "planar_xy_projector", None)
        horizon = getattr(self, "scan_range_max", None)
        if (
            message is None
            or pose is None
            or not callable(factory)
            or not horizon
            or float(horizon) <= 0.0
        ):
            return True
        info = getattr(message, "info", None)
        header = getattr(message, "header", None)
        if info is None or header is None:
            return True
        try:
            resolution = float(info.resolution)
            width, height = int(info.width), int(info.height)
            origin_x = float(info.origin.position.x)
            origin_y = float(info.origin.position.y)
        except (AttributeError, TypeError, ValueError):
            return True
        if resolution <= 0 or width <= 0 or height <= 0:
            return True
        map_frame = str(getattr(header, "frame_id", "") or "map")
        physical_to_map = factory(map_frame, "odom")
        if physical_to_map is None:
            return True
        occupancy = np.asarray(message.data, dtype=int).reshape(height, width)
        known_free = occupancy == 0
        unknown = occupancy < 0
        try:
            observer_x, observer_y = physical_to_map(
                float(pose.x), float(pose.y),
            )
            observer = (
                int((float(observer_y) - origin_y) / resolution),
                int((float(observer_x) - origin_x) / resolution),
            )
        except (AttributeError, TypeError, ValueError):
            return True
        quantum = self._active_support_quantum()
        pose_x, pose_y = float(pose.x), float(pose.y)
        limit_fn = getattr(self, "_active_work_item_distance_limit", None)
        limit = limit_fn() if callable(limit_fn) else None
        anchor_fn = getattr(self, "_active_work_item_anchor", None)
        anchor = anchor_fn(item_id) if callable(anchor_fn) else None
        if limit is not None and anchor is not None:
            if math.hypot(pose_x - anchor[0], pose_y - anchor[1]) <= limit * 1.25:
                return True
        horizon_m = float(horizon)
        for cell in support:
            x = float(cell[0]) * quantum
            y = float(cell[1]) * quantum
            if math.hypot(x - pose_x, y - pose_y) > horizon_m:
                # Beyond the sensor horizon: no pose here could observe it.
                continue
            try:
                mx, my = physical_to_map(x, y)
                row = int((float(my) - origin_y) / resolution)
                col = int((float(mx) - origin_x) / resolution)
            except (IndexError, TypeError, ValueError):
                continue
            if not (0 <= row < height and 0 <= col < width):
                continue
            if not unknown[row, col]:
                continue
            if grid_line_reaches_cell(
                known_free, observer[0], observer[1], row, col,
            ):
                # Unknown, in range, and in clear line of sight: this
                # viewpoint should have resolved it and did not.
                return False
        return True

    def _active_work_item_physical_viewpoint(self, item_id):
        """Return the route Attempt's physical viewpoint, if available.

        The Attempt stores the endpoint after the map-to-odom projection at
        dispatch time.  That is preferable to reprojecting the old map goal
        after a later SLAM correction.  The map-goal fallback keeps narrow
        legacy callers source-compatible.
        """
        ledger = getattr(self, "place_work_items", None)
        getter = getattr(ledger, "active_attempt_viewpoint", None)
        if callable(getter):
            viewpoint = getter(item_id)
            if viewpoint is not None:
                return viewpoint
        goal = getattr(self, "active_work_item_goal", None)
        if goal is None:
            return None
        try:
            x, y = float(goal[0]), float(goal[1])
        except (IndexError, TypeError, ValueError):
            return None
        return self.work_item_physical_xy(getattr(self, "map_msg", None), x, y)

    def _active_work_item_anchor(self, item_id):
        """Return the immutable WorkItem anchor with a legacy fallback."""
        ledger = getattr(self, "place_work_items", None)
        getter = getattr(ledger, "anchor_for", None)
        if callable(getter):
            anchor = getter(item_id)
            if anchor is not None:
                return anchor
        # ``dispatch()`` predates durable support anchors and stores its
        # endpoint in x/y.  Treat that endpoint as the reference only for
        # those compatibility records; reconciled WorkItems leave x/y empty.
        snapshot = getattr(ledger, "snapshot", None)
        if not callable(snapshot):
            return None
        for item in snapshot():
            try:
                if int(item.get("id")) != int(item_id):
                    continue
                if item.get("x") is None or item.get("y") is None:
                    return None
                return float(item["x"]), float(item["y"])
            except (TypeError, ValueError):
                return None
        return None

    def _active_work_item_normal(self, item_id):
        """Return the immutable WorkItem normal with a snapshot fallback."""
        ledger = getattr(self, "place_work_items", None)
        getter = getattr(ledger, "normal_for", None)
        if callable(getter):
            normal = getter(item_id)
            if normal is not None:
                return normal
        snapshot = getattr(ledger, "snapshot", None)
        if not callable(snapshot):
            return None
        for item in snapshot():
            try:
                if int(item.get("id")) != int(item_id):
                    continue
                normal = item.get("normal_xy")
                if normal is None:
                    return None
                return float(normal[0]), float(normal[1])
            except (TypeError, ValueError):
                return None
        return None

    def _active_work_item_distance_limit(self):
        """Use the existing viewpoint-distance contract for completion too."""
        message = getattr(self, "map_msg", None)
        try:
            resolution = float(message.info.resolution)
        except (AttributeError, TypeError, ValueError):
            resolution = None
        if resolution is None or not math.isfinite(resolution) or resolution <= 0.0:
            # A compatibility fixture may have no map message.  The production
            # route always has one; returning None makes a non-zero distance
            # fail closed instead of silently certifying it.
            return None
        limit_fn = getattr(self, "_work_item_viewpoint_distance_limit", None)
        if callable(limit_fn):
            try:
                limit = float(limit_fn(resolution))
            except (TypeError, ValueError):
                limit = None
            if limit is not None and math.isfinite(limit) and limit >= 0.0:
                return limit
        # Keep compatibility for a slim test/legacy composition that does not
        # include GlobalFrontierCandidateWorkItemMixin.  This is the same
        # derived formula used by the production mixin, not a new parameter.
        try:
            work_item_approach = float(getattr(self, "work_item_approach_distance", 0.55))
            depth = max(
                0.35,
                work_item_approach,
                resolution,
            )
        except (TypeError, ValueError):
            return None
        return math.sqrt(2.0) * depth + 2.0 * resolution

    def _validate_active_work_item_viewpoint(self, item_id):
        """Validate that the executed viewpoint is local to its WorkItem."""
        anchor = self._active_work_item_anchor(item_id)
        viewpoint = self._active_work_item_physical_viewpoint(item_id)
        if anchor is None or viewpoint is None:
            return False, None, None, anchor, viewpoint, "work_item_geometry_unavailable"
        distance = math.hypot(
            float(viewpoint[0]) - float(anchor[0]),
            float(viewpoint[1]) - float(anchor[1]),
        )
        limit_fn = getattr(self, "_active_work_item_distance_limit", None)
        limit = limit_fn() if callable(limit_fn) else None
        if limit is None:
            return False, distance, None, anchor, viewpoint, "work_item_distance_unavailable"
        if distance > limit:
            return False, distance, limit, anchor, viewpoint, "goal_too_far_from_work_item"
        return True, distance, limit, anchor, viewpoint, None

    def settle_active_work_item(self, now, state, reason):
        """Finish an Attempt and resolve WorkItem only from observation proof."""
        item_id = getattr(self, "active_work_item_id", None)
        if item_id is None:
            return None
        # A map observation may resolve the endpoint before the next planner
        # tick publishes a command. Preserve a zero-duration, explicitly
        # paired lifecycle record rather than an unmatched terminal event.
        self.announce_active_work_item_dispatch()
        attempt_id = getattr(self, "active_work_item_attempt_id", None)
        item = None
        is_dead_corner = any(
            k in str(reason).lower()
            for k in (
                "stall",
                "timeout",
                "recovery",
                "unreachable",
                "abort",
                "disconnected",
                "no_progress",
            )
        )
        if is_dead_corner:
            state = WORK_RESOLVED
            reason = str(reason) if str(reason).startswith("dead_corner_") else ("dead_corner_abandoned_" + str(reason))

        if state == WORK_RESOLVED:
            if str(reason) == "endpoint_coverage_recorded":
                (
                    viewpoint_ok,
                    viewpoint_distance,
                    viewpoint_limit,
                    viewpoint_anchor,
                    viewpoint,
                    viewpoint_reason,
                ) = self._validate_active_work_item_viewpoint(item_id)
                if not viewpoint_ok:
                    # A terminal route callback can arrive after a stale map
                    # snapshot selected an endpoint.  It is a failed Attempt,
                    # never semantic completion: keep the WorkItem unresolved
                    # so the next planner pass can compile a nearer viewpoint.
                    state = WORK_DEFERRED
                    reason = viewpoint_reason or "goal_too_far_from_work_item"
                    item, attempt = self.place_work_items.fail_attempt(
                        item_id,
                        now,
                        reason,
                        state=ATTEMPT_FAILED,
                        attempt_id=attempt_id,
                    )
                    if item is not None and attempt is not None:
                        self.publish_status(
                            "viewpoint_attempt_settled",
                            work_item_id=int(item_id),
                            attempt_id=int(attempt["id"]),
                            place_id=int(item["place_id"]),
                            work_item_state=WORK_UNRESOLVED,
                            attempt_state=str(attempt["state"]),
                            reason=str(reason),
                            viewpoint_xy=(
                                None
                                if viewpoint is None
                                else [round(float(value), 3) for value in viewpoint]
                            ),
                            anchor_xy=(
                                None
                                if viewpoint_anchor is None
                                else [
                                    round(float(value), 3)
                                    for value in viewpoint_anchor
                                ]
                            ),
                            viewpoint_distance=(
                                None
                                if viewpoint_distance is None
                                else round(float(viewpoint_distance), 3)
                            ),
                            viewpoint_distance_limit=(
                                None
                                if viewpoint_limit is None
                                else round(float(viewpoint_limit), 3)
                            ),
                        )
            if state == WORK_RESOLVED and str(reason) == "endpoint_coverage_recorded":
                # Coverage is an observation fact, not an arrival fact.  Keep
                # only the cells the current map proves were seen, so the
                # descendant rule cannot later suppress the unseen remainder.
                checker = self.active_work_item_unknown_support_checker()
                if checker is not None:
                    self.place_work_items.restrict_support_to_observed(
                        item_id, checker,
                    )
                item = self.place_work_items.resolve(
                    item_id, now, reason, attempt_id=attempt_id,
                )
                if item is not None:
                    self.publish_status(
                        "work_item_settled",
                        work_item_id=int(item_id),
                        attempt_id=(None if attempt_id is None else int(attempt_id)),
                        place_id=int(item["place_id"]),
                        work_item_state=WORK_RESOLVED,
                        attempt_state=ATTEMPT_SUCCEEDED,
                        reason=str(reason),
                    )
            elif state == WORK_RESOLVED:
                item = self.place_work_items.resolve(
                    item_id, now, reason, attempt_id=attempt_id,
                )
                if item is not None:
                    self.publish_status(
                        "work_item_settled",
                        work_item_id=int(item_id),
                        attempt_id=(None if attempt_id is None else int(attempt_id)),
                        place_id=int(item["place_id"]),
                        work_item_state=WORK_RESOLVED,
                        attempt_state=(
                            ATTEMPT_FAILED
                            if str(reason).startswith("dead_corner_")
                            else ATTEMPT_SUCCEEDED
                        ),
                        reason=str(reason),
                    )
        elif state == WORK_DEFERRED:
            item, attempt = self.place_work_items.fail_attempt(
                item_id,
                now,
                reason,
                state=ATTEMPT_FAILED,
                attempt_id=attempt_id,
            )
            if item is not None and attempt is not None:
                self.publish_status(
                    "viewpoint_attempt_settled",
                    work_item_id=int(item_id),
                    attempt_id=int(attempt["id"]),
                    place_id=int(item["place_id"]),
                    work_item_state=WORK_UNRESOLVED,
                    attempt_state=str(attempt["state"]),
                    reason=str(reason),
                )
        else:
            raise ValueError("resolved WorkItem or deferred Attempt required")
        # Prevent a later watchdog/terminal callback for the same route from
        # producing a second terminal record for one observation lease.
        self.active_work_item_id = None
        self.active_work_item_attempt_id = None
        self.active_work_item_place_id = None
        self.active_work_item_goal = None
        self.active_work_item_route_kind = None
        self.active_work_item_dispatch_announced = False
        return item
