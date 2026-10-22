"""Rehydrate Place-owned observation WorkItems after a SLAM map change.

The frontier grid is a short-lived projection.  A durable WorkItem keeps its
physical boundary anchor and unknown-side normal, which is enough to compile a
new safe observation viewpoint when the original frontier arc disappears.
"""

import math

import numpy as np

try:
    import rospy
except ImportError:
    rospy = None

from global_frontier_models import FrontierCandidateRoute


class GlobalFrontierCandidateWorkItemMixin:
    """Build current-map candidates for unresolved local WorkItems."""

    def _work_item_viewpoint_distance_limit(self, resolution):
        """Return the maximum local standoff-to-anchor distance.

        The normal and two lateral views are generated at ``depth`` from the
        physical boundary anchor.  Reassociation can move a desired point by
        a couple of cells, so keep that existing grid scale in the bound
        instead of introducing a second user-facing distance parameter.
        """
        resolution = max(1e-6, float(resolution))
        work_item_approach = float(getattr(self, "work_item_approach_distance", 0.55))
        depth = max(
            0.35,
            work_item_approach,
            resolution,
        )
        reassociation_radius = float(getattr(self, "active_reassociation_radius", 1.0))
        return max(
            math.sqrt(2.0) * depth + 2.0 * resolution,
            depth + reassociation_radius,
        )

    def _work_item_viewpoint_distance(self, request, anchor, x, y):
        """Measure a candidate endpoint against a durable physical anchor."""
        anchor = self._finite_xy(anchor)
        if anchor is None:
            return None
        candidate = (float(x), float(y))
        projector = getattr(request, "map_to_physical_xy", None)
        if projector is not None:
            try:
                projected = self._finite_xy(projector(candidate[0], candidate[1]))
            except Exception:
                projected = None
            if projected is None:
                return None
            candidate = projected
        return math.hypot(candidate[0] - anchor[0], candidate[1] - anchor[1])

    def _work_item_viewpoint_is_local(self, request, item, x, y):
        """Return whether a WorkItem endpoint stays near its anchor."""
        distance = self._work_item_viewpoint_distance(
            request, item.get("anchor_xy"), x, y,
        )
        return (
            distance is not None
            and distance <= self._work_item_viewpoint_distance_limit(
                request.message.info.resolution
            )
        )

    @staticmethod
    def _finite_xy(value):
        if value is None:
            return None
        try:
            point = float(value[0]), float(value[1])
        except (IndexError, TypeError, ValueError):
            return None
        return point if all(math.isfinite(item) for item in point) else None

    @staticmethod
    def _finite_normal(value):
        point = GlobalFrontierCandidateWorkItemMixin._finite_xy(value)
        if point is None:
            return None
        length = math.hypot(point[0], point[1])
        if not math.isfinite(length) or length <= 1e-9:
            return None
        return point[0] / length, point[1] / length

    @staticmethod
    def _viewpoint_ladder(anchor, normal, depth):
        """Return one normal and two lateral views of the same boundary."""
        tangent = -normal[1], normal[0]
        source = (
            anchor[0] - depth * normal[0],
            anchor[1] - depth * normal[1],
        )
        return (
            ("normal", source),
            (
                "lateral_left",
                (source[0] + depth * tangent[0], source[1] + depth * tangent[1]),
            ),
            (
                "lateral_right",
                (source[0] - depth * tangent[0], source[1] - depth * tangent[1]),
            ),
        )

    def _work_item_map_anchor(self, request, item):
        """Project a durable physical anchor into the current map frame."""
        physical_anchor = self._finite_xy(item.get("anchor_xy"))
        map_anchor = self._finite_xy((item.get("x"), item.get("y")))
        anchor = physical_anchor or map_anchor
        if anchor is None:
            return None
        header = getattr(request.message, "header", None)
        map_frame = getattr(header, "frame_id", "map") or "map"
        projector_factory = getattr(self, "planar_xy_projector", None)
        if projector_factory is not None:
            projector = projector_factory(map_frame, "odom")
            if projector is not None:
                projected = self._finite_xy(projector(anchor[0], anchor[1]))
                if projected is not None:
                    return projected
            # A durable ``anchor_xy`` is physical-frame data. Do not silently
            # reinterpret it as map coordinates when the inverse TF is missing.
            return map_anchor
        # Geometry-only fixtures may intentionally use one frame for both map
        # and physical coordinates. Production callers keep the physical frame
        # contract by providing the inverse TF projector above.
        return anchor if request.map_to_physical_xy is None else None

    def _find_safe_viewpoint_cell(
        self,
        request,
        steps,
        desired,
        anchor,
        distance_limit,
        item_id,
        map_epoch,
        ledger,
        depth=0.55,
    ):
        """Find a reachable viewpoint cell in safe, un-inflated free space."""
        message = request.message
        resolution = max(1e-6, float(message.info.resolution))
        reassociate = getattr(self, "nearest_reachable_cell", None)

        preferred_mask = getattr(request, "preferred_mask", None)
        validation = getattr(request, "validation", None)
        occupied = getattr(request, "occupied", None)

        # In pure unit test fixtures where neither preferred_mask nor occupied
        # is supplied, preserve the legacy single-cell contract.
        if preferred_mask is None and occupied is None:
            if callable(reassociate):
                return reassociate(message, steps, desired[0], desired[1])
            return None

        search_radius = max(
            float(getattr(self, "active_reassociation_radius", 1.0)),
            1.2,
        )
        target_col = int((desired[0] - message.info.origin.position.x) / resolution)
        target_row = int((desired[1] - message.info.origin.position.y) / resolution)
        cell_radius = max(1, int(math.ceil(search_radius / resolution)))

        r0 = max(0, target_row - cell_radius)
        r1 = min(steps.shape[0], target_row + cell_radius + 1)
        c0 = max(0, target_col - cell_radius)
        c1 = min(steps.shape[1], target_col + cell_radius + 1)

        reachable = np.argwhere(steps[r0:r1, c0:c1] >= 0)
        if reachable.size == 0:
            if callable(reassociate):
                return reassociate(message, steps, desired[0], desired[1])
            return None

        reachable[:, 0] += r0
        reachable[:, 1] += c0

        # Anchor in grid coordinates for line-of-sight checks
        r_anchor = int(round((anchor[1] - message.info.origin.position.y) / resolution))
        c_anchor = int(round((anchor[0] - message.info.origin.position.x) / resolution))

        def has_line_of_sight(r, c):
            if occupied is None:
                return True
            dr = abs(r_anchor - r)
            dc = abs(c_anchor - c)
            steps_count = max(dr, dc)
            if steps_count <= 1:
                return True
            rows, cols = occupied.shape
            for i in range(1, steps_count):
                rr = int(round(r + (r_anchor - r) * i / float(steps_count)))
                cc = int(round(c + (c_anchor - c) * i / float(steps_count)))
                if 0 <= rr < rows and 0 <= cc < cols:
                    if occupied[rr, cc]:
                        return False
            return True

        tier1_candidates = []
        tier2_candidates = []
        min_standoff = max(0.20, min(0.35, depth * 0.5))

        costmap_checker = getattr(self, "candidate_costmap_distance", None)

        for r, c in reachable:
            cx, cy = self.cell_xy(message, r, c)
            d_anchor = math.hypot(cx - anchor[0], cy - anchor[1])
            if distance_limit is not None and d_anchor > distance_limit:
                continue
            if d_anchor < min_standoff:
                continue

            physical = (
                (cx, cy)
                if request.map_to_physical_xy is None
                else request.map_to_physical_xy(cx, cy)
            )
            if ledger is not None and not ledger.viewpoint_is_available(
                item_id, physical, map_epoch=map_epoch
            ):
                continue

            if not has_line_of_sight(r, c):
                continue

            # Check costmap validation if available
            if validation is not None and callable(costmap_checker):
                costmap_dist = costmap_checker(validation, cx, cy)
                if costmap_dist is None:
                    continue

            d_desired_sq = (cx - desired[0]) ** 2 + (cy - desired[1]) ** 2

            # Check preferred mask
            in_preferred = (
                preferred_mask is not None
                and 0 <= r < preferred_mask.shape[0]
                and 0 <= c < preferred_mask.shape[1]
                and bool(preferred_mask[r, c])
            )

            if in_preferred:
                tier1_candidates.append((d_desired_sq, int(r), int(c)))
            else:
                tier2_candidates.append((d_desired_sq, int(r), int(c)))

        if tier1_candidates:
            tier1_candidates.sort(key=lambda item: item[0])
            best = tier1_candidates[0]
            return best[1], best[2]

        if tier2_candidates:
            tier2_candidates.sort(key=lambda item: item[0])
            best = tier2_candidates[0]
            return best[1], best[2]

        return None

    def pending_local_work_item_candidates(
        self, request, context, frontier, approach_cells,
    ):
        """Return one safe current-map viewpoint for each unresolved item.

        Doorway-owned WorkItems are excluded because their PortalProbe ledger
        is the sole owner of source/destination evidence.  A candidate is still
        validated by the current BFS and costmap; durable geometry only supplies
        the identity and the desired side of the observation.
        """
        ledger = getattr(self, "place_work_items", None)
        place_id = getattr(context, "source_place_id", None) or 1
        del frontier, approach_cells
        if ledger is None:
            return []
        snapshot = getattr(ledger, "snapshot", None)
        records = () if snapshot is None else snapshot()
        work_item_filter = getattr(self, "graph_route_work_item_filter", None)
        selected_work_item_id = getattr(self, "graph_route_work_item_id", None)
        if work_item_filter == "none":
            return []
        if work_item_filter == "selected":
            try:
                selected_work_item_id = int(selected_work_item_id)
            except (TypeError, ValueError):
                return []
            if selected_work_item_id <= 0:
                return []
        probe_ledger = getattr(self, "portal_probe_ledger", None)
        probe_for_work = (
            None if probe_ledger is None
            else getattr(probe_ledger, "probe_for_work_item", None)
        )
        message = request.message
        steps = request.steps
        if steps is None or not hasattr(steps, "shape"):
            return []
        reassociate = getattr(self, "nearest_reachable_cell", None)
        if not callable(reassociate):
            return []
        resolution = max(1e-6, float(message.info.resolution))
        map_epoch = getattr(
            getattr(request, "components", None), "epoch", None
        )
        work_item_approach = float(getattr(self, "work_item_approach_distance", 0.55))
        depth = max(
            0.35,
            work_item_approach,
            resolution,
        )
        distance_limit = self._work_item_viewpoint_distance_limit(resolution)
        report = {
            "place_id": int(place_id),
            "map_epoch": getattr(
                getattr(request, "components", None), "epoch", None,
            ),
            "candidate_count": 0,
            "rejection_reasons": {},
            "items_seen": 0,
            "item_audit": [],
        }
        candidates = []
        target_item_ids = {
            int(r["id"])
            for r in records
            if "id" in r and (r.get("is_target") or r.get("kind") == "target_observation")
        }

        def reject(reason):
            report["rejection_reasons"][reason] = (
                report["rejection_reasons"].get(reason, 0) + 1
            )

        for item in records:
            if getattr(self, "planning_should_preempt", lambda: False)():
                return []
            try:
                item_id = int(item.get("id"))
                item_place = int(item.get("place_id"))
            except (TypeError, ValueError):
                continue
            if work_item_filter == "selected" and item_id != selected_work_item_id:
                continue
            if str(item.get("state", "")).strip().lower() != "unresolved":
                continue
            report["items_seen"] += 1
            audit = {"work_item_id": item_id}
            if item.get("active_attempt_id") is not None:
                audit["reason"] = "attempt_active"
                report["item_audit"].append(audit)
                reject("attempt_active")
                continue
            if callable(probe_for_work) and probe_for_work(item_id) is not None:
                audit["reason"] = "owned_by_portal_probe"
                report["item_audit"].append(audit)
                reject("owned_by_portal_probe")
                continue
            anchor = self._work_item_map_anchor(request, item)
            normal = self._finite_normal(item.get("normal_xy"))
            normal_source = "durable_unknown_side"
            if anchor is None:
                audit["reason"] = "physical_anchor_projection_unavailable"
                report["item_audit"].append(audit)
                reject("physical_anchor_projection_unavailable")
                continue

            if normal is None:

                # A one-cell or symmetric frontier arc can have a valid
                # physical anchor but no unique unknown-side normal.  That is
                # insufficient for a durable *identity* update, but it is
                # still enough for this snapshot to choose the known-side
                # approach from the robot toward the anchor.  Keep this
                # fallback map-local; never persist it as the item's normal.
                robot_anchor = self._finite_xy(
                    getattr(request, "robot_map", None)
                )
                if robot_anchor is None:
                    robot_anchor = self._finite_xy(
                        getattr(request, "route_anchor_xy", None)
                    )
                if robot_anchor is not None:
                    direction = (
                        anchor[0] - robot_anchor[0],
                        anchor[1] - robot_anchor[1],
                    )
                    normal = self._finite_normal(direction)
                    if normal is not None:
                        normal_source = "current_robot_approach"
            if normal is None:
                audit["anchor_map"] = [round(float(value), 3) for value in anchor]
                audit["reason"] = "unknown_side_normal_missing"
                report["item_audit"].append(audit)
                reject("unknown_side_normal_missing")
                continue
            audit["anchor_map"] = [round(float(value), 3) for value in anchor]
            audit["normal"] = [round(float(value), 3) for value in normal]
            audit["normal_source"] = normal_source
            selected = None
            for strategy, desired in self._viewpoint_ladder(anchor, normal, depth):
                view_audit = dict(audit)
                view_audit["strategy"] = strategy
                view_audit["desired_viewpoint"] = [
                    round(float(desired[0]), 3), round(float(desired[1]), 3),
                ]
                cell = self._find_safe_viewpoint_cell(
                    request=request,
                    steps=steps,
                    desired=desired,
                    anchor=anchor,
                    distance_limit=distance_limit,
                    item_id=item_id,
                    map_epoch=map_epoch,
                    ledger=ledger,
                    depth=depth,
                )
                if cell is None or steps[cell] < 0:
                    view_audit["reason"] = "no_reachable_observation_viewpoint"
                    report["item_audit"].append(view_audit)
                    reject("no_reachable_observation_viewpoint")
                    continue
                row, col = cell
                x, y = self.cell_xy(message, row, col)
                viewpoint_distance = self._work_item_viewpoint_distance(
                    request, item.get("anchor_xy"), x, y,
                )
                if (
                    viewpoint_distance is None
                    or viewpoint_distance > distance_limit
                ):
                    view_audit["cell"] = [int(row), int(col)]
                    view_audit["viewpoint_distance"] = (
                        None
                        if viewpoint_distance is None
                        else round(float(viewpoint_distance), 3)
                    )
                    view_audit["viewpoint_distance_limit"] = round(
                        float(distance_limit), 3,
                    )
                    view_audit["reason"] = "viewpoint_too_far_from_work_item"
                    report["item_audit"].append(view_audit)
                    reject("viewpoint_too_far_from_work_item")
                    continue
                costmap_distance = self.candidate_costmap_distance(
                    request.validation, x, y,
                )
                if request.validation is not None and costmap_distance is None:
                    view_audit["cell"] = [int(row), int(col)]
                    view_audit["reason"] = "costmap_route_unreachable"
                    report["item_audit"].append(view_audit)
                    reject("costmap_route_unreachable")
                    continue
                physical = (
                    (x, y)
                    if request.map_to_physical_xy is None
                    else request.map_to_physical_xy(x, y)
                )
                if not ledger.viewpoint_is_available(
                    item_id, physical, map_epoch=map_epoch,
                ):
                    view_audit["cell"] = [int(row), int(col)]
                    route_rejection = getattr(
                        ledger, "route_rejection_for_viewpoint", None
                    )
                    view_audit["reason"] = (
                        "route_rejected_in_map_epoch"
                        if callable(route_rejection)
                        and route_rejection(
                            item_id, physical, map_epoch=map_epoch,
                        ) is not None
                        else "viewpoint_already_attempted"
                    )
                    report["item_audit"].append(view_audit)
                    reject(view_audit["reason"])
                    continue
                path_distance = (
                    float(costmap_distance)
                    if costmap_distance is not None
                    else float(steps[row, col]) * resolution
                )
                retry = bool(
                    ledger.has_failed_viewpoint(item_id, map_epoch=map_epoch)
                )
                selected = (
                    row, col, x, y, costmap_distance, path_distance, physical,
                    strategy, retry,
                )
                break
            if selected is None:
                continue
            row, col, x, y, costmap_distance, path_distance, physical, strategy, retry = selected
            cand_is_target = int(item_id) in target_item_ids
            candidate = FrontierCandidateRoute(
                row=int(row),
                col=int(col),
                action_tier="target" if cand_is_target else "local",
                place_hops=0,
                x=float(x),
                y=float(y),
                path_distance=path_distance,
                score_path_distance=0.0 if cand_is_target else path_distance,
                work_item_id=item_id,
                work_item_match="rehydrated_work_item",
                work_item_support_cells=int(item.get("support_cell_count", 0)),
                work_item_normal_xy=normal,
                viewpoint_retry=retry,
                graph_action="reinspect_target" if cand_is_target else ("retry_viewpoint" if retry else "observe_local_work"),
                graph_action_reason="target_observation_approach" if cand_is_target else ("durable_work_item_rehydrated_%s" % strategy),
            )
            candidates.append((int(row), int(col), candidate))
            report["candidate_count"] += 1
            audit["strategy"] = strategy
            audit["cell"] = [int(row), int(col)]
            audit["viewpoint_map"] = [round(float(x), 3), round(float(y), 3)]
            audit["reason"] = "executable"
            report["item_audit"].append(audit)

        # Spatial NMS on rehydrated candidates: prevent clustering multiple viewpoints
        # within density_separation distance.
        density_sep = float(getattr(self, "work_item_density_radius", 0.8))
        if len(candidates) > 1 and density_sep > 0.0:
            candidates.sort(
                key=lambda entry: (
                    0 if int(getattr(entry[2], "work_item_id", 0) or 0) in target_item_ids else 1,
                    -int(getattr(entry[2], "work_item_support_cells", 0)),
                    float(getattr(entry[2], "path_distance", 0.0)),
                )
            )
            filtered_candidates = []
            for r, c, cand in candidates:
                cand_is_target = int(getattr(cand, "work_item_id", 0) or 0) in target_item_ids
                if cand_is_target:
                    filtered_candidates.append((r, c, cand))
                    continue
                cand_xy = (cand.x, cand.y)
                suppressed = False
                for _fr, _fc, accepted in filtered_candidates:
                    dist = math.hypot(cand_xy[0] - accepted.x, cand_xy[1] - accepted.y)
                    if dist <= density_sep:
                        suppressed = True
                        break
                if not suppressed:
                    filtered_candidates.append((r, c, cand))
                else:
                    reject("suppressed_by_denser_candidate")
            candidates = filtered_candidates
            report["candidate_count"] = len(candidates)

        report["rejection_reasons"] = dict(sorted(report["rejection_reasons"].items()))
        if report != getattr(self, "last_local_work_item_rehydration_report", None):
            self.last_local_work_item_rehydration_report = report
            publish = getattr(self, "publish_status", None)
            if callable(publish):
                publish("local_work_item_rehydration", **report)
        return candidates


__all__ = ["GlobalFrontierCandidateWorkItemMixin"]
