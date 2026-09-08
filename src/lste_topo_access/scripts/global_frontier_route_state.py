#!/usr/bin/env python3

"""Map cache and active-route state helpers for global-frontier exploration."""

import copy
import math
import time

import rospy

from global_frontier_topology import grid_frontier_observed_from_viewpoint


class GlobalFrontierRouteStateMixin:
    def on_map(self, message):
        self.map_msg = message
        scheduler = getattr(self, "decision_wake_scheduler", None)
        if scheduler is not None:
            scheduler.observe_map(message)

    def on_costmap(self, message):
        self.costmap_msg = message
        self.costmap_message_count += 1
        self.costmap_last_receive_wall = time.monotonic()
        scheduler = getattr(self, "decision_wake_scheduler", None)
        if scheduler is not None:
            # Full costmap publications are the dependency boundary. The
            # incremental update stream may run faster than the graph layer;
            # the cached full message is still invalidated for execution,
            # while the next full publication wakes deliberation.
            scheduler.observe_costmap(message)
        rospy.loginfo_once(
            "Global frontier received costmap frame=%s size=%dx%d resolution=%.3f "
            "stamp=%.3f cells=%d",
            message.header.frame_id,
            message.info.width,
            message.info.height,
            message.info.resolution,
            message.header.stamp.to_sec(),
            len(message.data),
        )

    def on_costmap_update(self, update):
        """Apply costmap_2d's incremental update to the cached full grid."""
        base = self.costmap_msg
        if base is None:
            return
        width = int(base.info.width)
        height = int(base.info.height)
        x, y = int(update.x), int(update.y)
        update_width, update_height = int(update.width), int(update.height)
        # When gmapping expands the map, costmap_2d can emit a full-grid
        # OccupancyGridUpdate whose dimensions no longer fit the old cached
        # OccupancyGrid. This is a replacement, not a malformed increment.
        # Dropping it leaves the frontier validator permanently stale until a
        # future full /costmap publication happens to arrive.
        if (
            x == 0
            and y == 0
            and update_width > 0
            and update_height > 0
            and len(update.data) == update_width * update_height
            and (update_width != width or update_height != height)
        ):
            updated = copy.deepcopy(base)
            updated.header = copy.deepcopy(update.header)
            updated.info.width = update_width
            updated.info.height = update_height
            updated.data = list(update.data)
            self.costmap_msg = updated
            self.costmap_message_count += 1
            self.costmap_last_receive_wall = time.monotonic()
            self.cached_costmap_validation = None
            self.cached_costmap_validation_wall = 0.0
            rospy.loginfo(
                "Global frontier accepted resized full costmap update "
                "old=%dx%d new=%dx%d",
                width, height, update_width, update_height,
            )
            return
        if (
            width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + update_width > width
            or y + update_height > height
            or len(update.data) != update_width * update_height
        ):
            rospy.logwarn_throttle(
                5.0,
                "Global frontier ignored malformed costmap update "
                "origin=(%d,%d) size=%dx%d base=%dx%d cells=%d",
                x,
                y,
                update_width,
                update_height,
                width,
                height,
                len(update.data),
            )
            return
        # Do not mutate a message currently being read by the timer thread.
        updated = copy.deepcopy(base)
        data = list(updated.data)
        for row in range(update_height):
            start = (y + row) * width + x
            end = start + update_width
            source_start = row * update_width
            data[start:end] = update.data[source_start:source_start + update_width]
        updated.data = data
        self.costmap_msg = updated
        self.costmap_last_receive_wall = time.monotonic()
        self.cached_costmap_validation = None
        self.cached_costmap_validation_wall = 0.0

    def on_pose(self, message):
        self.pose_odom = message
        scheduler = getattr(self, "decision_wake_scheduler", None)
        if scheduler is not None:
            scheduler.observe_pose(message is not None)

    def _odom_coverage_cell(self, pose):
        """Return a stable, coarse odom cell for one active route transaction."""
        if pose is None:
            return None
        scale = self.odom_novel_cell_size
        return (
            int(math.floor(float(pose.x) / scale)),
            int(math.floor(float(pose.y) / scale)),
        )

    def _reset_active_odom_coverage(self):
        """Start route-local coverage accounting at the current base pose."""
        self.active_visited_odom_cells.clear()
        cell = self._odom_coverage_cell(self.pose_odom)
        if cell is not None:
            self.active_visited_odom_cells.add(cell)

    def _entered_novel_active_odom_cell(self):
        """Return true exactly once for each newly entered active-route cell."""
        cell = self._odom_coverage_cell(self.pose_odom)
        if cell is None or cell in self.active_visited_odom_cells:
            return False
        self.active_visited_odom_cells.add(cell)
        return True

    def _clear_active_post_turn_watchdog(self):
        """Discard turn-completion evidence when frontier ownership changes."""
        self.active_turn_completed_route_id = 0
        self.active_turn_completed_goal = None
        self.active_turn_completed_wall = 0.0
        self.active_turn_completed_odom_xy = None
        self.active_turn_completed_translation = 0.0
        self.active_turn_completed_launched = False

    def _clear_active_place_departure(self):
        """Forget an uncommitted cross-place transition without closing it."""
        self.place_departure.clear()

    def active_route_history_pose(self, fallback_map_pose):
        """Return the stable physical frame available for route history."""
        if self.pose_odom is not None:
            return "odom", (float(self.pose_odom.x), float(self.pose_odom.y))
        return "map", (float(fallback_map_pose[0]), float(fallback_map_pose[1]))

    def begin_active_route_history(self, robot_map):
        """Start a route lease without mixing odom and map-frame samples."""
        frame, pose = self.active_route_history_pose(robot_map)
        self.active_route_history_frame = frame
        self.active_route_history.begin(pose)

    def remember_active_route_position(self, robot_map):
        """Record physical motion, restarting safely if its frame changed."""
        frame, pose = self.active_route_history_pose(robot_map)
        if frame != self.active_route_history_frame:
            self.active_route_history_frame = frame
            self.active_route_history.begin(pose)
            return
        self.active_route_history.remember(
            pose,
            max(self.waypoint_release_radius, self.frontier_approach_distance),
        )

    def clear_active_route_history(self):
        """Drop all recovery anchors when the route lease ends."""
        self.active_route_history.clear()
        self.active_route_history_frame = None

    def local_egress_anchor(self):
        """Resolve a reached pose into the current map frame for Navfn."""
        if self.active_last_robot_xy is None:
            return None
        frame, current_pose = self.active_route_history_pose(
            self.active_last_robot_xy
        )
        if frame != self.active_route_history_frame:
            return None
        anchor = self.active_route_history.nearest_exit_anchor(
            current_pose,
            max(self.waypoint_release_radius, self.frontier_approach_distance),
        )
        if anchor is None or frame == "map":
            return anchor
        map_frame = (
            self.map_msg.header.frame_id if self.map_msg is not None else "map"
        ) or "map"
        return self.transform_xy(map_frame, frame, anchor[0], anchor[1])

    def prepare_local_egress(self, reason, source_region_id=None):
        """Queue one recovery goal on this route's demonstrated safe history."""
        if self.active_route_kind == "local_egress":
            return False
        resumes_portal_transition = (
            self.active_route_kind == "portal_transition"
            and self.pending_portal_retry is not None
        )
        anchor = self.local_egress_anchor()
        if anchor is None:
            return False
        if source_region_id is None:
            source_region_id = self.active_frontier_region_id
        lease = getattr(self, "local_egress_place_lease", None)
        if lease is not None:
            # A normal endpoint egress temporarily owns its source place and
            # must rebind it after recovery. A failed portal remains an
            # explicit edge transaction instead, so it retries from fresh
            # topology and must not reopen the source room.
            if resumes_portal_transition:
                lease.clear()
            else:
                lease.queue(source_region_id)
        self.pending_local_egress = {
            "anchor_map": anchor,
            "source_route_id": int(self.active_route_id),
            "source_region_id": source_region_id,
            "reason": str(reason),
            "resume_portal_transition": resumes_portal_transition,
        }
        self.publish_status(
            "local_egress_prepared",
            route_id=int(self.active_route_id),
            source_goal=(
                None if self.active_frontier is None else [
                    round(float(self.active_frontier[2]), 3),
                    round(float(self.active_frontier[3]), 3),
                ]
            ),
            recovery_goal=[round(float(anchor[0]), 3), round(float(anchor[1]), 3)],
            reason=str(reason),
            history_points=len(self.active_route_history),
            source_region_id=source_region_id,
            resume_portal_transition=resumes_portal_transition,
        )
        return True

    def clear_prefetched_frontier(self):
        """Discard the cached successor and all metadata bound to it."""
        self.prefetched_frontier = None
        self.prefetched_goal_map = None
        self.prefetched_frontier_information = None
        self.prefetched_frontier_component = None
        self.prefetched_work_item_id = None
        self.prefetched_work_item_match = None
        self.prefetched_work_item_support_cells = 0

    def release_active_frontier(
        self, discard_prefetch=False, preserve_place_departure=False,
    ):
        """Release the current endpoint lease without changing place closure.

        Successful endpoint observation is deliberately weaker than leaving a
        place. This reset only ends the controller action. A completed portal
        action may retain its departure transaction until the next map
        snapshot proves real translation outside the source observation
        footprint; route success alone is not a doorway-crossing proof.
        """
        released_route_kind = self.active_route_kind
        released_route_id = int(getattr(self, "active_route_id", 0) or 0)
        # In persistent mode this method ends the graph lease before the
        # action client necessarily reaches DONE. Preserve the identity and
        # whether the graph already accepted a terminal so a delayed failure
        # can be consumed as controller cleanup rather than mistaken for a
        # second portal attempt.
        self.last_released_route_id = released_route_id
        self.last_released_route_kind = str(released_route_kind or "")
        self.last_released_route_terminal_received = bool(
            getattr(self, "active_terminal_received", False)
        )
        self.last_released_route_controller_pending = released_route_id > 0
        decision_scheduler = getattr(self, "decision_wake_scheduler", None)
        if decision_scheduler is not None:
            # Route IDs remain monotonic in the explorer after release, so the
            # scheduler receives the explicit lifecycle boundary here instead
            # of mistaking the latest ID for a still-live actuator lease.
            decision_scheduler.finish_route(
                getattr(self, "active_route_id", 0),
                "route_released",
            )
        clear_lease = getattr(self, "_clear_graph_route_plan_lease", None)
        if clear_lease is not None:
            clear_lease("active_route_released")
        transaction = getattr(self, "portal_transaction", None)
        if transaction is not None and released_route_kind == "portal_transition":
            pending_retry = getattr(self, "pending_portal_retry", None)
            if transaction.state in ("source_probe", "throat") and pending_retry is not None:
                transition = transaction.retry("portal_retry_queued")
                if transition is not None:
                    self.publish_status(
                        "portal_transaction_phase",
                        transaction_id=int(transition.transaction_id),
                        route_id=int(transition.route_id),
                        state=transition.state,
                        transition=transition.last_transition,
                        reason=transition.last_reason,
                        retry_count=int(transition.retry_count),
                    )
            elif transaction.state in ("source_probe", "throat"):
                transition = transaction.abort("portal_route_released_before_crossing")
                if transition is not None:
                    self.publish_status(
                        "portal_transaction_aborted",
                        transaction_id=int(transition.transaction_id),
                        route_id=int(transition.route_id),
                        state=transition.state,
                        reason=transition.last_reason,
                    )
                    transaction.finish("portal_transaction_aborted")
            elif transaction.state == "place_commit":
                transition = transaction.finish("portal_transaction_finished")
                if transition is not None:
                    self.publish_status(
                        "portal_transaction_finished",
                        transaction_id=int(transition.transaction_id),
                        portal_id=transition.portal_id,
                        source_place_id=transition.source_place_id,
                        state=transition.state,
                        reason=transition.last_reason,
                    )
        self.active_frontier = None
        self.active_frontier_component = None
        self.active_portal_gate_xy = None
        self.active_portal_gate_odom_xy = None
        self.active_portal_destination_odom_xy = None
        self.active_portal_gate_approached_at = None
        self.active_portal_crossing_observed = False
        self.active_portal_crossing_preobserved = False
        self.active_portal_crossing_rejected = False
        self.active_portal_probe_id = None
        self.active_portal_probe_phase = ""
        self.active_frontier_region_id = None
        self.active_work_item_id = None
        self.active_work_item_attempt_id = None
        self.active_work_item_place_id = None
        self.active_work_item_goal = None
        self.active_work_item_route_kind = None
        self.active_work_item_dispatch_announced = False
        self.active_place_hops = None
        self.active_observation_session_started_at = None
        # A route failure releases the uncommitted cross-place intent. A
        # successful route retains it only until physical boundary evidence
        # commits the source closure on the next planning snapshot.
        if not preserve_place_departure:
            self._clear_active_place_departure()
        self.active_since = 0.0
        self.active_best_distance = None
        self.active_best_goal_distance = None
        self.active_best_path_distance = None
        self.active_last_robot_xy = None
        self.active_start_odom_xy = None
        self.active_best_detour_odom_distance = 0.0
        self.active_visited_odom_cells.clear()
        self.clear_active_route_history()
        self.active_last_waypoint_map = None
        self.active_last_waypoint_yaw = None
        self.active_route_kind = "frontier_endpoint"
        self.active_portal_retry = False
        self.active_local_egress_resumes_portal = False
        self.active_terminal_received = False
        self.recovery_pending_route_id = 0
        self.recovery_pending_behavior = ""
        self.recovery_pending_reason = ""
        self.turn_connector_released = True
        self.last_status_command_map = None
        self.last_status_command_yaw = None
        self.last_status_mission_map = None
        self.active_last_progress_signal = "none"
        self.active_unreachable_since = None
        if discard_prefetch:
            self.clear_prefetched_frontier()

    def active_endpoint_observed_from_map(
        self, message, known_free, unknown, row, col, robot_map, distance,
    ):
        """Prove an ordinary endpoint was observed from the current standoff."""
        if (
            self.active_route_kind == "portal_transition"
            or self.scan_range_max is None
            or self.scan_range_max <= 0.0
            or float(distance) > self.scan_range_max + float(message.info.resolution)
        ):
            return False
        observer = self.xy_to_grid_cell(message, robot_map[0], robot_map[1])
        if observer is None:
            return False
        return grid_frontier_observed_from_viewpoint(
            known_free,
            unknown,
            observer[0],
            observer[1],
            row,
            col,
            self.dead_end_unknown_cells,
        )
