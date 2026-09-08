"""Timer-driven persistent-mission progression for the TEB bridge."""

import copy
import time

import rospy


class TebGoalBridgeMissionRuntimeMixin:
    def on_timer(self, _event):
        with self.lock:
            if self._is_active_mode():
                if self.persistent_execution:
                    self._promote_persistent_frontier_prefetch_locked()
                else:
                    self.maybe_handoff_locked()
                self.dispatch_locked(force=False, reason="coalesced_global_goal")

    def _adopt_persistent_mission_goal_locked(self):
        """Advance the semantic route while retaining the actionlib lease."""
        if self.latest_goal is None or not self.action_active:
            return
        semantic_goal = self._goal_in_global_frame(self.latest_goal)
        if semantic_goal is None:
            return
        previous_route_kind = self.active_route_kind
        previous_route_id = int(self.active_route_id)
        self.active_goal_global = copy.deepcopy(semantic_goal)
        self.last_dispatched_goal = copy.deepcopy(semantic_goal)
        # Persistent execution changes the semantic route without creating a
        # new actionlib goal. Keep idle retry diagnostics aligned with the
        # latest route represented by ``last_dispatched_goal``.
        self._remember_last_dispatch_identity_locked()
        self.active_intent_source = self.latest_intent_source
        self.active_intent_priority = int(self.latest_intent_priority)
        self.active_goal_transaction_id = int(self.latest_goal_transaction_id)
        self.active_route_kind = self.latest_route_kind
        self.active_route_id = int(self.latest_route_id)
        if self.active_route_kind == "portal_transition":
            if (
                previous_route_kind != "portal_transition"
                or previous_route_id != self.active_route_id
                or getattr(self, "active_portal_source_goal", None) is None
            ):
                self.active_portal_source_goal = copy.deepcopy(self.latest_goal)
        else:
            self.active_portal_source_goal = None
        self.active_target_epoch = int(self.latest_target_epoch)
        self.active_target_track_id = self.latest_target_track_id
        self.active_target_viewpoint_candidate_id = str(
            getattr(self, "latest_target_viewpoint_candidate_id", "") or ""
        )
        self.active_target_viewpoint_attempt_id = str(
            getattr(self, "latest_target_viewpoint_attempt_id", "") or ""
        )
        self.active_goal_context = dict(self.latest_goal_context)
        self.active_best_distance = None
        self.active_progress_monotonic = time.monotonic()
        self.active_motion_reference = None
        self.active_motion_progress_monotonic = time.monotonic()
        self.active_navfn_plan_points = []
        self.active_navfn_plan_endpoint = None
        self.active_navfn_remaining = None
        self.active_navfn_best_remaining = None
        self.active_navfn_progress_monotonic = 0.0
        self.publish_bridge_status(
            "persistent_mission_path_adopted",
            route_id=int(self.active_route_id),
            priority=int(self.active_intent_priority),
            source=self.active_intent_source,
            transaction_id=int(self.latest_goal_transaction_id),
            target_track_id=self.active_target_track_id,
            goal_context=self.active_goal_context,
            goal=[
                round(float(semantic_goal.pose.position.x), 3),
                round(float(semantic_goal.pose.position.y), 3),
            ],
        )

    def _promote_persistent_frontier_prefetch_locked(self):
        """Release one pre-admitted successor before a route-end stop."""
        if (
            not self.persistent_execution
            or not self.action_active
            or self.task_done
            or self.active_intent_priority != 0
            or self.latest_intent_priority != 0
            or self.active_intent_source != "global_slam_frontier"
            or self.latest_intent_source != "global_slam_frontier"
            or self.active_route_kind != "frontier_endpoint"
            or self.active_route_id <= 0
            or self.active_feedback_distance is None
            or self.active_feedback_distance
            > (
                self.persistent_frontier_lookahead_trigger_distance
                if self.persistent_frontier_lookahead_handoff_enabled
                else self.frontier_observation_completion_radius
            )
            or self.last_dispatched_goal is None
            or self.prefetched_frontier_goal is None
            or self.prefetched_frontier_route_id != self.active_route_id + 1
        ):
            return False
        route_pair = (int(self.active_route_id), int(self.prefetched_frontier_route_id))
        if route_pair in self.persistent_frontier_prefetch_promoted_pairs:
            return False
        admission = None
        if self.persistent_frontier_lookahead_handoff_enabled:
            admitted, admission = self._admit_persistent_frontier_prefetch_locked()
            if not admitted:
                now = time.monotonic()
                if now - self.persistent_prefetch_admission_last_report_monotonic >= 1.0:
                    self.persistent_prefetch_admission_last_report_monotonic = now
                    self.publish_bridge_status(
                        "persistent_frontier_prefetch_admission_deferred",
                        **admission,
                    )
                return False
        elif not self._frontier_observation_speed_ready_locked():
            # Legacy persistent behavior waits for TEB's terminal speed envelope.
            return False
        source_goal = copy.deepcopy(self.last_dispatched_goal)
        feedback_distance = float(self.active_feedback_distance)
        self.persistent_frontier_prefetch_promoted_pairs.add(route_pair)
        self._publish_execution_terminal_locked(source_goal)
        self._clear_target_failure_locked("persistent_frontier_prefetch_progress")
        # The terminal event releases the successor. StreamingNavfnPlanner
        # updates the route under the existing move_base action, with no cancel.
        self.terminal_count += 1
        transition_kind = (
            str(admission.get("transition_kind", "smooth_handoff"))
            if isinstance(admission, dict)
            else "smooth_handoff"
        )
        self.publish_bridge_status(
            (
                "persistent_frontier_curve_handoff_promoted"
                if transition_kind == "curve_handoff"
                else (
                    "persistent_frontier_lookahead_handoff_promoted"
                    if self.persistent_frontier_lookahead_handoff_enabled
                    else "persistent_frontier_prefetch_promoted"
                )
            ),
            route_id=route_pair[0],
            successor_route_id=route_pair[1],
            feedback_distance=round(feedback_distance, 3),
            completion_radius=round(self.frontier_observation_completion_radius, 3),
            lookahead_trigger_distance=round(
                self.persistent_frontier_lookahead_trigger_distance, 3
            ),
            selected_linear_speed=round(float(self.latest_teb_selected_linear), 3),
            completion_max_linear_speed=round(
                float(self.frontier_observation_completion_max_linear_speed), 3
            ),
            lifecycle=(
                "pre_admitted_%s_then_streaming_plan_update" % transition_kind
                if self.persistent_frontier_lookahead_handoff_enabled
                else "logical_terminal_then_streaming_plan_update"
            ),
            transition_kind=transition_kind,
            admission=admission,
        )
        rospy.loginfo(
            "TEB goal bridge promoted persistent frontier successor: "
            "route_id=%d successor_route_id=%d distance=%.2fm lookahead=%s",
            route_pair[0],
            route_pair[1],
            feedback_distance,
            self.persistent_frontier_lookahead_handoff_enabled,
        )
        return True

    def on_terminal_timer(self, _event):
        with self.lock:
            self.terminal_dispatch_timer = None
            if self._is_active_mode():
                self.dispatch_locked(force=False, reason="terminal_followup")

    def schedule_terminal_dispatch_locked(self):
        if self.terminal_dispatch_timer is None and not rospy.is_shutdown():
            self.terminal_dispatch_timer = rospy.Timer(
                # This is a lifecycle synchronization barrier, not a planning
                # delay. It gives actionlib one callback turn to finish DONE.
                rospy.Duration(0.10), self.on_terminal_timer, oneshot=True
            )
