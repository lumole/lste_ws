"""Live TEB execution observations used by the action-health policy."""

import json
import time

import rospy


class TebGoalBridgeTebRuntimeMixin:
    def on_turn_supervisor_status(self, message):
        """Hold same-priority frontier replacements during an atomic turn."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        state = str(payload.get("state", "UNKNOWN")).strip().upper() or "UNKNOWN"
        event = str(payload.get("event", "unknown"))
        with self.lock:
            self.turn_supervisor_state = state
            self.turn_supervisor_last_event = event
            if (
                event in ("turn_completed", "turn_released")
                and self.action_active
                and self.active_route_kind == "frontier_turn_connector"
            ):
                # XY feedback is stationary during a mandated yaw connector.
                # Completing it is still action-health progress.
                self.active_progress_monotonic = time.monotonic()
                if self.active_feedback_distance is not None:
                    self.active_best_distance = self.active_feedback_distance
            if (
                event == "turn_completed"
                and self.action_active
                and self.active_route_kind == "frontier_turn_connector"
            ):
                self.turn_transition_ready = True
                self.dispatch_locked(
                    force=False, reason="turn_completed_route_release"
                )
        if state == "TURNING":
            rospy.loginfo_throttle(
                3.0,
                "TEB goal bridge holds frontier action while turn supervisor is TURNING",
            )

    def _reset_teb_reorientation_locked(self):
        """Forget a native TEB turn when its command/action boundary changes."""
        self.teb_reorientation_started_monotonic = 0.0
        self.teb_reorientation_reference_yaw = None
        self.teb_reorientation_last_yaw_progress_monotonic = 0.0
        self.teb_reorientation_total_yaw = 0.0

    def _teb_reorientation_command_active_locked(self, now):
        """Return true only for a fresh selected TEB in-place command."""
        return bool(
            self.teb_reorientation_enabled
            and self.latest_teb_selected_linear is not None
            and self.latest_teb_selected_angular is not None
            and now - self.latest_teb_feedback_monotonic
            <= self.teb_reorientation_feedback_timeout
            and abs(float(self.latest_teb_selected_linear))
            <= self.teb_reorientation_linear_threshold
            and abs(float(self.latest_teb_selected_angular))
            >= self.teb_reorientation_angular_threshold
        )

    def on_teb_feedback(self, message):
        """Track TEB's selected command without becoming a local controller."""
        trajectories = list(message.trajectories)
        selected_index = int(message.selected_trajectory_idx)
        selected = (
            trajectories[selected_index]
            if 0 <= selected_index < len(trajectories)
            else None
        )
        first = (
            selected.trajectory[0]
            if selected is not None and selected.trajectory
            else None
        )
        now = time.monotonic()
        with self.lock:
            if first is None:
                self.latest_teb_selected_linear = None
                self.latest_teb_selected_angular = None
                self.latest_teb_feedback_monotonic = 0.0
                self._reset_teb_reorientation_locked()
                return
            self.latest_teb_selected_linear = float(first.velocity.linear.x)
            self.latest_teb_selected_angular = float(first.velocity.angular.z)
            self.latest_teb_feedback_monotonic = now
            if not self._teb_reorientation_command_active_locked(now):
                self._reset_teb_reorientation_locked()
            elif (
                self.action_active
                and self.odom_yaw is not None
                and self.teb_reorientation_started_monotonic <= 0.0
            ):
                self.teb_reorientation_started_monotonic = now
                self.teb_reorientation_reference_yaw = float(self.odom_yaw)
                self.teb_reorientation_last_yaw_progress_monotonic = now
                self.teb_reorientation_total_yaw = 0.0

    def on_teb_planner_command(self, message):
        """Track raw TEB output used for endpoint lifecycle release."""
        now = time.monotonic()
        linear = float(message.linear.x)
        angular = float(message.angular.z)
        with self.lock:
            self.latest_teb_planner_linear = linear
            self.latest_teb_planner_angular = angular
            self.latest_teb_planner_command_monotonic = now
            if abs(linear) <= self.frontier_observation_completion_max_linear_speed:
                if self.teb_planner_stationary_since <= 0.0:
                    self.teb_planner_stationary_since = now
            else:
                self.teb_planner_stationary_since = 0.0

    def on_pose(self, message):
        """Measure real turn progress in odometry, independent of SLAM drift."""
        now = time.monotonic()
        yaw = float(message.theta)
        with self.lock:
            self.odom_yaw = yaw
            self.odom_pose_monotonic = now
            if (
                self.teb_reorientation_started_monotonic <= 0.0
                or self.teb_reorientation_reference_yaw is None
            ):
                return
            yaw_delta = abs(
                self._angle_delta(yaw, self.teb_reorientation_reference_yaw)
            )
            if yaw_delta >= self.teb_reorientation_yaw_progress:
                self.teb_reorientation_total_yaw += yaw_delta
                self.teb_reorientation_reference_yaw = yaw
                self.teb_reorientation_last_yaw_progress_monotonic = now

    def _teb_reorientation_progressing_locked(self, now):
        """Check the finite action-health exemption for a native TEB turn."""
        if (
            not self.action_active
            or self.turn_supervisor_state == "TURNING"
            or not self._teb_reorientation_command_active_locked(now)
            or self.odom_yaw is None
            or now - self.odom_pose_monotonic
            > self.teb_reorientation_feedback_timeout
        ):
            self._reset_teb_reorientation_locked()
            return False
        if self.teb_reorientation_started_monotonic <= 0.0:
            self.teb_reorientation_started_monotonic = now
            self.teb_reorientation_reference_yaw = float(self.odom_yaw)
            self.teb_reorientation_last_yaw_progress_monotonic = now
            self.teb_reorientation_total_yaw = 0.0
            return False
        duration = now - self.teb_reorientation_started_monotonic
        yaw_progress_age = now - self.teb_reorientation_last_yaw_progress_monotonic
        if (
            duration > self.teb_reorientation_max_extension
            or yaw_progress_age > self.teb_reorientation_stagnation_timeout
            or self.teb_reorientation_total_yaw < self.teb_reorientation_yaw_progress
        ):
            return False
        return True
