"""Command continuity and timer-driven output for the TEB turn supervisor."""

import copy
import math
import time

import rospy
from geometry_msgs.msg import Twist

from teb_turn_supervisor_contract import (
    STATE_PASS_THROUGH,
    STATE_TURNING,
    normalize_angle,
)


class TebTurnSupervisorControlMixin:
    """Forward TEB commands, with narrowly bounded execution-phase handling."""

    def _trajectory_feedback_timeout_locked(self):
        """Return bounded freshness based on the observed feedback cadence."""
        if self.trajectory_feedback_period_ema is None:
            return self.trajectory_feedback_timeout
        return min(
            self.trajectory_feedback_timeout_cap,
            max(
                self.trajectory_feedback_timeout,
                self.trajectory_feedback_period_scale
                * self.trajectory_feedback_period_ema,
            ),
        )

    def _trajectory_continuity_command_locked(self, now, planner_command_stale=False):
        """Forward one fresh, selected TEB command across a transient raw gap.

        The adapter never creates a velocity.  It can only pass through TEB's
        selected forward sample for a short bounded interval, outside terminal
        and explicit-turn envelopes.
        """
        raw = self.latest_planner_command
        raw_linear_is_zero = abs(float(raw.linear.x)) <= 0.01
        raw_is_zero = raw_linear_is_zero and abs(float(raw.angular.z)) <= 0.01
        raw_is_reverse_only = (
            float(raw.linear.x) < -0.01
            and abs(float(raw.angular.z)) <= 0.01
        )
        transient_gap = raw_is_zero or raw_is_reverse_only or planner_command_stale
        sharp_entry_error = (
            self._sharp_navfn_entry_heading_error_locked()
            if transient_gap else None
        )
        if sharp_entry_error is not None:
            action_identity = self.active_action_identity
            if action_identity != self.trajectory_continuity_sharp_entry_identity:
                self.trajectory_continuity_sharp_entry_identity = action_identity
                self.trajectory_continuity_sharp_entry_suppressions += 1
                self.publish_status_locked(
                    "trajectory_continuity_suppressed",
                    reason="sharp_navfn_entry",
                    entry_heading_error_deg=round(
                        math.degrees(sharp_entry_error), 2
                    ),
                    threshold_deg=90.0,
                    raw_command=[
                        round(float(raw.linear.x), 4),
                        round(float(raw.angular.z), 4),
                    ],
                    count=int(self.trajectory_continuity_sharp_entry_suppressions),
                )
            self.trajectory_zero_started_wall = 0.0
            return None
        goal_distance = self._goal_distance_in_pose_frame_locked(
            self.active_action_goal
        )
        feedback_age = max(0.0, now - self.latest_trajectory_command_wall)
        feedback_timeout = self._trajectory_feedback_timeout_locked()
        self.trajectory_continuity_goal_distance = goal_distance
        action_or_same_tier_intent = (
            self._intent_matches_goal_locked(
                self.active_action_source_goal or self.active_action_goal
            )
            or (
                self.latest_intent_source == self.active_action_source
                and self.latest_intent_priority == self.active_action_priority
            )
        )
        if not (
            self.trajectory_continuity_enabled
            and self.state == STATE_PASS_THROUGH
            and self._is_continuity_eligible_action_locked()
            and action_or_same_tier_intent
            and not self.task_done
            and not self.navigation_hold
            and transient_gap
            and feedback_age <= feedback_timeout
            and float(self.latest_trajectory_command.linear.x)
            >= self.trajectory_continuity_min_forward
            and goal_distance is not None
            and goal_distance > self.trajectory_continuity_terminal_radius
        ):
            self.trajectory_zero_started_wall = 0.0
            return None
        if self.trajectory_zero_started_wall <= 0.0:
            self.trajectory_zero_started_wall = now
        if now - self.trajectory_zero_started_wall > self.trajectory_continuity_max_hold:
            return None
        self.trajectory_continuity_events += 1
        self.publish_status_locked(
            "trajectory_continuity",
            continuity_scope=(
                "target_same_track"
                if self._is_same_target_segment_continuation_locked()
                else "frontier_or_connector"
            ),
            gap_kind=(
                "planner_publish_gap"
                if planner_command_stale
                else ("reverse_only" if raw_is_reverse_only else "zero")
            ),
            raw_command=[
                round(float(raw.linear.x), 4),
                round(float(raw.angular.z), 4),
            ],
            selected_command=[
                round(float(self.latest_trajectory_command.linear.x), 4),
                round(float(self.latest_trajectory_command.angular.z), 4),
            ],
            feedback_age_seconds=round(feedback_age, 4),
            feedback_timeout_seconds=round(feedback_timeout, 4),
            feedback_period_seconds=(
                None
                if self.trajectory_feedback_period_ema is None
                else round(float(self.trajectory_feedback_period_ema), 4)
            ),
            action_goal_distance=round(float(goal_distance), 4),
            count=int(self.trajectory_continuity_events),
        )
        return copy.deepcopy(self.latest_trajectory_command)

    def on_mode(self, message):
        mode = message.data.strip().lower()
        if not mode:
            return
        with self.lock:
            previous = self.mode
            self.mode = mode
            if mode != self.active_mode and self.state == STATE_TURNING:
                self._release_turn_locked(
                    "controller_switched_to_%s" % mode, completed=False
                )
            elif previous != self.active_mode and mode == self.active_mode:
                self._activate_turn_locked()

    def on_task_done(self, message):
        with self.lock:
            self.task_done = bool(message.data)
            if self.task_done:
                self._release_turn_locked("task_done", completed=False)
            else:
                self.completed_turn_key = None
                self._activate_turn_locked()

    def on_navigation_hold(self, message):
        with self.lock:
            self.navigation_hold = bool(message.data)
            if self.navigation_hold:
                self._release_turn_locked("navigation_hold", completed=False)
            else:
                self._activate_turn_locked()

    def _turn_command_locked(self, dt):
        """Return one acceleration-limited turn command."""
        if self.pose is None or self.turn_target_yaw is None:
            self.turn_velocity = 0.0
            return Twist()
        error = normalize_angle(self.turn_target_yaw - self.pose.theta)
        if abs(error) <= self.yaw_goal_tolerance:
            self._release_turn_locked("yaw_tolerance_reached", completed=True)
            return copy.deepcopy(self.latest_planner_command)

        remaining = max(0.0, abs(error) - self.yaw_goal_tolerance)
        desired = min(
            self.max_vel_theta,
            math.sqrt(max(0.0, 2.0 * self.acc_lim_theta * remaining)),
        )
        requested_sign = 1.0 if error > 0.0 else -1.0
        previous = float(self.turn_velocity)
        step = self.acc_lim_theta * max(0.001, min(0.2, float(dt)))
        if previous != 0.0 and math.copysign(1.0, previous) != requested_sign:
            magnitude = max(0.0, abs(previous) - step)
            self.turn_velocity = (
                math.copysign(magnitude, previous) if magnitude else 0.0
            )
        else:
            previous_magnitude = abs(previous)
            if desired >= previous_magnitude:
                magnitude = min(desired, previous_magnitude + step)
            else:
                magnitude = max(desired, previous_magnitude - step)
            self.turn_velocity = requested_sign * magnitude

        self.turn_abs_rotation += abs(self.turn_velocity) * max(0.001, float(dt))
        if self.turn_abs_rotation > self.turn_rotation_cap:
            self.turn_capped_releases += 1
            rospy.logwarn(
                "TEB turn supervisor forced turn release: abs_rotation=%.0fdeg "
                "cap=%.0fdeg initial_error=%.0fdeg",
                math.degrees(self.turn_abs_rotation),
                math.degrees(self.turn_rotation_cap),
                math.degrees(self.turn_initial_abs_error),
            )
            self._release_turn_locked("rotation_budget_exceeded", completed=False)
            return copy.deepcopy(self.latest_planner_command)

        command = Twist()
        command.angular.z = self.turn_velocity
        return command

    def on_timer(self, _event):
        with self.lock:
            if (
                self.mode == self.active_mode
                and not self.task_done
                and not self.navigation_hold
                and self.state == STATE_TURNING
            ):
                clearance_reason = self._turn_clearance_reason_locked(
                    time.monotonic()
                )
                if clearance_reason:
                    self._release_turn_locked(
                        "turn_clearance_lost_%s" % clearance_reason,
                        completed=False,
                    )
                    command = copy.deepcopy(self.latest_planner_command)
                else:
                    command = self._turn_command_locked(
                        1.0 / self.command_frequency
                    )
                state = self.state
            elif (
                self.mode == self.active_mode
                and not self.task_done
                and not self.navigation_hold
            ):
                planner_command_stale = (
                    time.monotonic() - self.latest_planner_command_wall
                    > self.planner_command_timeout
                )
                if not planner_command_stale:
                    command = copy.deepcopy(self.latest_planner_command)
                    continuity = self._trajectory_continuity_command_locked(
                        time.monotonic()
                    )
                    if continuity is not None:
                        command = continuity
                else:
                    command = Twist()
                    continuity = self._trajectory_continuity_command_locked(
                        time.monotonic(), planner_command_stale=True
                    )
                    if continuity is not None:
                        command = continuity
                if (
                    self.turn_settle_until_wall > 0.0
                    and self.pose is not None
                    and self.turn_settle_yaw is not None
                ):
                    if time.monotonic() < self.turn_settle_until_wall:
                        yaw_err = normalize_angle(
                            self.turn_settle_yaw - self.pose.theta
                        )
                        command = copy.deepcopy(command)
                        command.angular.z = max(-0.20, min(0.20, yaw_err * 0.6))
                    else:
                        self.turn_settle_until_wall = 0.0
                        self.turn_settle_yaw = None
                state = self.state
            else:
                command = Twist()
                self.trajectory_zero_started_wall = 0.0
                state = self.state
            if (
                state == STATE_PASS_THROUGH
                and self._is_managed_action_locked()
                and not self.task_done
                and not self.navigation_hold
            ):
                self._activate_turn_locked()
        self.output_pub.publish(command)
        now = time.monotonic()
        with self.lock:
            if now - self.last_status_wall >= 1.0:
                self.last_status_wall = now
                self.publish_status_locked(
                    "heartbeat",
                    command=[
                        round(float(command.linear.x), 4),
                        round(float(command.angular.z), 4),
                    ],
                )

    def on_shutdown(self):
        try:
            self.output_pub.publish(Twist())
        except Exception:
            pass
