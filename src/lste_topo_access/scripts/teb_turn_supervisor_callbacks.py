"""ROS input adapters for :mod:`lste_teb_turn_supervisor`.

This mixin owns message decoding and cached runtime observations.  It never
decides a velocity itself; route and execution modules own those decisions.
"""

import copy
import json
import math
import time

import rospy
from geometry_msgs.msg import PoseStamped, Twist

from teb_turn_supervisor_contract import (
    FRONTIER_ENDPOINT_KIND,
    FRONTIER_SOURCE,
    LOCAL_EGRESS_KIND,
    STATE_TURNING,
    TURN_ROUTE_KIND,
    PORTAL_TRANSITION_KIND,
)


class TebTurnSupervisorCallbacksMixin:
    """Decode ROS callbacks into the supervisor's synchronized state."""

    def on_intent(self, message):
        source = "unknown"
        priority = 0
        route_kind = ""
        intent_goal = None
        target_track_id = ""
        try:
            payload = json.loads(message.data)
            if isinstance(payload, dict):
                source = str(payload.get("source", source)).strip().lower() or source
                priority = int(payload.get("priority", priority))
                route_kind = str(payload.get("route_kind", "")).strip().lower()
                target_track_id = str(
                    payload.get("target_track_id", "")
                ).strip()
                raw_goal = payload.get("goal")
                if isinstance(raw_goal, (list, tuple)) and len(raw_goal) >= 2:
                    intent_goal = (float(raw_goal[0]), float(raw_goal[1]))
            else:
                source = str(message.data).strip().lower() or source
        except (TypeError, ValueError, json.JSONDecodeError):
            source = str(message.data).strip().lower() or source
        with self.lock:
            self.latest_intent_source = source
            self.latest_intent_priority = max(0, min(3, priority))
            self.latest_route_kind = route_kind
            self.latest_intent_goal = intent_goal
            self.latest_target_track_id = target_track_id
            endpoint_intent = (
                route_kind == FRONTIER_ENDPOINT_KIND
                and source == FRONTIER_SOURCE
            )
            portal_intent = (
                route_kind == PORTAL_TRANSITION_KIND
                and source == FRONTIER_SOURCE
            )
            local_egress_intent = (
                route_kind == LOCAL_EGRESS_KIND
                and source == FRONTIER_SOURCE
            )
            if (
                route_kind != TURN_ROUTE_KIND
                and not endpoint_intent
                and not portal_intent
                and not local_egress_intent
            ):
                self._release_turn_locked("route_contract_released", completed=False)
            else:
                # Intent describes a pending mission decision. Bridge status
                # remains the authority that makes an action executable.
                self._activate_turn_locked()

    def on_bridge_status(self, message):
        """Synchronize execution state with the bridge's active action."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self.lock:
            previous_identity = self.active_action_identity
            self.active_action = bool(payload.get("active", False))
            self.active_action_route_kind = str(
                payload.get("active_route_kind", "")
            ).strip().lower()
            self.active_action_source = str(
                payload.get("active_intent_source", "unknown")
            ).strip().lower() or "unknown"
            self.active_action_priority = max(
                0, min(3, int(payload.get("active_intent_priority", 0) or 0))
            )
            self.active_action_target_track_id = str(
                payload.get("active_target_track_id", "")
            ).strip()
            self.latest_target_track_id = str(
                payload.get("latest_target_track_id", self.latest_target_track_id)
            ).strip()
            raw_goal = payload.get("active_goal")
            frame = str(payload.get("active_goal_frame", "") or "map").strip()
            if (
                isinstance(raw_goal, (list, tuple))
                and len(raw_goal) >= 2
                and self.active_action
            ):
                active_goal = PoseStamped()
                active_goal.header.stamp = rospy.Time.now()
                active_goal.header.frame_id = frame.lstrip("/") or "map"
                active_goal.pose.position.x = float(raw_goal[0])
                active_goal.pose.position.y = float(raw_goal[1])
                yaw = float(raw_goal[2]) if len(raw_goal) >= 3 else 0.0
                active_goal.pose.orientation.z = math.sin(0.5 * yaw)
                active_goal.pose.orientation.w = math.cos(0.5 * yaw)
                self.active_action_goal = active_goal
                raw_source_goal = payload.get("active_source_goal")
                source_frame = str(
                    payload.get("active_source_goal_frame", frame) or frame
                ).strip().lstrip("/") or "map"
                if (
                    isinstance(raw_source_goal, (list, tuple))
                    and len(raw_source_goal) >= 2
                ):
                    source_goal = PoseStamped()
                    source_goal.header.stamp = rospy.Time.now()
                    source_goal.header.frame_id = source_frame
                    source_goal.pose.position.x = float(raw_source_goal[0])
                    source_goal.pose.position.y = float(raw_source_goal[1])
                    source_yaw = (
                        float(raw_source_goal[2])
                        if len(raw_source_goal) >= 3 else 0.0
                    )
                    source_goal.pose.orientation.z = math.sin(0.5 * source_yaw)
                    source_goal.pose.orientation.w = math.cos(0.5 * source_yaw)
                    self.active_action_source_goal = source_goal
                else:
                    self.active_action_source_goal = None
            else:
                self.active_action_goal = None
                self.active_action_source_goal = None
            self.active_action_identity = self._active_action_key_locked()
            if self.active_action_identity != previous_identity:
                if (
                    previous_identity is not None
                    and self.state == STATE_TURNING
                    and self.turn_action_identity != self.active_action_identity
                ):
                    self._release_turn_locked(
                        "bridge_action_identity_changed",
                        completed=False,
                    )
                self.pre_turn_checked_identity = None
                self.completed_turn_key = None
                self.latest_navfn_plan = None
                self.trajectory_continuity_sharp_entry_identity = None
                self.stalled_route_candidate_identity = None
                self.stalled_route_candidate_since_wall = 0.0
                self.stalled_route_ready_identity = None
                self.stalled_route_completed_identity = None
            if not self._is_managed_action_locked():
                if self.state == STATE_TURNING:
                    self._release_turn_locked("bridge_action_changed", completed=False)
            else:
                self._activate_turn_locked()

    def on_goal(self, message):
        goal = copy.deepcopy(message)
        if not goal.header.frame_id:
            goal.header.frame_id = "odom"
        with self.lock:
            self.latest_goal = goal
            if self._is_managed_action_locked():
                activated = self._activate_turn_locked()
                if (
                    not activated
                    and self.state == STATE_TURNING
                    and not self._intent_matches_goal_locked(goal)
                ):
                    # A different frontier pose is a queued next segment, not
                    # permission to interrupt the current turn.
                    pass
            elif self.state == STATE_TURNING:
                self._release_turn_locked("goal_route_changed", completed=False)

    def on_navfn_plan(self, message):
        """Cache the active global path for one-time endpoint alignment."""
        with self.lock:
            self.latest_navfn_plan = copy.deepcopy(message)
            if (
                self._is_frontier_endpoint_action_locked()
                or self._is_portal_transition_action_locked()
            ):
                self._activate_turn_locked()

    def on_pose(self, message):
        with self.lock:
            self.pose = copy.deepcopy(message)

    def on_scan(self, message):
        minimum = float("inf")
        for value in message.ranges:
            if math.isfinite(value) and value > 0.01:
                minimum = min(minimum, float(value))
        with self.lock:
            self.scan_minimum = minimum
            self.scan_monotonic = time.monotonic()

    def on_planner_command(self, message):
        with self.lock:
            self.latest_planner_command = copy.deepcopy(message)
            self.latest_planner_command_wall = time.monotonic()

    def on_teb_feedback(self, message):
        """Cache the selected TEB velocity for a bounded raw-command gap."""
        selected_index = int(message.selected_trajectory_idx)
        trajectories = list(message.trajectories)
        if selected_index < 0 or selected_index >= len(trajectories):
            return
        points = list(trajectories[selected_index].trajectory)
        if not points:
            return
        velocity = points[0].velocity
        command = Twist()
        command.linear.x = float(velocity.linear.x)
        command.angular.z = float(velocity.angular.z)
        with self.lock:
            now = time.monotonic()
            if self.latest_trajectory_command_wall > 0.0:
                period = now - self.latest_trajectory_command_wall
                if 0.005 <= period <= self.trajectory_feedback_timeout_cap:
                    if self.trajectory_feedback_period_ema is None:
                        self.trajectory_feedback_period_ema = period
                    else:
                        self.trajectory_feedback_period_ema = (
                            0.75 * self.trajectory_feedback_period_ema
                            + 0.25 * period
                        )
            self.latest_trajectory_command = command
            self.latest_trajectory_command_wall = now
