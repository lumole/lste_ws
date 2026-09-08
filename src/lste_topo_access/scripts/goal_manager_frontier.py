"""Online-SLAM frontier route ownership for :mod:`lste_goal_manager`.

The GoalManager remains the ROS-node composition root.  This mixin owns only
the atomic frontier command contract and its explicit replan boundary; target
tracking and TEB action terminals deliberately live elsewhere.
"""

import json
import math

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from goal_context import (
    GOAL_CONTEXT_ROLES,
    goal_context_identity,
    normalize_goal_context,
)


class GoalManagerFrontierMixin:
    """Receive, invalidate, and explicitly replan map-connected routes."""

    _GOAL_CONTEXT_ROLES = GOAL_CONTEXT_ROLES

    @staticmethod
    def _goal_context_id(value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @classmethod
    def normalize_frontier_goal_context(cls, value):
        """Keep route identity explicit without trusting arbitrary JSON input."""
        return normalize_goal_context(value)

    def frontier_mission_goal_context(self):
        """Attach the LSTE task identity without changing physical ownership."""
        context = dict(self.global_frontier_goal_context)
        context["task_id"] = str(getattr(self, "current_task_id", "") or "").strip()
        context["mission_id"] = str(
            getattr(self, "current_mission_id", "") or context["task_id"]
        ).strip()
        context["task_version"] = str(
            getattr(self, "current_task_version", "") or ""
        ).strip()
        return normalize_goal_context(context)

    def frontier_goal_context_changed(self):
        """Return whether a same-coordinate route has a different owner."""
        return goal_context_identity(self.frontier_mission_goal_context()) != (
            goal_context_identity(getattr(self, "last_frontier_goal_context", None))
        )

    def on_global_frontier(self, msg: PoseStamped):
        # The normal online planner publishes a complete route-command
        # transaction. Retain this topic only for RViz and old integrations;
        # accepting both would restore the cross-topic metadata race.
        if self.global_frontier_command_enabled:
            return
        frame = (msg.header.frame_id or "odom").strip().lstrip("/")
        if frame not in ("odom", "map"):
            rospy.logwarn_throttle(
                3.0, "Ignoring global frontier in unsupported frame: %s", msg.header.frame_id
            )
            return
        self.latest_global_frontier_goal = msg
        self.last_frontier_goal = msg
        key = (
            round(float(msg.pose.position.x), 3),
            round(float(msg.pose.position.y), 3),
        )
        # The explorer includes its stable route transaction in header.seq.
        # This is an atomic PoseStamped delivery, unlike the older status plus
        # rounded-coordinate lookup which could see the pose first and silently
        # downgrade a valid continuation to route_id=0.
        embedded_route_id = int(msg.header.seq or 0)
        if embedded_route_id > 0:
            self.global_frontier_route_id = embedded_route_id
        else:
            route_id = self.global_frontier_route_ids_by_goal.get(key)
            # Keep the legacy lookup for external or old frontier publishers.
            self.global_frontier_route_id = 0 if route_id is None else int(route_id)
        self.global_frontier_goal_context = self.normalize_frontier_goal_context(None)
        # A terminal means the next fresh frontier message is actionable now.
        if self.teb_terminal_goal is not None:
            self.next_update_time = 0.0

    def on_global_frontier_command(self, message: String):
        """Consume one atomic online-frontier route transaction."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("event") != "route_command":
            return
        goal_xy = payload.get("goal")
        if not isinstance(goal_xy, (list, tuple)) or len(goal_xy) < 2:
            return
        try:
            x, y = float(goal_xy[0]), float(goal_xy[1])
            route_id = max(0, int(payload.get("route_id", 0) or 0))
        except (TypeError, ValueError):
            return
        frame = str(payload.get("frame_id", "map")).strip().lstrip("/") or "map"
        if frame not in ("odom", "map"):
            rospy.logwarn_throttle(
                3.0, "Ignoring global frontier command in unsupported frame: %s", frame
            )
            return
        yaw = payload.get("yaw")
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = frame
        goal.pose.position.x = x
        goal.pose.position.y = y
        if yaw is None:
            goal.pose.orientation.w = 1.0
        else:
            try:
                yaw = float(yaw)
            except (TypeError, ValueError):
                return
            goal.pose.orientation.z = math.sin(0.5 * yaw)
            goal.pose.orientation.w = math.cos(0.5 * yaw)
        self.latest_global_frontier_goal = goal
        self.last_frontier_goal = goal
        self.global_frontier_route_id = route_id
        self.global_frontier_goal_context = self.normalize_frontier_goal_context(
            payload.get("goal_context")
        )
        self.global_frontier_transition_kind = str(
            payload.get("transition_kind", "unknown")
        ).strip().lower() or "unknown"
        try:
            self.global_frontier_predecessor_route_id = max(
                0, int(payload.get("predecessor_route_id", 0) or 0)
            )
        except (TypeError, ValueError):
            self.global_frontier_predecessor_route_id = 0
        try:
            transition_distance = payload.get(
                "transition_distance_to_previous_endpoint"
            )
            self.global_frontier_transition_distance = (
                None if transition_distance is None else float(transition_distance)
            )
        except (TypeError, ValueError):
            self.global_frontier_transition_distance = None
        route_kind = str(payload.get("route_kind", "")).strip().lower()
        if route_kind in (
            "frontier_connector",
            "frontier_turn_connector",
            "frontier_endpoint",
            "portal_transition",
            "local_egress",
            "portal_probe",
        ):
            self.global_frontier_route_kind = route_kind
        mission_route_kind = str(
            payload.get("mission_route_kind", route_kind)
        ).strip().lower()
        if mission_route_kind in (
            "frontier_endpoint",
            "portal_transition",
            "local_egress",
        ):
            self.global_frontier_mission_route_kind = mission_route_kind
        # A completed frontier segment can start the next map-connected one
        # immediately. A visual target owns its post-arrival observation
        # transaction, so that fast path must defer until it releases.
        now = rospy.Time.now().to_sec()
        target_active = self.target_tracking_active(now)
        target_segment_active = self.target_segment_ownership_active()
        target_ownership_active = (
            self.target_terminal_reobserve_pending
            or self.navigation_hold_active
            or target_active
            or target_segment_active
        )
        terminal_frontier_fast_path = (
            self.teb_terminal_goal is not None
            and self.controller_mode == "teb"
            and self.last_goal_source == "global_slam_frontier"
        )
        if terminal_frontier_fast_path and target_ownership_active:
            target_age = None
            if self.target_last_seen is not None:
                target_age = round(max(0.0, now - self.target_last_seen), 3)
            self.next_update_time = 0.0
            self.publish_goal_arbitration(
                "frontier_command_deferred_target_ownership",
                candidate_goal=[round(x, 3), round(y, 3)],
                candidate_route_id=route_id,
                candidate_route_kind=route_kind,
                target_track_id=self.target_track_id,
                target_active=bool(target_active),
                target_segment_active=bool(target_segment_active),
                target_terminal_reobserve_pending=bool(
                    self.target_terminal_reobserve_pending
                ),
                navigation_hold=bool(self.navigation_hold_active),
                target_age_seconds=target_age,
            )
            rospy.loginfo(
                "GoalManager: defer frontier route=%d while target owns "
                "terminal observation (track=%s active=%s hold=%s)",
                route_id,
                self.target_track_id or "-",
                target_active,
                self.navigation_hold_active,
            )
            return
        if terminal_frontier_fast_path:
            self.goal_source = "global_slam_frontier"
            self.publish_goal(goal)
        else:
            self.next_update_time = 0.0

    def on_global_frontier_status(self, message: String):
        """Synchronize mission ownership when the explorer abandons a route."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        event = str(payload.get("event", "")).strip()
        route_kind = str(payload.get("route_kind", "")).strip().lower()
        # Status messages are an audit stream. A WorkItem event can report
        # its mission kind after an execution command arrived on a different
        # ROS connection, so only route_command may update the live phase.
        if event == "route_command" and route_kind in (
            "frontier_connector",
            "frontier_turn_connector",
            "frontier_endpoint",
            "portal_transition",
            "local_egress",
        ):
            self.global_frontier_route_kind = route_kind
            mission_route_kind = str(
                payload.get("mission_route_kind", route_kind)
            ).strip().lower()
            if mission_route_kind in (
                "frontier_endpoint",
                "portal_transition",
                "local_egress",
                "portal_probe",
            ):
                self.global_frontier_mission_route_kind = mission_route_kind
        route_id = int(payload.get("route_id", 0) or 0)
        command_goal = payload.get("command_goal")
        if (
            route_id > 0
            and isinstance(command_goal, (list, tuple))
            and len(command_goal) >= 2
        ):
            key = (round(float(command_goal[0]), 3), round(float(command_goal[1]), 3))
            self.global_frontier_route_ids_by_goal[key] = route_id
            # The topic pair is normally emitted status -> pose in one frontier
            # cycle. Retain only a small current history to bound this lookup.
            if len(self.global_frontier_route_ids_by_goal) > 32:
                oldest = next(iter(self.global_frontier_route_ids_by_goal))
                self.global_frontier_route_ids_by_goal.pop(oldest, None)
        if payload.get("event") == "replan_ready":
            request_id = int(payload.get("replan_request_id", 0) or 0)
            if (
                self.frontier_replan_pending_id > 0
                and request_id == self.frontier_replan_pending_id
            ):
                self.frontier_replan_ready_id = request_id
                self.next_update_time = 0.0
                self.publish_goal_arbitration(
                    "frontier_replan_ready",
                    replan_request_id=request_id,
                    reason=str(payload.get("reason", "unknown")),
                    goal=payload.get("goal"),
                )
                rospy.loginfo(
                    "GoalManager: received fresh frontier replan id=%d goal=%s",
                    request_id,
                    payload.get("goal"),
                )
            return
        if event not in (
            "route_invalidated",
            "frontier_exhausted",
            "frontier_route_unavailable",
        ):
            return
        with_status_goal = payload.get("goal")
        rospy.logwarn(
            "GoalManager: releasing frontier ownership event=%s reason=%s goal=%s",
            event,
            payload.get("reason", "unknown"),
            with_status_goal,
        )
        self.latest_global_frontier_goal = None
        self.last_frontier_goal = None
        self.global_frontier_route_kind = "frontier_endpoint"
        self.global_frontier_mission_route_kind = "frontier_endpoint"
        self.global_frontier_route_id = 0
        self.teb_terminal_goal = None
        self.teb_frontier_goal_history = []
        self.frontier_goal_sent_at = None
        if (
            self.last_goal_source == "global_slam_frontier"
            and not self.target_tracking_active(rospy.Time.now().to_sec())
            and not self.target_segment_ownership_active()
        ):
            self.last_goal = None
            self.last_goal_source = "waiting_global_slam_frontier"
        self.goal_source = "waiting_global_slam_frontier"
        self.next_update_time = 0.0

    def request_global_frontier_replan(self, reason: str, **fields):
        """Request a branch recomputed from the current robot pose."""
        if not self.global_frontier_enabled:
            return 0
        target_room_claim_release = bool(
            fields.get("target_room_claim_release", False)
        )
        if (
            self.frontier_replan_pending_id > 0
            and self.frontier_replan_ready_id != self.frontier_replan_pending_id
            and not target_room_claim_release
        ):
            return self.frontier_replan_pending_id
        self.frontier_replan_request_id += 1
        request_id = self.frontier_replan_request_id
        self.frontier_replan_pending_id = request_id
        self.frontier_replan_ready_id = 0
        # Do not accidentally re-publish a latched goal while the frontier
        # explorer is rebuilding its connected route from the latest map.
        self.latest_global_frontier_goal = None
        self.last_frontier_goal = None
        payload = {
            "event": "replan_request",
            "request_id": request_id,
            "reason": str(reason),
        }
        payload.update(fields)
        self.pub_global_frontier_replan.publish(
            String(data=json.dumps(payload, sort_keys=True))
        )
        self.publish_goal_arbitration(
            "frontier_replan_requested",
            replan_request_id=request_id,
            reason=str(reason),
            **fields
        )
        rospy.loginfo(
            "GoalManager: requested fresh global frontier replan id=%d reason=%s",
            request_id,
            reason,
        )
        return request_id

    def release_target_room_claim(self, reason: str, target_track_id: str = ""):
        """End one target-place lease and request an unrestricted successor.

        Candidate claims protect a place while direct visual evidence remains
        alive. They are not a permanent exploration state. Releasing through
        the same transaction channel is the one edge at which the frontier
        planner may again choose an outward doorway transition.
        """
        return self.request_global_frontier_replan(
            "target_room_claim_release",
            target_room_claim_release=True,
            target_room_claim=False,
            target_room_claim_release_reason=str(reason),
            target_track_id=str(target_track_id or ""),
        )
