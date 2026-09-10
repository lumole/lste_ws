"""ROS input adapters for :mod:`lste_teb_turn_supervisor`.

This mixin owns message decoding and cached runtime observations.  It never
decides a velocity itself; route and execution modules own those decisions.
"""

import copy
import json
import math

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from lste_topo_access.msg import PlannerCommandContract

from clock_provider import now_for

from teb_turn_supervisor_contract import (
    FRONTIER_ENDPOINT_KIND,
    FRONTIER_SOURCE,
    LOCAL_EGRESS_KIND,
    PORTAL_PROBE_KIND,
    STATE_TURNING,
    TURN_ROUTE_KIND,
    PORTAL_TRANSITION_KIND,
)


class TebTurnSupervisorCallbacksMixin:
    """Decode ROS callbacks into the supervisor's synchronized state."""

    @staticmethod
    def _planner_contract_identity(message):
        """Decode a typed planner frame into scalar identity fields."""
        def integer(value, default=0):
            try:
                return int(value or default)
            except (TypeError, ValueError):
                return int(default)

        return {
            "transaction_id": integer(getattr(message, "transaction_id", 0)),
            "route_id": integer(getattr(message, "route_id", 0)),
            "graph_transaction_id": integer(
                getattr(message, "graph_transaction_id", 0)
            ),
            "map_epoch": integer(getattr(message, "map_epoch", 0)),
            "lifecycle_transaction_id": integer(
                getattr(message, "lifecycle_transaction_id", "")
            ),
            "action_generation": integer(
                getattr(message, "action_generation", 0)
            ),
            "planner_sequence": integer(
                getattr(message, "planner_sequence", 0)
            ),
            "state": integer(getattr(message, "state", 0)),
            "producer": str(getattr(message, "producer", "") or ""),
        }

    @staticmethod
    def _planner_contract_command(message):
        command = Twist()
        source = getattr(message, "command", None)
        if source is not None:
            command.linear.x = float(source.linear.x)
            command.angular.z = float(source.angular.z)
        return command

    def _planner_contract_alignment_pending_locked(self, identity):
        """Return whether a newer frame is waiting for Bridge status."""
        if not self.active_action:
            return True
        if self._lifecycle_transaction_id_locked() <= 0:
            return True
        for field, active_field in (
            ("transaction_id", "active_action_goal_transaction_id"),
            ("route_id", "active_action_route_id"),
            ("graph_transaction_id", "active_action_graph_transaction_id"),
            ("action_generation", "active_action_generation"),
        ):
            active_value = int(getattr(self, active_field, 0) or 0)
            received_value = int(identity[field] or 0)
            if active_value <= 0:
                return True
            if received_value > active_value:
                return True
        active_epoch = getattr(self, "active_action_map_epoch", None)
        received_epoch = int(identity.get("map_epoch", 0) or 0)
        if active_epoch is None:
            return received_epoch > 0
        try:
            if received_epoch > int(active_epoch):
                return True
        except (TypeError, ValueError):
            return True
        return False

    def _planner_contract_mismatch_locked(self, identity):
        current_lifecycle = self._lifecycle_transaction_id_locked()
        if identity["lifecycle_transaction_id"] <= 0:
            return "lifecycle_transaction_id_missing"
        if (
            current_lifecycle <= 0
            or identity["lifecycle_transaction_id"] != current_lifecycle
        ):
            return "lifecycle_transaction_id_mismatch"
        if identity["action_generation"] <= 0:
            return "action_generation_missing"
        if not self.active_action:
            return "bridge_action_inactive"
        for field, active_field in (
            ("transaction_id", "active_action_goal_transaction_id"),
            ("route_id", "active_action_route_id"),
            ("graph_transaction_id", "active_action_graph_transaction_id"),
            ("action_generation", "active_action_generation"),
        ):
            active_value = int(getattr(self, active_field, 0) or 0)
            if active_value > 0 and identity[field] != active_value:
                return "planner_%s_mismatch" % field
        active_epoch = getattr(self, "active_action_map_epoch", None)
        if active_epoch is not None and identity["map_epoch"] != int(active_epoch):
            return "planner_map_epoch_mismatch"
        if not identity["producer"]:
            return "planner_producer_missing"
        return ""

    def _store_planner_contract_locked(
        self, identity, command, state, reason, producer
    ):
        self.planner_contract_valid = state == PlannerCommandContract.STATE_ACTIVE
        self.planner_contract_state = int(state)
        self.planner_contract_reason = str(reason or "")
        self.planner_contract_identity = dict(identity)
        self.planner_contract_sequence = int(identity["planner_sequence"])
        self.planner_contract_wall = now_for(self)
        self.planner_contract_command = copy.deepcopy(command)
        self.planner_contract_awaiting_feedback = bool(
            self.planner_contract_valid
        )
        self.latest_planner_command = copy.deepcopy(command)
        self.latest_planner_command_wall = self.planner_contract_wall
        self.planner_command_transaction_id = int(
            identity["lifecycle_transaction_id"]
        )
        self.planner_command_route_id = int(identity["route_id"])
        self.planner_command_map_epoch = int(identity["map_epoch"])
        self.planner_command_graph_transaction_id = int(
            identity["graph_transaction_id"]
        )
        self.planner_command_action_generation = int(
            identity["action_generation"]
        )
        self.planner_command_action_identity = self.active_action_identity
        if self.planner_contract_valid:
            self._invalidate_trajectory_feedback_locked(
                "planner_contract_command_changed", clear_planner=False
            )
        else:
            self._invalidate_trajectory_feedback_locked(
                "planner_contract_%s" % (reason or "boundary"),
                clear_planner=True,
            )

    def _apply_planner_contract_locked(
        self, identity, command, state, reason, producer
    ):
        """Validate and install one planner contract after bridge alignment."""
        if state in (
            PlannerCommandContract.STATE_ZERO,
            PlannerCommandContract.STATE_INVALIDATED,
        ):
            current_lifecycle = self._lifecycle_transaction_id_locked()
            if (
                identity["lifecycle_transaction_id"] <= 0
                or (
                    current_lifecycle <= 0
                    or (
                        current_lifecycle > 0
                    and identity["lifecycle_transaction_id"] != current_lifecycle
                    )
                )
            ):
                self.publish_status_locked(
                    "planner_contract_rejected",
                    reason="stale_boundary_lifecycle",
                    planner_contract_identity=identity,
                )
                return False
            self.planner_contract_pending = None
            self._store_planner_contract_locked(
                identity, command, state, reason, producer
            )
            self.publish_status_locked(
                "planner_contract_boundary",
                planner_contract_identity=identity,
                reason=str(reason or "planner_boundary"),
            )
            return True

        missing_identity = [
            field
            for field in (
                "transaction_id",
                "route_id",
                "lifecycle_transaction_id",
                "action_generation",
                "map_epoch",
            )
            if int(identity.get(field, 0) or 0) <= 0
        ]
        if missing_identity:
            self._reject_planner_contract_locked(
                "missing_identity_%s" % missing_identity[0],
                identity,
            )
            return False
        mismatch = self._planner_contract_mismatch_locked(identity)
        if mismatch:
            self._reject_planner_contract_locked(mismatch, identity)
            return False
        previous = getattr(self, "planner_contract_identity", None)
        if (
            isinstance(previous, dict)
            and identity["planner_sequence"] <= int(
                previous.get("planner_sequence", 0) or 0
            )
        ):
            self._reject_planner_contract_locked(
                "stale_planner_sequence", identity
            )
            return False
        self.planner_contract_pending = None
        self._store_planner_contract_locked(
            identity, command, state, reason, producer
        )
        self.publish_status_locked(
            "planner_command_authorized",
            planner_contract_identity=identity,
            linear_x=round(float(command.linear.x), 6),
            angular_z=round(float(command.angular.z), 6),
        )
        return True

    def _reject_planner_contract_locked(self, reason, identity):
        """Invalidate the supervisor output until a fresh identity arrives."""
        self.planner_contract_pending = None
        self.planner_contract_valid = False
        self.planner_contract_state = PlannerCommandContract.STATE_INVALIDATED
        self.planner_contract_reason = str(reason or "identity_mismatch")
        self.planner_contract_identity = dict(identity or {})
        self.planner_contract_sequence = int(
            identity.get("planner_sequence", 0) or 0
        )
        self._invalidate_trajectory_feedback_locked(
            "identity_mismatch", clear_planner=True
        )
        self.publish_status_locked(
            "planner_contract_rejected",
            reason=str(reason or "identity_mismatch"),
            identity_mismatch=True,
            planner_contract_identity=dict(identity or {}),
        )

    def on_planner_command_contract(self, message):
        """Consume the identity-bearing planner stream in persistent mode."""
        if not getattr(self, "require_planner_command_contract", False):
            return
        identity = self._planner_contract_identity(message)
        command = self._planner_contract_command(message)
        reason = str(getattr(message, "reason", "") or "")
        with self.lock:
            self._apply_planner_contract_locked(
                identity,
                command,
                identity["state"],
                reason,
                identity["producer"],
            )

    def _promote_pending_planner_contract_locked(self):
        pending = getattr(self, "planner_contract_pending", None)
        if not isinstance(pending, dict):
            return False
        accepted = self._apply_planner_contract_locked(
            pending["identity"],
            pending["command"],
            pending["state"],
            pending["reason"],
            pending["producer"],
        )
        return bool(accepted)

    def _invalidate_trajectory_feedback_locked(
        self, reason, clear_planner=False
    ):
        """Drop identity-free TEB feedback at a control ownership boundary."""
        reason = str(reason or "control_pipeline_boundary")
        previous_command = self.latest_trajectory_command
        previous_selected = [
            round(float(previous_command.linear.x), 4),
            round(float(previous_command.angular.z), 4),
        ]
        previous_valid = bool(getattr(self, "trajectory_feedback_valid", False))
        previous_reason = str(
            getattr(self, "trajectory_feedback_invalid_reason", "") or ""
        )
        previous_required = bool(
            getattr(
                self,
                "trajectory_feedback_requires_fresh_planner_command",
                False,
            )
        )
        changed = bool(
            previous_valid
            or abs(previous_command.linear.x) > 0.001
            or abs(previous_command.angular.z) > 0.01
            or previous_reason != reason
            or not previous_required
            or clear_planner
        )
        self.latest_trajectory_command = Twist()
        self.latest_trajectory_command_wall = 0.0
        self.trajectory_feedback_valid = False
        self.trajectory_feedback_invalid_reason = reason
        self.trajectory_feedback_requires_fresh_planner_command = True
        self.trajectory_feedback_identity = None
        self.trajectory_zero_started_wall = 0.0
        self.trajectory_continuity_goal_distance = None
        self.trajectory_feedback_invalidation_count = int(
            getattr(self, "trajectory_feedback_invalidation_count", 0) or 0
        ) + 1
        if clear_planner:
            self.latest_planner_command = Twist()
            self.latest_planner_command_wall = 0.0
            self.planner_command_transaction_id = 0
            self.planner_command_route_id = 0
            self.planner_command_map_epoch = None
            self.planner_command_graph_transaction_id = 0
            self.planner_command_action_identity = None
        if changed:
            self.publish_status_locked(
                "trajectory_feedback_invalidated",
                reason=reason,
                clear_planner=bool(clear_planner),
                previous_selected_command=previous_selected,
                previous_valid=previous_valid,
                invalidation_count=int(
                    self.trajectory_feedback_invalidation_count
                ),
            )

    def on_intent(self, message):
        source = "unknown"
        priority = 0
        route_kind = ""
        intent_goal = None
        target_track_id = ""
        route_id = 0
        map_epoch = None
        graph_transaction_id = 0
        try:
            payload = json.loads(message.data)
            if isinstance(payload, dict):
                source = str(payload.get("source", source)).strip().lower() or source
                priority = int(payload.get("priority", priority))
                route_kind = str(payload.get("route_kind", "")).strip().lower()
                route_id = max(0, int(payload.get("route_id", 0) or 0))
                raw_map_epoch = payload.get("map_epoch")
                map_epoch = (
                    None
                    if raw_map_epoch is None
                    else max(0, int(raw_map_epoch or 0))
                )
                graph_transaction_id = max(
                    0, int(payload.get("graph_transaction_id", 0) or 0)
                )
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
            self.latest_route_id = route_id
            self.latest_map_epoch = map_epoch
            self.latest_graph_transaction_id = graph_transaction_id
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
            portal_probe_intent = (
                route_kind == PORTAL_PROBE_KIND
                and source == FRONTIER_SOURCE
            )
            if (
                route_kind != TURN_ROUTE_KIND
                and not endpoint_intent
                and not portal_intent
                and not local_egress_intent
                and not portal_probe_intent
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
            try:
                self.active_action_goal_transaction_id = int(
                    payload.get("active_goal_transaction_id", 0) or 0
                )
            except (TypeError, ValueError):
                self.active_action_goal_transaction_id = 0
            try:
                self.active_action_route_id = max(
                    0,
                    int(
                        payload.get(
                            "active_route_id", payload.get("route_id", 0)
                        )
                        or 0
                    ),
                )
            except (TypeError, ValueError):
                self.active_action_route_id = 0
            raw_active_epoch = payload.get(
                "active_route_map_epoch",
                payload.get("map_epoch", payload.get("latest_route_map_epoch")),
            )
            try:
                self.active_action_map_epoch = (
                    None
                    if raw_active_epoch is None
                    else max(0, int(raw_active_epoch or 0))
                )
            except (TypeError, ValueError):
                self.active_action_map_epoch = None
            try:
                self.active_action_graph_transaction_id = max(
                    0,
                    int(
                        payload.get(
                            "active_graph_transaction_id",
                            payload.get("graph_transaction_id", 0),
                        )
                        or 0
                    ),
                )
            except (TypeError, ValueError):
                self.active_action_graph_transaction_id = 0
            self.active_action_target_track_id = str(
                payload.get("active_target_track_id", "")
            ).strip()
            try:
                self.active_action_generation = int(
                    payload.get("active_action_generation", 0) or 0
                )
            except (TypeError, ValueError):
                self.active_action_generation = 0
            try:
                self.active_action_transaction_id = int(
                    payload.get("lifecycle_transaction_id", 0) or 0
                )
            except (TypeError, ValueError):
                self.active_action_transaction_id = 0
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
                self._invalidate_trajectory_feedback_locked(
                    "route_owner_changed", clear_planner=True
                )
            elif not self.active_action:
                self._invalidate_trajectory_feedback_locked(
                    "bridge_inactive", clear_planner=True
                )
            elif payload.get("teb_feedback_valid") is False:
                self._invalidate_trajectory_feedback_locked(
                    str(
                        payload.get(
                            "teb_feedback_invalid_reason",
                            "bridge_feedback_invalidated",
                        )
                        or "bridge_feedback_invalidated"
                    ),
                    clear_planner=bool(payload.get("clear_planner", False)),
                )
            self._promote_pending_planner_contract_locked()
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
            self.scan_monotonic = now_for(self)

    def on_planner_command(self, message):
        if getattr(self, "require_planner_command_contract", False):
            with self.lock:
                self.publish_status_locked(
                    "planner_raw_command_ignored",
                    reason="typed_planner_contract_required",
                    raw_command=[
                        round(float(message.linear.x), 4),
                        round(float(message.angular.z), 4),
                    ],
                )
            return
        with self.lock:
            self.latest_planner_command = copy.deepcopy(message)
            self.latest_planner_command_wall = now_for(self)
            self.planner_command_sequence += 1
            self.planner_command_transaction_id = int(
                getattr(
                    getattr(self, "lifecycle_manager", None),
                    "current_transaction_id",
                    0,
                )
                or 0
            )
            self.planner_command_route_id = int(
                getattr(self, "active_action_route_id", 0) or 0
            )
            self.planner_command_map_epoch = getattr(
                self, "active_action_map_epoch", None
            )
            self.planner_command_graph_transaction_id = int(
                getattr(self, "active_action_graph_transaction_id", 0) or 0
            )
            self.planner_command_action_generation = int(
                getattr(self, "active_action_generation", 0) or 0
            )
            self.planner_contract_sequence = int(
                getattr(self, "planner_command_sequence", 0) or 0
            )
            self.planner_command_action_identity = self.active_action_identity
            if (
                abs(float(message.linear.x)) <= 0.001
                and abs(float(message.angular.z)) <= 0.01
            ):
                self._invalidate_trajectory_feedback_locked(
                    "planner_zero_command"
                )

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
            now = now_for(self)
            if not self.active_action:
                self._invalidate_trajectory_feedback_locked("no_active_action")
                return
            if getattr(self, "require_planner_command_contract", False):
                if not bool(getattr(self, "planner_contract_valid", False)):
                    self.publish_status_locked(
                        "trajectory_feedback_rejected",
                        reason="planner_contract_missing",
                    )
                    return
                if self.planner_contract_awaiting_feedback:
                    contract_command = self.planner_contract_command
                    linear_delta = abs(
                        command.linear.x - contract_command.linear.x
                    )
                    angular_delta = abs(
                        command.angular.z - contract_command.angular.z
                    )
                    if linear_delta > 0.20 or angular_delta > 0.35:
                        self.publish_status_locked(
                            "trajectory_feedback_rejected",
                            reason="planner_contract_command_mismatch",
                            linear_delta=round(linear_delta, 4),
                            angular_delta=round(angular_delta, 4),
                        )
                        return
                    self.planner_contract_awaiting_feedback = False
                    self.trajectory_feedback_requires_fresh_planner_command = False
            if self.trajectory_feedback_requires_fresh_planner_command:
                # This may be a delayed sample from the previous route. It has
                # no wire-level identity and cannot re-arm the control clock.
                return
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
            self.trajectory_feedback_valid = True
            self.trajectory_feedback_invalid_reason = ""
            self.trajectory_feedback_requires_fresh_planner_command = False
            self.trajectory_feedback_identity = self.active_action_identity
