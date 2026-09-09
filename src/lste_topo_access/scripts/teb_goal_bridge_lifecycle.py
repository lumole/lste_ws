#!/usr/bin/env python3
"""Single-threaded ingress boundary for the TEB action bridge."""

from copy import deepcopy
import json

import rospy

from experiment_reset_contract import decode_reset_request, publish_reset_ack
from lifecycle_manager import EventType, LifecycleManager, State
from teb_goal_bridge_action_client import TebGoalBridgeActionClientMixin
from teb_goal_bridge_action_feedback import TebGoalBridgeActionFeedbackMixin
from teb_goal_bridge_action_terminal import TebGoalBridgeActionTerminalMixin
from teb_goal_bridge_frontier_status import TebGoalBridgeFrontierStatusMixin
from teb_goal_bridge_intent import TebGoalBridgeIntentMixin
from teb_goal_bridge_mission_control import TebGoalBridgeMissionControlMixin
from teb_goal_bridge_mission_input import TebGoalBridgeMissionInputMixin
from teb_goal_bridge_mission_runtime import TebGoalBridgeMissionRuntimeMixin
from teb_goal_bridge_persistent_frontier_endpoint import (
    TebGoalBridgePersistentFrontierEndpointMixin,
)
from teb_goal_bridge_persistent_target_result import (
    TebGoalBridgePersistentTargetResultMixin,
)
from teb_goal_bridge_route_monitoring import TebGoalBridgeRouteMonitoringMixin
from teb_goal_bridge_teb_runtime import TebGoalBridgeTebRuntimeMixin
from teb_goal_bridge_state import initialize_bridge_state


class TebGoalBridgeLifecycleMixin:
    """Gather ROS/actionlib facts and compute bridge state from one timer."""

    def _initialize_lifecycle_manager(self):
        gp = rospy.get_param
        self.lifecycle_tick_period = max(
            0.05, float(gp("~lifecycle_tick_period", 0.2))
        )
        self.lifecycle_manager = LifecycleManager(
            event_handler=self._handle_lifecycle_event,
            transition_handler=self._on_lifecycle_transition,
            timeout_handler=self._on_lifecycle_timeout,
            timeouts={
                State.DISPATCHED: max(
                    1.0, float(gp("~lifecycle_dispatch_timeout", 120.0))
                ),
            },
        )
        self._last_hard_reset_id = None
        self._last_hard_reset_transaction_id = 0
        self._hard_reset_count = 0

    def _transaction_id_from_message(self, message):
        try:
            payload = json.loads(message.data)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        raw = payload.get(
            "lifecycle_transaction_id",
            payload.get("transaction_id"),
        )
        try:
            value = int(raw or 0)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _enqueue_bridge_event(self, event_type, payload=None, transaction_id=None):
        if transaction_id is None:
            transaction_id = self._transaction_id_from_message(payload)
        return self.lifecycle_manager.enqueue_type(
            event_type,
            deepcopy(payload),
            transaction_id=transaction_id,
        )

    # Gather-only topic callbacks.
    def on_goal(self, message):
        if self.use_goal_command:
            return False
        return self._enqueue_bridge_event(EventType.BRIDGE_GOAL, message)

    def on_goal_command(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_GOAL_COMMAND, message)

    def on_intent(self, message):
        if self.use_goal_command:
            return False
        return self._enqueue_bridge_event(EventType.BRIDGE_INTENT, message)

    def on_frontier_status(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_FRONTIER_STATUS, message)

    def on_turn_supervisor_status(self, message):
        # Turn-supervisor status is telemetry from a downstream adapter, not a
        # new mission owner. Do not let its independently generated lifecycle
        # UUID advance this bridge past a GoalManager transaction that is still
        # waiting at the warm-slice boundary.
        current_transaction_id = int(
            getattr(self.lifecycle_manager, "current_transaction_id", 0) or 0
        )
        return self._enqueue_bridge_event(
            EventType.BRIDGE_TURN_STATUS,
            message,
            transaction_id=current_transaction_id,
        )

    def on_teb_feedback(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_TEB_FEEDBACK, message)

    def on_teb_planner_command(self, message):
        return self._enqueue_bridge_event(
            EventType.BRIDGE_PLANNER_COMMAND, message
        )

    def on_navfn_plan(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_NAVFN_PLAN, message)

    def on_local_costmap(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_COSTMAP, message)

    def on_pose(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_POSE, message)

    def on_mode(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_MODE, message)

    def on_task_done(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_TASK_DONE, message)

    def on_persistent_target_plan_result(self, message):
        # ``transaction_id`` in the C++ result is the semantic target command
        # identity, not this bridge's distributed lifecycle transaction. Keep
        # the local lifecycle identity on the ingress event so a valid target
        # installation cannot be discarded as stale after a newer mission
        # command adopted a UUID-based lifecycle transaction.
        return self._enqueue_bridge_event(
            EventType.BRIDGE_TARGET_RESULT,
            message,
            transaction_id=int(
                getattr(self.lifecycle_manager, "current_transaction_id", 0)
                or 0
            ),
        )

    def on_persistent_frontier_endpoint_reached(self, message):
        return self._enqueue_bridge_event(
            EventType.BRIDGE_FRONTIER_ENDPOINT, message
        )

    def on_hard_reset(self, message):
        """Queue a reset so cancellation and state clearing share one tick."""
        return self._enqueue_bridge_event(EventType.RESET, deepcopy(message))

    # Actionlib callbacks are also gather-only.
    def on_active(self, generation):
        return self._enqueue_bridge_event(
            EventType.ACTION_FEEDBACK, ("active", generation)
        )

    def on_feedback(self, generation, feedback):
        return self._enqueue_bridge_event(
            EventType.ACTION_FEEDBACK, ("feedback", generation, deepcopy(feedback))
        )

    def on_done(self, generation, status, result):
        return self._enqueue_bridge_event(
            EventType.ACTION_DONE,
            (generation, status, deepcopy(result)),
        )

    def on_terminal_timer(self, event):
        del event
        return self._enqueue_bridge_event(EventType.BRIDGE_WAKE, None)

    def on_timer(self, event):
        return self.lifecycle_manager.tick(
            compute_handler=lambda _now: TebGoalBridgeMissionRuntimeMixin.on_timer(
                self, event
            )
        )

    def _handle_lifecycle_event(self, event):
        if event.type == EventType.RESET:
            return self._apply_hard_reset(event)
        if (
            self.lifecycle_manager.current_transaction_id == 0
            and event.type
            in (
                EventType.BRIDGE_GOAL,
                EventType.BRIDGE_GOAL_COMMAND,
                EventType.BRIDGE_INTENT,
            )
        ):
            self.lifecycle_manager.begin_transaction(State.DISPATCHED)
        if event.type in (
            EventType.BRIDGE_GOAL_COMMAND,
            EventType.BRIDGE_INTENT,
            EventType.BRIDGE_FRONTIER_STATUS,
        ):
            external_id = self._transaction_id_from_message(event.payload)
            if external_id is not None:
                if external_id < self.lifecycle_manager.current_transaction_id:
                    return None
                self.lifecycle_manager.adopt_transaction(
                    external_id, State.IDLE
                )
        handlers = {
            EventType.BRIDGE_GOAL: TebGoalBridgeMissionInputMixin.on_goal,
            EventType.BRIDGE_GOAL_COMMAND: TebGoalBridgeMissionInputMixin.on_goal_command,
            EventType.BRIDGE_INTENT: TebGoalBridgeIntentMixin.on_intent,
            EventType.BRIDGE_FRONTIER_STATUS: TebGoalBridgeFrontierStatusMixin.on_frontier_status,
            EventType.BRIDGE_TURN_STATUS: TebGoalBridgeTebRuntimeMixin.on_turn_supervisor_status,
            EventType.BRIDGE_TEB_FEEDBACK: TebGoalBridgeTebRuntimeMixin.on_teb_feedback,
            EventType.BRIDGE_PLANNER_COMMAND: TebGoalBridgeTebRuntimeMixin.on_teb_planner_command,
            EventType.BRIDGE_NAVFN_PLAN: TebGoalBridgeRouteMonitoringMixin.on_navfn_plan,
            EventType.BRIDGE_COSTMAP: TebGoalBridgeRouteMonitoringMixin.on_local_costmap,
            EventType.BRIDGE_POSE: TebGoalBridgeTebRuntimeMixin.on_pose,
            EventType.BRIDGE_MODE: TebGoalBridgeMissionControlMixin.on_mode,
            EventType.BRIDGE_TASK_DONE: TebGoalBridgeMissionControlMixin.on_task_done,
            EventType.BRIDGE_TARGET_RESULT: TebGoalBridgePersistentTargetResultMixin.on_persistent_target_plan_result,
            EventType.BRIDGE_FRONTIER_ENDPOINT: TebGoalBridgePersistentFrontierEndpointMixin.on_persistent_frontier_endpoint_reached,
        }
        if event.type == EventType.ACTION_FEEDBACK:
            payload = event.payload
            if payload[0] == "active":
                return TebGoalBridgeActionClientMixin.on_active(
                    self, payload[1]
                )
            return TebGoalBridgeActionFeedbackMixin.on_feedback(
                self, payload[1], payload[2]
            )
        if event.type == EventType.ACTION_DONE:
            generation, status, result = event.payload
            return TebGoalBridgeActionTerminalMixin.on_done(
                self, generation, status, result
            )
        handler = handlers.get(event.type)
        if handler is None:
            return None
        if self.use_goal_command and event.type in (
            EventType.BRIDGE_GOAL,
            EventType.BRIDGE_INTENT,
        ):
            return None
        previous_goal_transaction = int(
            getattr(self, "latest_goal_transaction_id", 0) or 0
        )
        handler(self, event.payload)
        if event.type == EventType.BRIDGE_GOAL_COMMAND:
            # An ignored/stale command must not manufacture a lifecycle state
            # that claims an executable route exists. Accepted mission and
            # target-terminal commands both advance this semantic transaction.
            if int(getattr(self, "latest_goal_transaction_id", 0) or 0) <= previous_goal_transaction:
                return None
            return State.DISPATCHED
        if event.type in (EventType.BRIDGE_GOAL, EventType.BRIDGE_INTENT):
            return State.DISPATCHED
        if event.type == EventType.BRIDGE_TASK_DONE and bool(
            getattr(event.payload, "data", False)
        ):
            return State.COMPLETED
        return None

    def _apply_hard_reset(self, event):
        """Cancel MoveBase and clear every bridge-side route lease."""
        request = decode_reset_request(event.payload)
        if request is None:
            return None
        reset_id = request["reset_id"]
        if reset_id == self._last_hard_reset_id:
            publish_reset_ack(
                self.hard_reset_ack_pub,
                "lste_teb_goal_bridge",
                request,
                state=State.IDLE.value,
                transaction_id=int(
                    getattr(self, "_last_hard_reset_transaction_id", 0) or 0
                ),
                duplicate=True,
                action_active=False,
                route_owner=False,
                active_route_id=0,
                latest_route_id=0,
                intent_seen=False,
                route_lease_watchdog_active=False,
                persistent_commands_cleared=True,
            )
            return None

        previous_generation = int(getattr(self, "action_generation", 0) or 0)
        previous_route_id = int(getattr(self, "active_route_id", 0) or 0)
        previous_intent = bool(getattr(self, "intent_seen", False))
        previous_action_active = bool(getattr(self, "action_active", False))
        if callable(getattr(self, "cancel_locked", None)):
            self.cancel_locked("experiment_hard_reset")
        clear_persistent = getattr(
            self, "_publish_persistent_hard_reset_locked", None
        )
        persistent_commands_cleared = True
        if callable(clear_persistent):
            persistent_commands_cleared = bool(clear_persistent(request["reason"]))
        new_transaction_id = self.lifecycle_manager.hard_reset(
            event=event,
            now=rospy.Time.now().to_sec(),
            transaction_id=request.get("transaction_id"),
        )
        # Reuse the bridge's complete initialization list so newly added route
        # fields cannot survive a slice boundary by accident. Preserve a
        # monotonic action generation to reject late callbacks from the old
        # MoveBase goal after the reset.
        initialize_bridge_state(self)
        self.action_generation = max(1, previous_generation + 1)
        self._last_hard_reset_id = reset_id
        self._last_hard_reset_transaction_id = int(new_transaction_id)
        self._hard_reset_count += 1
        self.publish_bridge_status(
            "hard_reset_complete",
            reset_id=reset_id,
            reason=request["reason"],
            state=State.IDLE.value,
            previous_route_id=previous_route_id,
            previous_intent_seen=previous_intent,
            previous_action_active=previous_action_active,
            hard_reset_count=int(self._hard_reset_count),
            persistent_commands_cleared=persistent_commands_cleared,
        )
        publish_reset_ack(
            self.hard_reset_ack_pub,
            "lste_teb_goal_bridge",
            request,
            state=State.IDLE.value,
            transaction_id=new_transaction_id,
            action_active=bool(getattr(self, "action_active", False)),
            route_owner=bool(getattr(self, "active_route_id", 0)),
            active_route_id=int(getattr(self, "active_route_id", 0) or 0),
            latest_route_id=int(getattr(self, "latest_route_id", 0) or 0),
            intent_seen=bool(getattr(self, "intent_seen", False)),
            route_lease_watchdog_active=bool(
                getattr(self, "route_lease_watchdog", None)
            ),
            persistent_commands_cleared=persistent_commands_cleared,
        )
        return None

    def _on_lifecycle_transition(self, previous, current, event):
        publish = getattr(self, "publish_bridge_status", None)
        if callable(publish):
            publish(
                "lifecycle_transition",
                previous_state=previous.value,
                current_state=current.value,
                lifecycle_transaction_id=int(
                    self.lifecycle_manager.current_transaction_id
                ),
                transition_event=(
                    None if event is None else event.type.value
                ),
            )
        if current == State.FAILED:
            cancel = getattr(self, "cancel_locked", None)
            if callable(cancel):
                cancel("lifecycle_failed")
            if callable(publish):
                publish(
                    "lifecycle_failure_cleanup",
                    lifecycle_transaction_id=int(
                        self.lifecycle_manager.current_transaction_id
                    ),
                )

    def _on_lifecycle_timeout(self, event):
        publish = getattr(self, "publish_bridge_status", None)
        if callable(publish):
            publish(
                "lifecycle_timeout",
                state=event.payload.get("state"),
                timeout=event.payload.get("timeout"),
                lifecycle_transaction_id=int(event.transaction_id),
            )
