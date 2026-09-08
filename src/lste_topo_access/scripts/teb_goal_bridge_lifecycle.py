#!/usr/bin/env python3
"""Single-threaded ingress boundary for the TEB action bridge."""

from copy import deepcopy
import json

import rospy

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
        return self._enqueue_bridge_event(EventType.BRIDGE_GOAL, message)

    def on_goal_command(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_GOAL_COMMAND, message)

    def on_intent(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_INTENT, message)

    def on_frontier_status(self, message):
        return self._enqueue_bridge_event(EventType.BRIDGE_FRONTIER_STATUS, message)

    def on_turn_supervisor_status(self, message):
        return self._enqueue_bridge_event(
            EventType.BRIDGE_TURN_STATUS, message
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
        return self._enqueue_bridge_event(
            EventType.BRIDGE_TARGET_RESULT, message
        )

    def on_persistent_frontier_endpoint_reached(self, message):
        return self._enqueue_bridge_event(
            EventType.BRIDGE_FRONTIER_ENDPOINT, message
        )

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
        handler(self, event.payload)
        if event.type in (
            EventType.BRIDGE_GOAL,
            EventType.BRIDGE_GOAL_COMMAND,
            EventType.BRIDGE_INTENT,
        ):
            return State.DISPATCHED
        if event.type == EventType.BRIDGE_TASK_DONE and bool(
            getattr(event.payload, "data", False)
        ):
            return State.COMPLETED
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
                event=(None if event is None else event.type.value),
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
