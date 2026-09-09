#!/usr/bin/env python3
"""Lifecycle ingress and tick ownership for the online frontier node."""

from copy import deepcopy
import json

import rospy

from clock_provider import ClockProvider, now_for
from lifecycle_manager import EventType, LifecycleManager, State
from global_frontier_event_callbacks import GlobalFrontierEventCallbacksMixin
from global_frontier_planning_runtime import GlobalFrontierPlanningRuntimeMixin
from global_frontier_planning_navfn import GlobalFrontierPlanningNavfnMixin
from global_frontier_route_state import GlobalFrontierRouteStateMixin
from global_frontier_terminal_replan import GlobalFrontierTerminalReplanMixin


class GlobalFrontierLifecycleMixin:
    """Keep ROS ingress passive and route lifecycle state single-threaded."""

    def _initialize_lifecycle_manager(self):
        gp = rospy.get_param
        self.lifecycle_tick_period = max(
            0.05, float(gp("~lifecycle_tick_period", 0.2))
        )
        self.time_provider = getattr(self, "time_provider", None) or ClockProvider()
        self.lifecycle_manager = LifecycleManager(
            event_handler=self._handle_lifecycle_event,
            transition_handler=self._on_lifecycle_transition,
            timeout_handler=self._on_lifecycle_timeout,
            timeouts={
                State.DISPATCHED: max(
                    1.0, float(gp("~lifecycle_dispatch_timeout", 120.0))
                ),
                State.SYNCING: max(
                    0.1, float(gp("~lifecycle_sync_timeout", 10.0))
                ),
            },
            time_provider=self.time_provider,
        )
        self._lifecycle_sync_seen = set()
        self._lifecycle_sync_started_transaction = 0

    def lifecycle_state(self):
        return self.lifecycle_manager.current_state

    def lifecycle_transaction_id(self):
        return self.lifecycle_manager.current_transaction_id

    def lifecycle_active(self):
        return self.lifecycle_manager.is_in(
            State.DISPATCHED,
            State.EXECUTION_DONE,
            State.SYNCING,
        )

    def _enqueue_lifecycle_event(self, event_type, payload=None, transaction_id=None):
        """The only operation exposed to asynchronous ROS ingress."""
        if transaction_id is None:
            transaction_id = self._transaction_id_from_message(payload, event_type)
        return self.lifecycle_manager.enqueue_type(
            event_type,
            payload,
            transaction_id=transaction_id,
        )

    @staticmethod
    def _transaction_id_from_message(message, event_type=None):
        """Read a distributed lifecycle identity without applying the event."""
        # Bridge status is an observation stream, not an owner of this
        # lifecycle. Importing the bridge's local transaction here can adopt a
        # newer ID while a frontier route is being selected and lets a stale
        # status event cross the route boundary.
        if event_type == EventType.NAV_FAILED:
            return None
        try:
            value = int(getattr(message, "lifecycle_transaction_id", 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            if (
                event_type == EventType.EXECUTION_TERMINAL
                and int(getattr(message, "route_id", 0) or 0) <= 0
            ):
                return None
            return value
        try:
            payload = json.loads(message.data)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if event_type == EventType.NAV_FAILED:
            source = str(
                payload.get(
                    "active_intent_source",
                    payload.get("intent_source", ""),
                )
                or ""
            ).strip().lower()
            if source and source != "global_slam_frontier":
                return None
        if event_type == EventType.TURN_STATUS:
            source = str(
                payload.get(
                    "intent_source",
                    payload.get("active_action_source", ""),
                )
                or ""
            ).strip().lower()
            route_kind = str(
                payload.get(
                    "route_kind",
                    payload.get("active_action_route_kind", ""),
                )
                or ""
            ).strip().lower()
            if source and source != "global_slam_frontier" and route_kind not in {
                "frontier_endpoint",
                "portal_transition",
                "portal_probe",
                "local_egress",
            }:
                return None
        try:
            value = int(payload.get("lifecycle_transaction_id", 0) or 0)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    # ROS callbacks: gather an immutable message reference and return.
    def on_map(self, message):
        return self._enqueue_lifecycle_event(
            EventType.MAP_UPDATED, deepcopy(message)
        )

    def on_costmap(self, message):
        return self._enqueue_lifecycle_event(
            EventType.COSTMAP_UPDATED, ("full", deepcopy(message))
        )

    def on_costmap_update(self, message):
        return self._enqueue_lifecycle_event(
            EventType.COSTMAP_UPDATED, ("delta", deepcopy(message))
        )

    def on_pose(self, message):
        return self._enqueue_lifecycle_event(
            EventType.POSE_UPDATED, deepcopy(message)
        )

    def on_scan(self, message):
        return self._enqueue_lifecycle_event(
            EventType.SCAN_UPDATED, deepcopy(message)
        )

    def on_task(self, message):
        return self._enqueue_lifecycle_event(
            EventType.TASK_UPDATED, deepcopy(message)
        )

    def on_detections(self, message):
        return self._enqueue_lifecycle_event(
            EventType.DETECTIONS_UPDATED, deepcopy(message)
        )

    def on_goal_arbitration(self, message):
        return self._enqueue_lifecycle_event(
            EventType.GOAL_UPDATED, deepcopy(message)
        )

    def on_task_done(self, message):
        return self._enqueue_lifecycle_event(
            EventType.TASK_COMPLETED, deepcopy(message)
        )

    def on_move_base_recovery(self, message):
        return self._enqueue_lifecycle_event(
            EventType.RECOVERY_OBSERVED, deepcopy(message)
        )

    def on_bridge_status(self, message):
        return self._enqueue_lifecycle_event(
            EventType.NAV_FAILED, deepcopy(message)
        )

    def on_replan_request(self, message):
        return self._enqueue_lifecycle_event(
            EventType.REPLAN_REQUESTED, deepcopy(message)
        )

    def on_turn_status(self, message):
        return self._enqueue_lifecycle_event(
            EventType.TURN_STATUS, deepcopy(message)
        )

    def on_execution_terminal(self, message):
        return self._enqueue_lifecycle_event(
            EventType.EXECUTION_TERMINAL, deepcopy(message)
        )

    def on_immediate_plan(self, event):
        del event
        return self._enqueue_lifecycle_event(
            EventType.REPLAN_REQUESTED, {"event": "immediate_plan"}
        )

    def on_terminal_drain_timer(self, event):
        del event
        return self._enqueue_lifecycle_event(
            EventType.EXECUTION_TERMINAL, {"event": "terminal_drain"}
        )

    def on_timer(self, event):
        """Run gather, compute, and scatter from one fixed-rate timer."""
        return self.lifecycle_manager.tick(
            compute_handler=lambda _now: GlobalFrontierPlanningRuntimeMixin.on_timer(
                self, event
            )
        )

    def _handle_lifecycle_event(self, event):
        """Apply one gathered fact from inside ``LifecycleManager.tick``."""
        handlers = {
            EventType.MAP_UPDATED: self._apply_map,
            EventType.POSE_UPDATED: self._apply_pose,
            EventType.SCAN_UPDATED: self._apply_scan,
            EventType.TASK_UPDATED: self._apply_task,
            EventType.DETECTIONS_UPDATED: self._apply_detections,
            EventType.GOAL_UPDATED: self._apply_goal_arbitration,
            EventType.TASK_COMPLETED: self._apply_task_done,
            EventType.RECOVERY_OBSERVED: self._apply_move_base_recovery,
            EventType.EXECUTION_TERMINAL: self._apply_execution_terminal_event,
            EventType.NAV_FAILED: self._apply_bridge_status,
            EventType.REPLAN_REQUESTED: self._apply_replan_request_event,
            EventType.TURN_STATUS: self._apply_turn_status,
            EventType.NAVFN_RESULT: self._apply_navfn_result,
        }
        if event.type == EventType.COSTMAP_UPDATED:
            kind, payload = event.payload
            result = (
                self._apply_costmap(payload)
                if kind == "full"
                else (
                    self._apply_costmap_update(payload)
                    if kind == "delta"
                    else GlobalFrontierRouteStateMixin.invalidate_costmap(self)
                )
            )
            if (
                kind != "resync"
                and
                self.lifecycle_state() == State.SYNCING
                and event.created_at >= self.lifecycle_manager.state_entry_time
            ):
                self._lifecycle_sync_seen.add("costmap")
                if self._lifecycle_sync_seen == {"map", "costmap"}:
                    self._lifecycle_sync_seen.clear()
                    return State.IDLE
            return result
        handler = handlers.get(event.type)
        if handler is None:
            return None
        if event.type == EventType.MAP_UPDATED:
            if (
                self.lifecycle_state() == State.SYNCING
                and event.created_at >= self.lifecycle_manager.state_entry_time
            ):
                self._lifecycle_sync_seen.add("map")
        result = handler(event.payload)
        if result is not None:
            return result
        if (
            self.lifecycle_state() == State.SYNCING
            and self._lifecycle_sync_seen == {"map", "costmap"}
        ):
            self._lifecycle_sync_seen.clear()
            return State.IDLE
        if event.type == EventType.TASK_COMPLETED:
            return State.COMPLETED if bool(getattr(event.payload, "data", False)) else State.IDLE
        return None

    def _on_lifecycle_transition(self, previous, current, event):
        publish = getattr(self, "publish_status", None)
        if callable(publish):
            publish(
                "lifecycle_transition",
                previous_state=previous.value,
                current_state=current.value,
                transaction_id=int(self.lifecycle_transaction_id()),
                transition_event=(
                    None if event is None else event.type.value
                ),
            )
        if current == State.FAILED:
            self._settle_failed_route_obligations(
                now_for(self), reason="lifecycle_failed"
            )
            active_frontier = getattr(self, "active_frontier", None)
            release = getattr(self, "release_active_frontier", None)
            if active_frontier is not None and callable(release):
                release(discard_prefetch=True)
            else:
                clear_prefetch = getattr(self, "clear_prefetched_frontier", None)
                if callable(clear_prefetch):
                    clear_prefetch()
            self.last_planning_wall = 0.0
            if callable(publish):
                publish(
                    "lifecycle_failure_cleanup",
                    transaction_id=int(self.lifecycle_transaction_id()),
                    released_route=active_frontier is not None,
                )
        elif current == State.EXECUTION_DONE:
            self._lifecycle_sync_seen.clear()
            self._lifecycle_sync_started_transaction = (
                self.lifecycle_transaction_id()
            )
            self.lifecycle_manager.enqueue_type(EventType.SYNC_REQUESTED)
        elif current == State.SYNCING:
            self._enqueue_sync_snapshot()
            if callable(publish):
                publish(
                    "lifecycle_sync_requested",
                    transaction_id=int(self.lifecycle_transaction_id()),
                    timeout=float(
                        self.lifecycle_manager._timeouts.get(State.SYNCING, 0.0)
                    ),
                )

    def _settle_failed_route_obligations(self, now, reason):
        """Close semantic attempts before releasing their physical route.

        Lifecycle failure cleanup is the common boundary for controller
        failures. Clearing the active route first used to orphan a durable
        PortalProbe/WorkItem attempt, so every later planning pass saw an
        active owner and returned ``hold`` forever.
        """
        settle_probe = getattr(self, "settle_active_portal_probe", None)
        if (
            callable(settle_probe)
            and getattr(self, "active_portal_probe_id", None) is not None
        ):
            settle_probe("failed", now, reason)

        settle_work_item = getattr(self, "settle_active_work_item", None)
        if (
            callable(settle_work_item)
            and getattr(self, "active_work_item_id", None) is not None
        ):
            settle_work_item(now, "deferred", reason)

    def _enqueue_sync_snapshot(self):
        """Feed valid cached map state back through the SYNCING FSM.

        Map and full costmap topics may be quiet while the robot is stopped.
        Reusing a still-valid cache preserves the event contract without
        requiring a new geometric publication merely to prove synchronization.
        """
        publish = getattr(self, "publish_status", None)
        transaction_id = int(self.lifecycle_transaction_id())
        state_entry_time = self.lifecycle_manager.state_entry_time
        map_message = getattr(self, "map_msg", None)
        costmap_message = getattr(self, "costmap_msg", None)
        max_age = max(0.0, float(getattr(self, "costmap_max_age", 0.0)))
        last_receive = float(getattr(self, "costmap_last_receive_wall", 0.0))
        costmap_age = (
            float("inf")
            if last_receive <= 0.0
            else now_for(self) - last_receive
        )
        fresh_costmap = getattr(self, "fresh_costmap", None)
        if callable(fresh_costmap):
            costmap_message = fresh_costmap()
            costmap_valid = costmap_message is not None
        else:
            costmap_valid = (
                costmap_message is not None
                and max_age > 0.0
                and costmap_age <= max_age
            )
        if callable(publish):
            publish(
                "lifecycle_sync_snapshot_requested",
                transaction_id=transaction_id,
                map_cached=map_message is not None,
                costmap_cached=costmap_message is not None,
                costmap_age_seconds=(
                    None if costmap_age == float("inf") else round(costmap_age, 3)
                ),
            )
        if map_message is None or not costmap_valid:
            if callable(publish):
                publish(
                    "lifecycle_sync_waiting_for_publish",
                    transaction_id=transaction_id,
                    missing=(
                        ["map"] if map_message is None else []
                    ) + (["costmap"] if not costmap_valid else []),
                )
            return False
        map_event = self.lifecycle_manager.make_event(
            EventType.MAP_UPDATED,
            deepcopy(map_message),
            transaction_id=transaction_id,
            created_at=state_entry_time,
        )
        costmap_event = self.lifecycle_manager.make_event(
            EventType.COSTMAP_UPDATED,
            ("full", deepcopy(costmap_message)),
            transaction_id=transaction_id,
            created_at=state_entry_time,
        )
        map_enqueued = self.lifecycle_manager.enqueue(map_event)
        costmap_enqueued = self.lifecycle_manager.enqueue(costmap_event)
        if callable(publish):
            publish(
                "lifecycle_sync_snapshot_reused",
                transaction_id=transaction_id,
                map_enqueued=bool(map_enqueued),
                costmap_enqueued=bool(costmap_enqueued),
            )
        return bool(map_enqueued and costmap_enqueued)

    def _on_lifecycle_timeout(self, event):
        if hasattr(self, "publish_status"):
            self.publish_status(
                "lifecycle_timeout",
                state=event.payload.get("state"),
                timeout=event.payload.get("timeout"),
                transaction_id=int(event.transaction_id),
            )

    def _apply_map(self, message):
        return GlobalFrontierRouteStateMixin.apply_map(self, message)

    def _apply_costmap(self, message):
        return GlobalFrontierRouteStateMixin.apply_costmap(self, message)

    def _apply_costmap_update(self, message):
        return GlobalFrontierRouteStateMixin.apply_costmap_update(self, message)

    def _apply_pose(self, message):
        return GlobalFrontierRouteStateMixin.apply_pose(self, message)

    def _apply_scan(self, message):
        return GlobalFrontierEventCallbacksMixin.apply_scan(self, message)

    def _apply_task(self, message):
        previous = (
            getattr(self, "current_task_id", ""),
            getattr(self, "current_task_version", ""),
        )
        result = GlobalFrontierEventCallbacksMixin.apply_task(self, message)
        current = (
            getattr(self, "current_task_id", ""),
            getattr(self, "current_task_version", ""),
        )
        if current != previous:
            self.lifecycle_manager.begin_transaction(State.IDLE)
        return result

    def _apply_detections(self, message):
        return GlobalFrontierEventCallbacksMixin.apply_detections(self, message)

    def _apply_goal_arbitration(self, message):
        return GlobalFrontierEventCallbacksMixin.apply_goal_arbitration(
            self, message
        )

    def _apply_task_done(self, message):
        return GlobalFrontierEventCallbacksMixin.apply_task_done(self, message)

    def _apply_move_base_recovery(self, message):
        return GlobalFrontierEventCallbacksMixin.apply_move_base_recovery(
            self, message
        )

    def _apply_bridge_status(self, message):
        accepted = GlobalFrontierEventCallbacksMixin.apply_bridge_status(
            self, message
        )
        # NAV_FAILED is also the ingress type for non-terminal bridge status
        # events. Returning the current state suppresses LifecycleManager's
        # default NAV_FAILED -> FAILED transition for those observations.
        return State.FAILED if accepted else self.lifecycle_state()

    def _apply_turn_status(self, message):
        return GlobalFrontierEventCallbacksMixin.apply_turn_status(self, message)

    def _apply_navfn_result(self, message):
        if not isinstance(message, dict):
            return None
        GlobalFrontierPlanningNavfnMixin._store_navfn_validation_result(
            self,
            message.get("key"),
            message.get("payload", {}),
            message.get("result", {}),
            message.get("completed_wall"),
        )
        return None

    def _apply_execution_terminal_event(self, message):
        if isinstance(message, dict):
            return None
        before_route = getattr(self, "active_route_id", 0)
        GlobalFrontierTerminalReplanMixin._process_execution_terminal_locked(
            self, message
        )
        if (
            before_route > 0
            and getattr(self, "active_terminal_received", False)
        ):
            return State.EXECUTION_DONE
        return None

    def _apply_replan_request_event(self, message):
        if isinstance(message, dict):
            if message.get("event") in ("immediate_plan", "terminal_replan"):
                self.last_planning_wall = 0.0
            return None
        self.lifecycle_manager.begin_transaction(State.IDLE)
        return GlobalFrontierEventCallbacksMixin.apply_replan_request(
            self, message
        )
