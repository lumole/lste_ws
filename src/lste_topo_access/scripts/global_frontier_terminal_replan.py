"""Successor scheduling after a verified route terminal.

The live node gathers the terminal into ``LifecycleManager`` and consumes it
from the fixed tick.  The lock/timer path remains only for older compositions
and focused compatibility fixtures.
"""

import rospy

from lifecycle_manager import EventType


class GlobalFrontierTerminalReplanMixin:
    """Release completed work and request one fresh planning snapshot."""

    @staticmethod
    def _lease_release_ack_identity(payload):
        """Decode the exact predecessor identity returned by the bridge."""
        released = payload.get("released_controller_route")
        released = released if isinstance(released, dict) else {}

        def integer(*values):
            for value in values:
                try:
                    return int(value or 0)
                except (TypeError, ValueError):
                    continue
            return 0

        return {
            "route_id": integer(
                released.get("route_id"),
                payload.get("released_route_id"),
                payload.get("route_id"),
            ),
            "route_kind": str(
                released.get("route_kind")
                or payload.get("released_route_kind")
                or payload.get("route_kind")
                or ""
            ).strip().lower(),
            "lifecycle_transaction_id": integer(
                released.get("lifecycle_transaction_id"),
                payload.get("lifecycle_transaction_id"),
            ),
            "action_generation": integer(
                released.get("action_generation"),
                payload.get("action_generation"),
            ),
            "graph_transaction_id": integer(
                released.get("graph_transaction_id"),
                payload.get("graph_transaction_id"),
            ),
            "map_epoch": (
                released.get("map_epoch")
                if released.get("map_epoch") is not None
                else payload.get("map_epoch")
            ),
        }

    def _accept_controller_lease_release_ack(self, payload):
        """Open successor planning only after an exact transport ACK."""
        pending = getattr(self, "pending_controller_lease_release", None)
        identity = self._lease_release_ack_identity(payload)
        if not isinstance(pending, dict):
            self.publish_status(
                "stale_lease_release_ack",
                reason="no_pending_controller_release",
                received_identity=identity,
            )
            return False
        mismatch = None
        for field in (
            "route_id",
            "route_kind",
            "lifecycle_transaction_id",
            "action_generation",
            "graph_transaction_id",
        ):
            expected = pending.get(field, 0 if field != "route_kind" else "")
            if identity[field] != expected:
                mismatch = "lease_release_ack_%s_mismatch" % field
                break
        if mismatch is None and pending.get("map_epoch") is not None:
            try:
                received_epoch = int(identity.get("map_epoch"))
                expected_epoch = int(pending["map_epoch"])
            except (TypeError, ValueError):
                mismatch = "lease_release_ack_map_epoch_invalid"
            else:
                if received_epoch != expected_epoch:
                    mismatch = "lease_release_ack_map_epoch_mismatch"
        if mismatch is not None:
            self.publish_status(
                "stale_lease_release_ack",
                reason=mismatch,
                expected_identity=pending,
                received_identity=identity,
            )
            return False
        if str(payload.get("controller_lease", "")).strip().lower() not in {
            "released",
            "transport_released",
        }:
            self.publish_status(
                "stale_lease_release_ack",
                reason="lease_release_ack_not_released",
                expected_identity=pending,
                received_identity=identity,
            )
            return False

        self.pending_controller_lease_release = None
        self.awaiting_controller_lease_release_ack = False
        self.last_controller_lease_release_ack = dict(identity)
        self.last_released_route_controller_pending = False
        self.last_planning_wall = 0.0
        self.publish_status(
            "controller_lease_release_acknowledged",
            **identity,
            new_action_generation=int(
                payload.get("new_action_generation", 0) or 0
            ),
            successor_planning="released",
        )
        lifecycle = getattr(self, "lifecycle_manager", None)
        wake_payload = {
            "event": "terminal_replan_after_lease_ack",
            "route_id": identity["route_id"],
            "lifecycle_transaction_id": identity[
                "lifecycle_transaction_id"
            ],
        }
        if lifecycle is not None:
            lifecycle.enqueue_type(EventType.REPLAN_REQUESTED, wake_payload)
        elif self.immediate_plan_timer is None and not rospy.is_shutdown():
            self.immediate_plan_timer = rospy.Timer(
                rospy.Duration(0.01), self.on_immediate_plan, oneshot=True
            )
        return False

    def schedule_terminal_replan(self, completed, terminal_delta):
        """Release a finished lease and schedule normal successor selection."""
        terminal_contract = getattr(self, "active_terminal_contract", None)
        terminal_contract = (
            dict(terminal_contract)
            if isinstance(terminal_contract, dict)
            else None
        )
        # Do not let the next SLAM timer re-project a completed mission endpoint
        # onto a nearby grid cell and publish a synthetic follow-up action.
        awaiting_ack = self.release_active_frontier(
            preserve_place_departure=bool(self.place_departure.active),
            queue_successor=False,
            terminal_contract=terminal_contract,
        )
        self.active_progress_time = 0.0
        self.active_last_progress_signal = "terminal_replan"
        self.publish_status(
            "terminal_replan_requested",
            completed_goal=(
                None if completed is None else [
                    round(float(completed[2]), 3),
                    round(float(completed[3]), 3),
                ]
            ),
            awaiting_controller_lease_release_ack=bool(awaiting_ack),
        )
        # Bypass only the compute-throttle, never route validation. The next
        # fixed lifecycle tick consumes this fact; no callback-owned timer is
        # needed for the successor boundary.
        self.last_planning_wall = 0.0
        if awaiting_ack:
            rospy.loginfo(
                "Global frontier holds successor planning until controller "
                "lease ACK route_id=%d",
                int(terminal_contract.get("route_id", 0) or 0)
                if terminal_contract is not None
                else 0,
            )
            return
        lifecycle = getattr(self, "lifecycle_manager", None)
        if lifecycle is not None:
            lifecycle.enqueue_type(
                EventType.REPLAN_REQUESTED,
                {"event": "terminal_replan"},
            )
            return
        if self.immediate_plan_timer is None and not rospy.is_shutdown():
            self.immediate_plan_timer = rospy.Timer(
                rospy.Duration(0.01), self.on_immediate_plan, oneshot=True
            )
        rospy.loginfo(
            "Global frontier schedules immediate successor selection "
            "after terminal delta=%.3fm route_id=%d pending=%s",
            terminal_delta,
            self.active_route_id,
            self.prefetched_frontier is not None,
        )

    def _arm_terminal_drain_timer(self):
        """Schedule a fast, non-blocking retry for queued terminal facts."""
        if getattr(self, "lifecycle_manager", None) is not None:
            return
        with self.terminal_ingress_lock:
            if (
                not self.pending_execution_terminals
                or self.terminal_drain_timer is not None
                or getattr(rospy, "is_shutdown", lambda: False)()
            ):
                return
            self.terminal_drain_timer = rospy.Timer(
                rospy.Duration(0.01),
                self.on_terminal_drain_timer,
                oneshot=True,
            )

    def on_execution_terminal(self, message):
        """Queue a terminal immediately; never wait for the planning lock."""
        # Keep this method self-contained for the narrow AST policy fixtures
        # that load the callback without executing module-level imports.
        from copy import deepcopy

        lifecycle = getattr(self, "lifecycle_manager", None)
        if lifecycle is not None:
            return lifecycle.enqueue_type(
                EventType.EXECUTION_TERMINAL, deepcopy(message)
            )
        # Compatibility doubles and old external compositions may not have
        # the runtime ingress state yet. Preserve their synchronous contract;
        # the production node initializes the queue before subscriptions.
        if not hasattr(self, "terminal_ingress_lock"):
            self._process_execution_terminal_locked(message)
            return
        terminal = deepcopy(message)
        # Do this before scheduling the drain timer. If the planning thread is
        # already inside a slow candidate pass, it can cooperatively abandon
        # that obsolete snapshot even though the callback never takes the
        # planning lock.
        with self.terminal_ingress_lock:
            proposal_gate = getattr(self, "planning_proposal_gate", None)
            if proposal_gate is not None:
                proposal_gate.invalidate("execution_terminal")
            self.planning_preempt_requested = True
            self.planning_preempt_reason = "execution_terminal"
            if len(self.pending_execution_terminals) >= (
                self.pending_execution_terminals.maxlen
            ):
                self.pending_execution_terminals.popleft()
                self.terminal_ingress_dropped += 1
            self.pending_execution_terminals.append(terminal)
        self._arm_terminal_drain_timer()

    def _process_execution_terminal_locked(self, message):
        """Apply one queued terminal at the normal planning boundary."""
        terminal = self.matching_execution_terminal(message)
        if terminal is None:
            return
        completed, terminal_delta = terminal
        consume_connector = getattr(
            self, "consume_connector_terminal", lambda: False,
        )
        if consume_connector():
            return
        self.record_execution_terminal(completed)
        if self.promote_prefetched_terminal(message):
            return
        self.schedule_terminal_replan(completed, terminal_delta)

    def _drain_execution_terminals_locked(self):
        """Drain a stable batch while the planning lock is already held."""
        if getattr(self, "lifecycle_manager", None) is not None:
            return
        with self.terminal_ingress_lock:
            pending = list(self.pending_execution_terminals)
            self.pending_execution_terminals.clear()
            self.terminal_drain_timer = None
        for message in pending:
            self._process_execution_terminal_locked(message)
        self._arm_terminal_drain_timer()

    def on_terminal_drain_timer(self, _event):
        """Try the planning lock without turning terminal ingress into a wait."""
        if getattr(self, "lifecycle_manager", None) is not None:
            return
        # A one-shot timer is no longer a pending wake once its callback starts.
        # Clear the handle before trying the planning lock; otherwise a busy
        # planning cycle makes ``_arm_terminal_drain_timer`` see the expired
        # handle and leaves queued terminal facts stranded indefinitely.
        with self.terminal_ingress_lock:
            self.terminal_drain_timer = None
        # The slow planner owns the state transition until its proposal has
        # either committed or been discarded. The terminal is already queued
        # and will be drained by the cycle's short cleanup boundary; running
        # it concurrently would mutate the same route lease mid-selection.
        if getattr(self, "planning_cycle_active", False):
            return
        acquired = self.planning_lock.acquire(False)
        if not acquired:
            self._arm_terminal_drain_timer()
            return
        try:
            self._drain_execution_terminals_locked()
        finally:
            self.planning_lock.release()
