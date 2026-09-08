"""Immediate successor scheduling after a verified route terminal.

The terminal topic is an execution-critical ingress.  Its ROS callback must not
wait for the frontier planner's potentially long snapshot/BFS cycle, otherwise
TEB can sit at a zero command until move_base starts recovery.  Messages are
therefore queued first and consumed under the normal planning lock by a short
one-shot drain timer.
"""

import rospy


class GlobalFrontierTerminalReplanMixin:
    """Release completed work and request one fresh planning snapshot."""

    def schedule_terminal_replan(self, completed, terminal_delta):
        """Release a finished lease and schedule normal successor selection."""
        # Do not let the next SLAM timer re-project a completed mission endpoint
        # onto a nearby grid cell and publish a synthetic follow-up action.
        self.release_active_frontier(
            preserve_place_departure=bool(self.place_departure.active),
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
        )
        # Bypass only the compute-throttle, never route validation. A one-shot
        # timer keeps work outside the action callback.
        self.last_planning_wall = 0.0
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
        with self.terminal_ingress_lock:
            pending = list(self.pending_execution_terminals)
            self.pending_execution_terminals.clear()
            self.terminal_drain_timer = None
        for message in pending:
            self._process_execution_terminal_locked(message)
        self._arm_terminal_drain_timer()

    def on_terminal_drain_timer(self, _event):
        """Try the planning lock without turning terminal ingress into a wait."""
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
