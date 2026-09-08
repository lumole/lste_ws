"""Timer locking, error boundaries, and planning-cycle execution."""

import time
import traceback

import rospy

from global_frontier_planning_contract import PlanningProposal


class GlobalFrontierPlanningRuntimeMixin:

    def planning_should_preempt(self):
        """Return whether the current snapshot has lost execution ownership."""
        return bool(getattr(self, "planning_preempt_requested", False))

    def _clear_planning_preempt_if_drained(self):
        """Clear terminal invalidation only after its ingress batch is drained."""
        ingress_lock = getattr(self, "terminal_ingress_lock", None)
        pending = getattr(self, "pending_execution_terminals", None)
        if ingress_lock is None or pending is None:
            self.planning_preempt_requested = False
            self.planning_preempt_reason = ""
            return
        with ingress_lock:
            if not pending:
                self.planning_preempt_requested = False
                self.planning_preempt_reason = ""

    def _planning_proposal_is_current(self, proposal):
        """Check the optimistic proposal and the legacy terminal flag."""
        if proposal is None:
            return not self.planning_should_preempt()
        gate = getattr(self, "planning_proposal_gate", None)
        if gate is None:
            return not self.planning_should_preempt()
        decision = gate.check(
            proposal.token,
            getattr(self, "active_route_id", 0),
        )
        return bool(decision.accepted and not self.planning_should_preempt())

    def _finish_planning_cycle(self, token):
        """Drain queued execution facts after the slow pass releases state."""
        gate = getattr(self, "planning_proposal_gate", None)
        with self.planning_lock:
            if gate is not None:
                gate.finish(token)
            drain_terminals = getattr(
                self, "_drain_execution_terminals_locked", None
            )
            if drain_terminals is not None:
                drain_terminals()
            self._clear_planning_preempt_if_drained()
        self.planning_cycle_proposal = None

    def _prepare_planning_cycle(self):
        """Perform only short ingress work before the expensive selector."""
        with self.planning_lock:
            drain_terminals = getattr(
                self, "_drain_execution_terminals_locked", None
            )
            if drain_terminals is not None:
                drain_terminals()
            self._clear_planning_preempt_if_drained()
            decision_wake = None
            decision_scheduler = getattr(self, "decision_wake_scheduler", None)
            event_driven = bool(
                getattr(self, "event_driven_deliberation_enabled", False)
            )
            if event_driven and self.active_frontier is None:
                decision_wake = (
                    None
                    if decision_scheduler is None
                    else decision_scheduler.claim()
                )
                # Navfn workers enqueue a scheduler wake when a result is
                # committed.  Do not use the result ledger itself as a second
                # wake source: an unmatched/stale result would make every
                # timer tick re-enter the expensive planner indefinitely.
                # The fallback is retained only for narrow legacy fixtures
                # that do not expose a scheduler at all.
                navfn_result_ready = bool(
                    decision_scheduler is None
                    and getattr(self, "_navfn_validation_result_ready", lambda: False)()
                )
                if decision_wake is None and not navfn_result_ready:
                    return None, True
                if decision_wake is not None:
                    wake_payload = decision_wake.as_dict()
                else:
                    wake_payload = {
                        "sequence": None,
                        "reasons": ["navfn_validation_completed"],
                        "reason": "navfn_validation_completed",
                    }
                self.publish_status(
                    "graph_decision_wake",
                    decision_wake=wake_payload,
                )
            gate = getattr(self, "planning_proposal_gate", None)
            token = (
                None
                if gate is None
                else gate.begin(
                    active_route_id=getattr(self, "active_route_id", 0),
                    wake_sequence=(
                        None if decision_wake is None else decision_wake.sequence
                    ),
                )
            )
            self.planning_cycle_token = token
            return token, False

    def on_timer(self, _event):
        cycle_lock = getattr(self, "planning_cycle_lock", None)
        if cycle_lock is not None and not cycle_lock.acquire(False):
            self.planning_cycle_skipped = int(
                getattr(self, "planning_cycle_skipped", 0)
            ) + 1
            return
        started = time.monotonic()
        token = None
        self.planning_cycle_active = True
        self.planning_cycle_started_wall = started
        try:
            try:
                token, skip = self._prepare_planning_cycle()
                if skip:
                    return
                self._on_timer(_event, token=token)
            except Exception:
                rospy.logerr(
                    "Global frontier timer failed; retaining the last goal: %s",
                    traceback.format_exc().strip(),
                )
        finally:
            self._finish_planning_cycle(token)
            self.planning_cycle_token = None
            self.planning_cycle_active = False
            elapsed = time.monotonic() - started
            # A timer can finish after rospy has torn down its clock during
            # tests or an operator stop.  Teardown is not a planning failure;
            # keep the final heartbeat best-effort and avoid a stray thread
            # traceback obscuring the actual run result.
            try:
                rospy.loginfo_throttle(
                    5.0,
                    "Global frontier heartbeat active=%s pending=%s map_ready=%s "
                    "costmap_ready=%s cycle=%.3fs lock_skips=%d",
                    self.active_frontier is not None,
                    self.prefetched_frontier is not None,
                    self.map_msg is not None,
                    self.fresh_costmap() is not None,
                    elapsed,
                    int(getattr(self, "planning_cycle_skipped", 0)),
                )
            except Exception:
                pass
            if elapsed > 1.5:
                try:
                    rospy.logwarn(
                        "Global frontier cycle slow: %.3fs active=%s pending=%s "
                        "proposal_generation=%s",
                        elapsed,
                        self.active_frontier is not None,
                        self.prefetched_frontier is not None,
                        getattr(
                            getattr(self, "planning_proposal_gate", None),
                            "generation",
                            None,
                        ),
                    )
                except Exception:
                    pass
            if cycle_lock is not None:
                cycle_lock.release()

    def _on_timer(self, _event, token=None):
        pose = self.planning_pose_snapshot()
        if pose is None:
            return
        message, robot_map, robot_yaw_map, now = pose
        if self.active_route_planning_is_deferred(robot_map, now):
            return
        self.last_planning_wall = now
        snapshot_started = time.monotonic()
        snapshot = self.build_frontier_planning_snapshot(
            message, robot_map, robot_yaw_map, now,
        )
        snapshot_elapsed = time.monotonic() - snapshot_started
        if snapshot_elapsed > 1.0:
            rospy.logwarn(
                "Global frontier planning phase=snapshot elapsed=%.3fs "
                "map=%dx%d active=%s",
                snapshot_elapsed,
                int(message.info.width),
                int(message.info.height),
                self.active_frontier is not None,
            )
        if snapshot is None:
            return
        if self.planning_should_preempt() or (
            token is not None
            and not self._planning_proposal_is_current(
                PlanningProposal(token, None)
            )
        ):
            rospy.loginfo(
                "Global frontier planning preempted phase=post_snapshot reason=%s",
                getattr(self, "planning_preempt_reason", "unknown"),
            )
            return
        self.commit_departed_place_after_boundary_crossing(snapshot)
        replay = getattr(self, "replay_deferred_replan_if_ready", None)
        if replay is not None:
            replay()
        route_graph = snapshot.route_graph
        map_context = snapshot.map_context
        active_update_started = time.monotonic()
        active_cell, held_waypoint_map = self.update_active_frontier(snapshot)
        active_update_elapsed = time.monotonic() - active_update_started
        if active_update_elapsed > 1.0:
            rospy.logwarn(
                "Global frontier planning phase=active_update elapsed=%.3fs "
                "active=%s",
                active_update_elapsed,
                self.active_frontier is not None,
            )
        if self.planning_should_preempt() or (
            token is not None
            and not self._planning_proposal_is_current(
                PlanningProposal(token, None)
            )
        ):
            rospy.loginfo(
                "Global frontier planning preempted phase=active_update reason=%s",
                getattr(self, "planning_preempt_reason", "unknown"),
            )
            return
        if active_cell is None:
            selection_started = time.monotonic()
            active_cell, selection_mode, wait_for_validation = (
                self.select_next_active_frontier(snapshot)
            )
            selection_elapsed = time.monotonic() - selection_started
            if selection_elapsed > 1.0:
                report = getattr(
                    self, "last_local_work_item_rehydration_report", {}
                ) or {}
                rospy.logwarn(
                    "Global frontier planning phase=selection elapsed=%.3fs "
                    "wait=%s work_candidates=%s frontier_candidates=%s",
                    selection_elapsed,
                    bool(wait_for_validation),
                    report.get("candidate_count"),
                    getattr(self, "last_frontier_candidates_seen", None),
                )
            if self.planning_should_preempt() or (
                token is not None
                and not self._planning_proposal_is_current(
                    PlanningProposal(token, None)
                )
            ):
                rospy.loginfo(
                    "Global frontier planning preempted phase=selection reason=%s",
                    getattr(self, "planning_preempt_reason", "unknown"),
                )
                return
            if wait_for_validation:
                return
            if active_cell is None:
                self._report_no_frontier_selection()
                return
            proposal = PlanningProposal(
                token=token,
                candidate=active_cell,
                selection_mode=selection_mode,
                wait_for_validation=bool(wait_for_validation),
            )
            self.planning_cycle_proposal = proposal

            def commit_selected_route():
                # The expensive proposal is computed outside this lock. The
                # caller owns the short ``planning_lock -> proposal_gate``
                # commit order; keeping the callback lock-free avoids a
                # reverse acquisition cycle with terminal ingress.
                if not self.activate_selected_frontier(
                    message,
                    proposal.candidate,
                    map_context.components,
                    map_context.known_free,
                    now,
                    robot_map,
                    robot_yaw_map,
                    route_graph.route_steps,
                    route_graph.seed,
                    proposal.selection_mode,
                ):
                    return False
                self.publish_active_route_command(
                    message,
                    proposal.candidate,
                    held_waypoint_map,
                    route_graph.route_steps,
                    route_graph.seed,
                    robot_map,
                    robot_yaw_map,
                )
                return True

            gate = getattr(self, "planning_proposal_gate", None)
            if gate is None:
                with self.planning_lock:
                    commit_selected_route()
            else:
                with self.planning_lock:
                    decision = gate.commit_if_current(
                        proposal.token,
                        getattr(self, "active_route_id", 0),
                        commit_selected_route,
                    )
                if not decision.accepted:
                    rospy.loginfo(
                        "Global frontier discarded stale route proposal reason=%s",
                        decision.reason,
                    )
            return

        proposal = PlanningProposal(
            token=token,
            candidate=active_cell,
            selection_mode="active_route_refresh",
        )
        self.planning_cycle_proposal = proposal
        gate = getattr(self, "planning_proposal_gate", None)

        def commit():
            self.publish_active_route_command(
                message,
                proposal.candidate,
                held_waypoint_map,
                route_graph.route_steps,
                route_graph.seed,
                robot_map,
                robot_yaw_map,
            )
            return True

        if gate is None:
            with self.planning_lock:
                commit()
        else:
            with self.planning_lock:
                decision = gate.commit_if_current(
                    proposal.token,
                    getattr(self, "active_route_id", 0),
                    commit,
                )
            if not decision.accepted:
                rospy.loginfo(
                    "Global frontier discarded stale route refresh reason=%s",
                    decision.reason,
                )
