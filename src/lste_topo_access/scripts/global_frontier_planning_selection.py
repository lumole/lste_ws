"""Terminal successor selection and no-frontier reporting policy."""

import rospy

from global_frontier_graph_route_planner import (
    PLAN_READY,
)
from global_frontier_graph_route_gate import (
    candidate_refines_graph_plan,
    graph_plan_is_exclusive,
)


class GlobalFrontierPlanningSelectionMixin:

    def terminal_prefetch_novelty_decision(self, snapshot):
        """Prefer a newly exposed place over a cached revisit at a terminal.

        The return tuple is ``(candidate, mode, wait_for_validation)``.  A
        validation wait is distinct from no candidate: it must preserve the
        current decision rather than reporting exploration as exhausted.
        """
        if self.prefetched_frontier is None:
            return None, None, False
        pending_x, pending_y = self.prefetched_goal_map
        pending_information = (
            0.0
            if self.prefetched_frontier_information is None
            else self.prefetched_frontier_information
        )
        pending_tier, pending_region = self.region_memory.candidate_tier(
            pending_x,
            pending_y,
            pending_information,
            component=self.prefetched_frontier_component,
        )
        if pending_tier == "dormant":
            rospy.loginfo(
                "Global frontier discarded dormant prefetched region "
                "id=%s map=(%.2f,%.2f)",
                None if pending_region is None else pending_region["id"],
                pending_x,
                pending_y,
            )
            self.publish_status(
                "frontier_prefetch_discarded",
                reason="prefetched_region_dormant",
                pending_goal=[round(float(pending_x), 3), round(float(pending_y), 3)],
                pending_region_id=(
                    None if pending_region is None else int(pending_region["id"])
                ),
            )
            self.clear_prefetched_frontier()
            return None, None, False
        if pending_tier != "revisit":
            return None, None, False
        novel = self._choose_frontier_for_new_action(
            snapshot.message,
            snapshot.route_graph,
            snapshot.map_context,
            snapshot.now,
            snapshot.robot_map,
            snapshot.robot_yaw_map,
            allowed_region_tiers=("new",),
            include_semantic=False,
        )
        if novel is None:
            return None, None, self.frontier_validation_pending
        novel_component = novel[8] if len(novel) > 8 else None
        rospy.loginfo(
            "Global frontier replaced revisited prefetched branch "
            "map=(%.2f,%.2f) region=%s with newly exposed region "
            "map=(%.2f,%.2f)",
            pending_x,
            pending_y,
            None if pending_region is None else pending_region["id"],
            novel[2],
            novel[3],
        )
        self.publish_status(
            "frontier_prefetch_superseded",
            reason="new_region_available_at_terminal",
            prefetched_goal=[round(float(pending_x), 3), round(float(pending_y), 3)],
            prefetched_region_id=(
                None if pending_region is None else int(pending_region["id"])
            ),
            selected_goal=[round(float(novel[2]), 3), round(float(novel[3]), 3)],
            selected_component_cells=(
                None if novel_component is None else int(novel_component["cells"])
            ),
        )
        self.clear_prefetched_frontier()
        return novel, "terminal_new_region", False

    def select_pending_local_egress(self, snapshot):
        """Convert one queued safe-history anchor into a normal route command."""
        pending = self.pending_local_egress
        if pending is None:
            return None, None, False
        x, y = pending["anchor_map"]
        route_graph = snapshot.route_graph
        reassociated = self.nearest_reachable_cell(
            snapshot.message,
            route_graph.route_steps,
            x,
            y,
        )
        costmap_distance = self.candidate_costmap_distance(
            route_graph.validation, x, y,
        )
        reachable = self.navfn_goal_reachable(
            snapshot.robot_map,
            (x, y),
            snapshot.message.header.frame_id or "map",
        )
        if reachable is None:
            return None, None, True
        if reassociated is None or reachable is False or (
            route_graph.validation is not None and costmap_distance is None
        ):
            self.publish_status(
                "local_egress_discarded",
                source_route_id=int(pending["source_route_id"]),
                recovery_goal=[round(float(x), 3), round(float(y), 3)],
                reason="egress_anchor_not_reachable",
            )
            if pending.get("resume_portal_transition", False):
                self.discard_pending_portal_retry("egress_anchor_not_reachable")
            self.pending_local_egress = None
            return None, None, False
        row, col = reassociated
        path_distance = (
            float(costmap_distance)
            if costmap_distance is not None
            else float(route_graph.route_steps[row, col])
            * float(snapshot.message.info.resolution)
        )
        self.active_local_egress_resumes_portal = bool(
            pending.get("resume_portal_transition", False)
        )
        self.pending_local_egress = None
        self.publish_status(
            "local_egress_selected",
            source_route_id=int(pending["source_route_id"]),
            recovery_goal=[round(float(x), 3), round(float(y), 3)],
            reason=pending["reason"],
            path_distance=round(path_distance, 3),
            resume_portal_transition=self.active_local_egress_resumes_portal,
        )
        return (
            row,
            col,
            float(x),
            float(y),
            path_distance,
            0.0,
            0.0,
            0.0,
            None,
            0,
            "local_egress",
        ), "local_egress_recovery", False

    def select_next_active_frontier(self, snapshot):
        """Choose a terminal successor after resolving cache and claim policy."""
        transaction = getattr(self, "portal_transaction", None)
        if (
            transaction is not None
            and transaction.active
            and getattr(self, "pending_local_egress", None) is None
            and getattr(self, "pending_portal_retry", None) is None
        ):
            # A crossing_verified transaction may be waiting for the fresh
            # structural snapshot that can commit its destination Place. A
            # normal frontier is not an admissible substitute for that edge.
            self.publish_status(
                "portal_transaction_selection_blocked",
                transaction_id=int(transaction.snapshot().transaction_id),
                route_id=int(transaction.snapshot().route_id),
                state=transaction.state,
                reason="portal_transaction_owned",
            )
            return None, "portal_transaction_wait", True
        candidate, selection_mode, wait_for_validation = (
            self.select_pending_local_egress(snapshot)
        )
        if candidate is not None or wait_for_validation:
            return candidate, selection_mode, wait_for_validation
        candidate, selection_mode, wait_for_validation = (
            self.select_pending_portal_retry(snapshot)
        )
        if candidate is not None or wait_for_validation:
            return candidate, selection_mode, wait_for_validation

        # A durable graph action already has a physical identity. Give its
        # direct materializer first refusal before the generic frontier scorer
        # scans every boundary cell. This is the fast path for a pending
        # PortalProbe or remembered egress/crossing after a terminal.
        place_entry_rehydration_pending = getattr(
            self, "place_entry_rehydration_pending", None
        ) is not None
        if (
            getattr(self, "graph_route_planner_enabled", False)
            and not place_entry_rehydration_pending
        ):
            graph_prepare = getattr(self, "_prepare_graph_route_action", None)
            if graph_prepare is not None:
                try:
                    _graph_plan, graph_candidate = graph_prepare(
                        snapshot, reuse_existing_plan=True,
                    )
                except TypeError:
                    _graph_plan, graph_candidate = graph_prepare(snapshot)
                if graph_candidate is not None:
                    return graph_candidate, "graph_route_edge", False

        # The full graph method is a two-stage planner: the current map first
        # rehydrates WorkItems/Probes, then the durable graph owns the next
        # action.  Do not let a terminal prefetch or an unrelated frontier
        # bypass that decision.  Baseline methods retain their historical
        # prefetch order below for a clean experiment contract.
        if getattr(self, "graph_route_planner_enabled", False):
            candidate = self._choose_frontier_for_new_action(
                snapshot.message,
                snapshot.route_graph,
                snapshot.map_context,
                snapshot.now,
                snapshot.robot_map,
                snapshot.robot_yaw_map,
            )
            if candidate is not None:
                graph_candidate = getattr(
                    self, "last_graph_selected_candidate", None
                ) or candidate
                graph_plan = getattr(self, "last_graph_route_plan", None)
                if graph_plan is None:
                    graph_prepare = getattr(
                        self, "_prepare_graph_route_action", None
                    )
                    if graph_prepare is not None:
                        graph_plan, _unused_graph_candidate = graph_prepare(None)
                commit = getattr(self, "_commit_graph_route_candidate", None)
                if graph_plan is None or commit is None:
                    return candidate, self.last_frontier_selection_mode, False
                if getattr(graph_plan, "status", None) != PLAN_READY:
                    # A blocked/complete graph decision is a hard barrier. Do
                    # not feed its candidate into the transaction layer: that
                    # only creates an event storm with no possible commit.
                    self.publish_status(
                        "graph_route_materialization_wait",
                        status=getattr(graph_plan, "status", None),
                        action=getattr(graph_plan, "action", None),
                        obligation_kind=getattr(graph_plan, "obligation_kind", ""),
                        obligation_id=getattr(graph_plan, "obligation_id", None),
                        reason="graph_plan_not_ready_before_candidate_commit",
                    )
                    return None, "graph_route_materialization_wait", True
                if graph_plan_is_exclusive(graph_plan) and not candidate_refines_graph_plan(
                    graph_plan, graph_candidate
                ):
                    signature = (
                        graph_plan.signature(),
                        getattr(graph_candidate, "graph_action", None),
                        getattr(graph_candidate, "work_item_id", None),
                        getattr(graph_candidate, "portal_id", None),
                    )
                    if signature != getattr(
                        self, "last_graph_candidate_unavailable_signature", None
                    ):
                        self.last_graph_candidate_unavailable_signature = signature
                        self.publish_status(
                            "graph_route_candidate_unavailable",
                            action=str(graph_plan.action),
                            obligation_id=graph_plan.obligation_id,
                            first_portal_id=graph_plan.first_portal_id,
                            reason="candidate_does_not_refine_graph_plan",
                        )
                    publish_unavailable = getattr(
                        self, "_publish_graph_route_unavailable", None
                    )
                    if publish_unavailable is not None:
                        publish_unavailable(
                            graph_plan,
                            "candidate_does_not_refine_graph_plan",
                        )
                    return None, "graph_route_materialization_wait", True
                if commit(graph_plan, graph_candidate, source="frontier_selector"):
                    return candidate, self.last_frontier_selection_mode, False
                # The transaction validator has already emitted the exact
                # mismatch reason. Keep the durable intent pending instead of
                # letting a second selector branch replace it.
                return None, "graph_route_materialization_wait", True

            # If the context pass could not run (for example a transient
            # missing frontier projection), keep the entry barrier intact.  A
            # direct graph fallback here would again see an empty ledger and
            # select the reverse Portal before the next map can rehydrate it.
            if getattr(self, "place_entry_rehydration_pending", None) is not None:
                self.publish_status(
                    "place_entry_rehydration_wait",
                    place_id=getattr(self, "current_physical_place_id", None),
                    reason="successor_requires_first_post_arrival_snapshot",
                )
                return None, "place_entry_rehydration_wait", True
            # ``choose_valid_frontier`` may have selected the exact durable
            # WorkItem/Portal candidate while its asynchronous Navfn check is
            # still pending.  That is a validation wait, not evidence that
            # graph materialization failed.  Do not release the controller
            # lease here: the Navfn worker schedules a planning wake when the
            # result arrives, and the same graph intent must be retried then.
            graph_plan = getattr(self, "last_graph_route_plan", None)
            if (
                getattr(self, "frontier_validation_pending", False)
                and graph_plan is not None
                and getattr(graph_plan, "status", None) == PLAN_READY
            ):
                retain_lease = getattr(
                    self, "_retain_graph_route_plan_lease", None
                )
                if retain_lease is not None:
                    retain_lease(graph_plan, "navfn_validation_pending")
                publish = getattr(self, "publish_status", None)
                if publish is not None:
                    signature = (
                        graph_plan.signature(),
                        "navfn_validation_pending",
                    )
                    if signature != getattr(
                        self, "last_graph_validation_wait_signature", None
                    ):
                        self.last_graph_validation_wait_signature = signature
                        publish(
                            "graph_route_materialization_wait",
                            status=graph_plan.status,
                            action=graph_plan.action,
                            obligation_kind=graph_plan.obligation_kind,
                            obligation_id=graph_plan.obligation_id,
                            portal_path=list(graph_plan.portal_path),
                            first_portal_id=graph_plan.first_portal_id,
                            reason="navfn_validation_pending",
                        )
                return None, "graph_route_materialization_wait", True
            graph_prepare = getattr(self, "_prepare_graph_route_action", None)
            if graph_prepare is None:
                graph_plan, graph_candidate = None, None
            else:
                try:
                    graph_plan, graph_candidate = graph_prepare(
                        snapshot, reuse_existing_plan=True,
                    )
                except TypeError:
                    graph_plan, graph_candidate = graph_prepare(snapshot)
            if graph_candidate is not None:
                return graph_candidate, "graph_route_edge", False
            if (
                graph_plan is not None
                and getattr(graph_plan, "status", None) == PLAN_READY
            ):
                retain_lease = getattr(self, "_retain_graph_route_plan_lease", None)
                if retain_lease is not None:
                    retain_lease(
                        graph_plan,
                        "selected_graph_obligation_not_executable_in_snapshot",
                    )
                signature = graph_plan.signature()
                if signature != getattr(
                    self, "last_graph_materialization_wait_signature", None
                ):
                    self.last_graph_materialization_wait_signature = signature
                    self.publish_status(
                        "graph_route_materialization_wait",
                        status=graph_plan.status,
                        action=graph_plan.action,
                        obligation_kind=graph_plan.obligation_kind,
                        obligation_id=graph_plan.obligation_id,
                        portal_path=list(graph_plan.portal_path),
                        first_portal_id=graph_plan.first_portal_id,
                        reason="selected_graph_obligation_not_executable_in_snapshot",
                    )
                publish_unavailable = getattr(
                    self, "_publish_graph_route_unavailable", None
                )
                if publish_unavailable is not None:
                    publish_unavailable(
                        graph_plan,
                        "selected_graph_obligation_not_executable_in_snapshot",
                    )
                return None, "graph_route_materialization_wait", True
            # A blocked/complete plan is handed back to the normal completion
            # gate. It may become executable after the next map or evidence
            # event, but it must not be replaced by an unrelated route here.
            return None, self.last_frontier_selection_mode, False

        candidate, selection_mode, wait_for_validation = (
            self.terminal_prefetch_novelty_decision(snapshot)
        )
        if candidate is not None or wait_for_validation:
            return candidate, selection_mode, wait_for_validation
        promoted = self.promote_prefetched_frontier(
            snapshot.message,
            snapshot.route_graph.route_steps,
            snapshot.robot_map,
            snapshot.now,
            validation=snapshot.route_graph.validation,
        )
        if promoted is not None:
            candidate = self.active_cell_from_promoted_frontier(
                snapshot.route_graph.validation,
                snapshot.route_graph.route_steps,
                snapshot.message.info.resolution,
                promoted,
            )
            return candidate, "prefetched_successor", False

        candidate = self._choose_frontier_for_new_action(
            snapshot.message,
            snapshot.route_graph,
            snapshot.map_context,
            snapshot.now,
            snapshot.robot_map,
            snapshot.robot_yaw_map,
        )
        if candidate is not None:
            return candidate, self.last_frontier_selection_mode, False
        if (
            candidate is None
            and not self.frontier_validation_budget_exhausted
            and not self.frontier_validation_pending
        ):
            durable_probe = getattr(self, "select_durable_portal_probe", None)
            if durable_probe is not None:
                candidate = durable_probe(snapshot)
                if candidate is not None:
                    self.graph_route_probe_id = None
                    return candidate, (
                        "durable_portal_probe"
                    ), False
            durable_egress = getattr(
                self, "select_durable_portal_egress", None
            )
            if durable_egress is not None:
                candidate = durable_egress(snapshot)
                if candidate is not None:
                    self.graph_route_portal_id = None
                    return candidate, (
                        "durable_portal_egress"
                    ), False
        if (
            candidate is None
            and self.target_region_claim_active
            and not self.frontier_validation_budget_exhausted
            and not self.frontier_validation_pending
        ):
            self._release_exhausted_target_region_claim()
            candidate = self._choose_frontier_for_new_action(
                snapshot.message,
                snapshot.route_graph,
                snapshot.map_context,
                snapshot.now,
                snapshot.robot_map,
                snapshot.robot_yaw_map,
            )
        return candidate, self.last_frontier_selection_mode, False
