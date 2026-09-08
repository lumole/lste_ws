"""Build and rank one frontier-selection snapshot."""

import math

import numpy as np

from global_frontier_action_policy import (
    ACTION_ADJACENT,
    ACTION_LOCAL,
    ACTION_PROBE,
    ACTION_TARGET_DIRECTION,
    ACTION_VIEWPOINT_RETRY,
    unique_action_tiers,
)
from global_frontier_frontier_decision import (
    FRONTIER_ACTION_ORDER,
    candidate_from_route,
    select_frontier_action,
)
from global_frontier_models import FrontierSelectionRequest
from global_frontier_portal_probe_adapter import (
    select_probe_route,
    selection_report,
)
from global_frontier_graph_route_gate import (
    candidate_refines_graph_plan,
    graph_plan_is_exclusive,
)
from global_frontier_graph_route_transaction import with_graph_action
from global_frontier_scoring import (
    initialise_score_pools,
    record_scored_candidate,
)


class GlobalFrontierSelectionPlannerMixin:
    """Convert map evidence into local endpoints or one portal action."""

    def choose_frontier(
        self, message, steps, frontier, unknown, occupied, now, excluded=None,
        validation=None, robot_map=None, heading_reference=None,
        max_heading_delta=None, route_seed=None, semantic_hint=None,
        semantic_pursuit=False,
        preferred_steps=None, preferred_mask=None,
        selection_tier="strict_clearance", route_anchor_xy=None,
        score_path_from_steps=False, components=None, allowed_region_tiers=None,
        allow_portal_transitions=False, graph_route_planning=True,
    ):
        """Select a local endpoint or one deliberate doorway transition."""
        self.last_frontier_decision = None
        # Keep the named candidate alongside the legacy route tuple until the
        # outer lifecycle commits it.  The tuple is the public compatibility
        # boundary, but it cannot carry the candidate's physical Place ID;
        # dropping that identity would reject a valid same-Place refinement.
        self.last_graph_selected_candidate = None
        route_anchor_xy = robot_map if route_anchor_xy is None else route_anchor_xy
        header = getattr(message, "header", None)
        map_frame = getattr(header, "frame_id", "map") or "map"
        projector_factory = getattr(self, "planar_xy_projector", None)
        map_to_physical = (
            None
            if projector_factory is None
            else projector_factory("odom", map_frame)
        )
        request = FrontierSelectionRequest(
            message=message,
            steps=steps,
            unknown=unknown,
            occupied=occupied,
            now=now,
            excluded=excluded,
            validation=validation,
            robot_map=robot_map,
            heading_reference=heading_reference,
            max_heading_delta=max_heading_delta,
            route_seed=route_seed,
            semantic_hint=semantic_hint,
            semantic_pursuit=semantic_pursuit,
            preferred_steps=preferred_steps,
            preferred_mask=preferred_mask,
            selection_tier=selection_tier,
            route_anchor_xy=route_anchor_xy,
            score_path_from_steps=score_path_from_steps,
            components=components,
            allowed_region_tiers=allowed_region_tiers,
            allow_portal_transitions=allow_portal_transitions,
            map_to_physical_xy=map_to_physical,
        )
        minimum_steps = int(
            math.ceil(self.min_path_distance / message.info.resolution)
        )
        approach_cells = max(
            1,
            int(
                math.ceil(
                    self.frontier_approach_distance / message.info.resolution
                )
            ),
        )
        rows, cols = np.nonzero(frontier)
        structural_boundary_path = bool(
            graph_route_planning
            and getattr(self, "graph_route_planner_enabled", False)
            and getattr(self, "branch_first_enabled", False)
            and getattr(self, "structural_boundary_probe_candidates", None)
            is not None
        )
        can_rehydrate_local_work = bool(
            graph_route_planning
            and getattr(self, "graph_route_planner_enabled", False)
            and getattr(self, "pending_local_work_item_candidates", None)
            is not None
        )
        place_entry_rehydration_pending = (
            getattr(self, "place_entry_rehydration_pending", None) is not None
        )
        if (
            rows.size == 0
            and not structural_boundary_path
            and not can_rehydrate_local_work
            and not place_entry_rehydration_pending
        ):
            return None
        # Reconcile the durable Place/WorkItem ledger before checking whether
        # this snapshot contains ordinary frontier cells. A Portal arrival
        # deliberately installs ``place_entry_rehydration_pending`` so that
        # the first post-arrival map owns this handoff. The previous order
        # returned early when ``frontier`` was empty, leaving that flag set
        # forever and making the event-driven planner wait for a map event
        # that could never be required again. Context construction is the
        # explicit one-snapshot reconciliation boundary.
        context = self._prepare_frontier_selection_context(
            message,
            steps,
            frontier,
            unknown,
            occupied,
            route_anchor_xy,
            components,
            now,
            map_to_physical,
        )
        self.last_selection_context = context
        if rows.size > self.candidate_limit:
            indices = np.linspace(
                0, rows.size - 1, self.candidate_limit, dtype=np.int64,
            )
            rows, cols = rows[indices], cols[indices]

        # Reconcile the current map into durable identities before the slow
        # graph planner chooses an obligation.  A plan made one stage earlier
        # could otherwise point at an old WorkItem while this snapshot creates
        # its successor, allowing a different local candidate to win.
        graph_plan = None
        graph_prepare = getattr(self, "_prepare_graph_route_action", None)
        if graph_route_planning and graph_prepare is not None and getattr(
            self, "graph_route_planner_enabled", False
        ):
            reuse_graph_plan_lease = bool(
                getattr(self, "graph_route_plan_lease_active", False)
            )
            try:
                prepare_kwargs = {
                    "visible_work_item_ids": getattr(
                        context, "work_item_details", None
                    ),
                    "reuse_existing_plan": reuse_graph_plan_lease,
                }
                if not reuse_graph_plan_lease:
                    # The first pass only installs an identity preference for
                    # candidate collection.  It must not create a graph
                    # transaction before the complete snapshot is known.
                    prepare_kwargs["stage_only"] = True
                graph_plan, _unused_graph_candidate = graph_prepare(
                    None, **prepare_kwargs,
                )
            except TypeError:
                # Keep narrow legacy fixtures/source-compatible while the ROS
                # adapter adopts the named visibility contract.
                graph_plan, _unused_graph_candidate = graph_prepare(None)
        # A failed physical viewpoint is retried only through another safe
        # endpoint tied to the same WorkItem support, never as a generic goal.
        ranking_action_tiers = unique_action_tiers(
            (
                ACTION_VIEWPOINT_RETRY,
                ACTION_TARGET_DIRECTION,
                *tuple(context.action_tiers),
            ),
            include_probe=True,
        )
        pools = initialise_score_pools(ranking_action_tiers)
        portal_sources = {}
        probe_candidates = []
        frontier_decision_candidates = []
        graph_obligation_candidates = []
        self._collect_frontier_score_pools(
            request,
            rows,
            cols,
            context,
            pools,
            portal_sources,
            minimum_steps,
            approach_cells,
            probe_candidates=probe_candidates,
            frontier_decision_candidates=frontier_decision_candidates,
            graph_obligation_candidates=graph_obligation_candidates,
        )
        if getattr(self, "planning_should_preempt", lambda: False)():
            return None
        rehydrated_candidates = []
        rehydrate_work = getattr(
            self, "pending_local_work_item_candidates", None,
        )
        if rehydrate_work is not None and graph_route_planning:
            rehydrated_candidates.extend(
                rehydrate_work(request, context, frontier, approach_cells)
            )
        if getattr(self, "planning_should_preempt", lambda: False)():
            return None
        rehydrate_probes = getattr(self, "pending_portal_probe_candidates", None)
        if rehydrate_probes is not None:
            rehydrated_candidates.extend(
                rehydrate_probes(request, context, frontier, approach_cells)
            )
        if getattr(self, "planning_should_preempt", lambda: False)():
            return None
        # The full graph-first method also compiles a doorway when its far side
        # is already known free.  That opening is no longer a frontier, so it
        # cannot be found by the normal per-frontier loop above.  Keep this
        # capability behind the named branch-first method contract; baselines
        # and the strict ablation retain their historical frontier semantics.
        structural_boundary_probes = getattr(
            self, "structural_boundary_probe_candidates", None
        )
        if (
            structural_boundary_probes is not None
            and graph_route_planning
            and getattr(self, "graph_route_planner_enabled", False)
            and getattr(self, "branch_first_enabled", False)
        ):
            rehydrated_candidates.extend(
                structural_boundary_probes(request, context, approach_cells)
            )
        if getattr(self, "planning_should_preempt", lambda: False)():
            return None
        for frontier_row, frontier_col, candidate in rehydrated_candidates:
            scored = self._score_frontier_endpoint(
                request,
                frontier_row,
                frontier_col,
                candidate,
                context,
            )
            if scored is None:
                continue
            (
                scored_candidate,
                pool_name,
                pursuit_candidate,
                ranking_action_tier,
            ) = scored
            record_scored_candidate(
                pools,
                ranking_action_tier,
                scored_candidate,
                pool_name,
                pursuit_candidate,
                request.steps[candidate.row, candidate.col],
                minimum_steps,
                request.selection_tier,
                self.min_structure_cells,
            )
            route_steps = request.steps[candidate.row, candidate.col]
            route_is_executable = (
                route_steps >= minimum_steps
                or request.selection_tier == "navfn_observation_recovery"
            )
            if frontier_decision_candidates is not None and route_is_executable:
                frontier_decision_candidates.append(
                    candidate_from_route(
                        scored_candidate,
                        region_tier=pool_name,
                        ranking_category=ranking_action_tier,
                        map_epoch=getattr(request.components, "epoch", None),
                        place_id=context.source_place_id,
                    )
                )
            if graph_obligation_candidates is not None:
                graph_obligation_candidates.append(
                    candidate_from_route(
                        scored_candidate,
                        region_tier=pool_name,
                        ranking_category=ranking_action_tier,
                        map_epoch=getattr(request.components, "epoch", None),
                        place_id=context.source_place_id,
                    )
                )
            if (
                len(scored_candidate) > 15
                and scored_candidate[15] is not None
            ):
                probe_candidates.append(
                    (scored_candidate, pool_name, ranking_action_tier)
                )

        # Candidate construction can rehydrate a probe and bind a WorkItem.
        # Replan once more from the identities that actually survived route
        # admission; the initial context-only plan must not retain a stale
        # local ID created before this snapshot was fully interpreted.
        if graph_route_planning and graph_prepare is not None and getattr(
            self, "graph_route_planner_enabled", False
        ):
            reuse_graph_plan_lease = bool(
                getattr(self, "graph_route_plan_lease_active", False)
            )
            visible_ids = {
                candidate.work_item_id
                for candidate in graph_obligation_candidates
                if getattr(candidate, "work_item_id", None) is not None
            }
            visible_probe_ids = {
                candidate.portal_id
                for candidate in graph_obligation_candidates
                if getattr(candidate, "portal_id", None) is not None
            }
            try:
                graph_plan, _unused_graph_candidate = graph_prepare(
                    None,
                    visible_work_item_ids=visible_ids,
                    visible_probe_ids=visible_probe_ids,
                    reuse_existing_plan=reuse_graph_plan_lease,
                )
            except TypeError:
                graph_plan, _unused_graph_candidate = graph_prepare(None)

        # ``hold`` is a graph decision, not an empty score pool. Once the
        # durable planner reports a blocked plan, stop this snapshot before
        # generic frontier ranking can manufacture a candidate and collide
        # with the blocked transaction on every timer tick.
        if (
            graph_plan is not None
            and getattr(graph_plan, "status", None) == "blocked"
        ):
            clear_lease = getattr(self, "_clear_graph_route_plan_lease", None)
            if clear_lease is not None:
                clear_lease("blocked_plan")
            signature = graph_plan.signature()
            if signature != getattr(self, "last_graph_blocked_signature", None):
                self.last_graph_blocked_signature = signature
                self.publish_status(
                    "graph_route_blocked",
                    action=str(graph_plan.action),
                    reason=str(graph_plan.reason),
                    current_place_id=graph_plan.current_place_id,
                    obligation_kind=str(graph_plan.obligation_kind),
                    obligation_id=graph_plan.obligation_id,
                )
            return None
        if graph_plan is not None:
            self.last_graph_blocked_signature = None

        # The durable graph planner may have selected one particular Portal
        # edge as the next topological action. Materialize that identity before
        # any snapshot-local score pool can substitute a different doorway.
        if getattr(self, "graph_route_portal_id", None) is not None:
            selected = self._select_context_portal_transition(
                request, context, portal_sources,
            )
            if selected is not None:
                self.graph_route_portal_id = None
                return selected
        if graph_plan_is_exclusive(graph_plan):
            # A local intent may be promoted by a newly certified, outward
            # Portal only through the explicit branch-first policy.  The
            # transaction adapter records this as a plan reconciliation; it is
            # not a hidden score-based substitution.
            if (
                graph_plan.action in ("observe_local_work", "retry_viewpoint")
                and getattr(self, "branch_first_enabled", False)
            ):
                selected = self._select_context_portal_transition(
                    request, context, portal_sources,
                )
                if selected is not None:
                    # Candidate collection may have discovered and certified
                    # the doorway only after the durable planner's first
                    # pass. Commit that explicit branch-first promotion so the
                    # graph transaction changes from local WorkItem intent to
                    # the physical Portal edge before activation publishes it.
                    commit = getattr(
                        self, "_commit_graph_route_candidate", None,
                    )
                    if commit is None or commit(
                        graph_plan,
                        selected,
                        source="branch_first_portal_promotion",
                    ):
                        return selected
                    return None
            selected = self._select_graph_plan_candidate(
                graph_obligation_candidates, graph_plan, request,
            )
            if selected is not None:
                self.graph_route_probe_id = None
                self.graph_route_portal_id = None
                clear_lease = getattr(self, "_clear_graph_route_plan_lease", None)
                if clear_lease is not None:
                    clear_lease("graph_candidate_selected")
                self.last_graph_candidate_unavailable_signature = None
                self.last_graph_materialization_wait_signature = None
                return selected
            # A ready graph obligation is not equivalent to an arbitrary
            # frontier.  Keep it pending until the same durable boundary is
            # reprojected by a later map snapshot or the dedicated probe/egress
            # adapter can materialize it.
            retain_lease = getattr(self, "_retain_graph_route_plan_lease", None)
            if retain_lease is not None:
                retain_lease(graph_plan, "durable_identity_not_in_current_frontier_snapshot")
            signature = graph_plan.signature()
            if signature != getattr(
                self, "last_graph_candidate_unavailable_signature", None
            ):
                self.last_graph_candidate_unavailable_signature = signature
                self.publish_status(
                    "graph_route_candidate_unavailable",
                    action=str(graph_plan.action),
                    obligation_kind=str(graph_plan.obligation_kind),
                    obligation_id=graph_plan.obligation_id,
                    portal_path=list(graph_plan.portal_path),
                    first_portal_id=graph_plan.first_portal_id,
                    reason="durable_identity_not_in_current_frontier_snapshot",
                )
            publish_unavailable = getattr(
                self, "_publish_graph_route_unavailable", None
            )
            if publish_unavailable is not None:
                publish_unavailable(
                    graph_plan,
                    "durable_identity_not_in_current_frontier_snapshot",
                )
            return None
        if getattr(self, "frontier_action_policy", "legacy_scalar") == "event_pareto":
            selected = self._select_frontier_decision_candidate(
                frontier_decision_candidates, ("viewpoint_retry",), request,
            )
        else:
            selected = self._select_score_pool_candidate(
                pools, ("viewpoint_retry",), request,
            )
        if selected is not None:
            return selected

        # A target bearing is a directional information request, not a score
        # bonus. Prefer a legal WorkItem on that half-plane before generic
        # local/adjacent frontier work; absent such evidence this pool is
        # empty and the normal policy below is unchanged.
        if getattr(self, "frontier_action_policy", "legacy_scalar") == "event_pareto":
            selected = self._select_frontier_decision_candidate(
                frontier_decision_candidates, ("target_direction",), request,
            )
        else:
            selected = self._select_score_pool_candidate(
                pools, ("target_direction",), request,
            )
        if selected is not None:
            return selected

        # Once a physical Place has yielded a real observation but no task
        # evidence, expand the topological belief through a certified Portal
        # before spending another local frontier action.  The semantic layer
        # is a finite-state policy; it never bypasses portal admission or
        # creates a Place before a verified crossing.
        semantic_policy = getattr(self, "semantic_place_action", None)
        local_work_pending = self.context_has_pending_local_work(context)
        semantic_action = None
        if semantic_policy is not None:
            try:
                semantic_action = semantic_policy(
                    context.source_place_id,
                    True,
                    local_work_pending,
                )
            except TypeError:
                # Narrow test fixtures and older external callers may expose
                # the two-argument policy facade.
                semantic_action = semantic_policy(
                    context.source_place_id,
                    True,
                )
        if (
            getattr(self, "task_semantic_value_enabled", False)
            and
            semantic_policy is not None
            and context.source_place_id is not None
            and context.source_place_observed
            and semantic_action == "expand_unobserved_portal"
        ):
            self.last_frontier_decision = None
            selected = self._select_context_portal_transition(
                request, context, portal_sources,
            )
            if selected is not None:
                self.publish_status(
                    "semantic_topology_expansion_selected",
                    place_id=int(context.source_place_id),
                    policy="expand_unobserved_portal",
                    route_kind="portal_transition",
                    goal=[round(float(selected[2]), 3), round(float(selected[3]), 3)],
                    portal_gate=(
                        None
                        if len(selected) <= 11 or selected[11] is None
                        else [
                            round(float(selected[11][0]), 3),
                            round(float(selected[11][1]), 3),
                        ]
                    ),
                )
                return selected

        policy_action_tiers = unique_action_tiers(
            context.action_tiers,
            include_probe=True,
        )
        if context.action_tiers[:1] == (ACTION_ADJACENT,):
            # Branch-first is an explicit graph policy. Once the current Place
            # has one observation, prefer an already certified outward edge
            # before spending another action on a local frontier descendant.
            # This changes action legality/order, not a numeric frontier score.
            if getattr(self, "branch_first_enabled", False):
                self.last_frontier_decision = None
                selected = self._select_context_portal_transition(
                    request, context, portal_sources,
                )
                if selected is not None:
                    return selected
            if getattr(self, "frontier_action_policy", "legacy_scalar") == "event_pareto":
                selected = self._select_frontier_decision_candidate(
                    frontier_decision_candidates,
                    (ACTION_ADJACENT, ACTION_LOCAL, ACTION_PROBE),
                    request,
                )
                if selected is not None:
                    return selected
            if getattr(self, "frontier_action_policy", "legacy_scalar") != "event_pareto":
                selected = self._select_score_pool_candidate(
                    pools, (ACTION_ADJACENT,), request,
                )
                if selected is not None:
                    return selected
            # Ordinary ObservationWorkItems own the current Place's local
            # coverage phase.  Finish an executable local item before spending
            # the whole cycle on doorway viewpoints; otherwise a growing set
            # of Portal probes starves room coverage and the Place can never
            # reach its exit phase.
            if local_work_pending:
                if getattr(self, "frontier_action_policy", "legacy_scalar") != "event_pareto":
                    selected = self._select_score_pool_candidate(
                        pools, (ACTION_LOCAL,), request,
                    )
                    if selected is not None:
                        return selected
                selected = self._select_portal_probe_candidate(
                    request, context, probe_candidates,
                )
                if selected is not None:
                    return selected
            self.last_frontier_decision = None
            selected = self._select_context_portal_transition(
                request, context, portal_sources,
            )
            if selected is not None:
                return selected
            selected = self._select_portal_probe_candidate(
                request, context, probe_candidates,
            )
            if selected is not None:
                return selected
            if getattr(self, "frontier_action_policy", "legacy_scalar") != "event_pareto":
                return self._select_score_pool_candidate(
                    pools, (ACTION_PROBE, ACTION_LOCAL), request,
                )
            return None

        if ACTION_PROBE in policy_action_tiers:
            if getattr(self, "frontier_action_policy", "legacy_scalar") == "event_pareto":
                selected = self._select_frontier_decision_candidate(
                    frontier_decision_candidates,
                    policy_action_tiers,
                    request,
                )
                if selected is not None:
                    return selected
            selected = self._select_portal_probe_candidate(
                request, context, probe_candidates,
            )
            if selected is not None:
                return selected
        if getattr(self, "frontier_action_policy", "legacy_scalar") == "event_pareto":
            selected = self._select_frontier_decision_candidate(
                frontier_decision_candidates,
                policy_action_tiers,
                request,
            )
        else:
            selected = self._select_score_pool_candidate(
                pools, policy_action_tiers, request,
            )
        if selected is not None:
            return selected
        self.last_frontier_decision = None
        return self._select_context_portal_transition(
            request, context, portal_sources,
        )

    def _select_graph_plan_candidate(self, candidates, plan, request):
        """Select only a candidate carrying the planner's durable identity."""
        matching = [
            candidate
            for candidate in (candidates or ())
            if candidate_refines_graph_plan(plan, candidate)
        ]
        if not matching:
            return None
        if getattr(self, "frontier_action_policy", "legacy_scalar") == "event_pareto":
            selected_route = self._select_frontier_decision_candidate(
                matching, FRONTIER_ACTION_ORDER, request,
            )
            if selected_route is None:
                return None
            self.last_graph_selected_candidate = next(
                (
                    candidate for candidate in matching
                    if getattr(candidate, "route", None) == selected_route
                ),
                None,
            )
            if getattr(plan, "action", None) == "reinspect_target":
                selected_route = with_graph_action(
                    selected_route,
                    plan.action,
                    getattr(plan, "reason", None),
                )
            return selected_route
        # The legacy-rank ablation keeps its scalar tie-break, but it still
        # obeys the graph identity contract.  Only exact-identity routes are
        # in this pool, so this is not a second policy over unrelated actions.
        selected = max(
            matching,
            key=lambda candidate: (
                float(candidate.route[7]) if len(candidate.route) > 7 else -float("inf"),
                tuple(
                    getattr(candidate, "tie_break_key", None)
                    or (getattr(candidate, "candidate_id", ()),)
                ),
            ),
        )
        self.last_graph_selected_candidate = selected
        route = selected.route
        self.last_frontier_region_tier = selected.region_tier
        self.last_selected_place_hops = route[9] if len(route) > 9 else None
        self.last_selected_viewpoint_retry = bool(
            len(route) > 16 and route[16]
        )
        # The route tuple is the compatibility boundary consumed by
        # activation.  A target reinspection may reuse a normal local
        # WorkItem viewpoint, but its durable graph action must remain visible
        # after selection; otherwise the route is logged and lifecycle-owned as
        # ordinary local work even though the target lease owns it.
        if getattr(plan, "action", None) == "reinspect_target":
            route = with_graph_action(
                route,
                plan.action,
                getattr(plan, "reason", None),
            )
        return route

    def _select_frontier_decision_candidate(
        self, candidates, allowed_categories, request,
    ):
        """Select one legal local action without reading the scalar score."""
        if not candidates:
            return None
        decision = select_frontier_action(
            candidates,
            allowed_categories=tuple(allowed_categories or FRONTIER_ACTION_ORDER),
            allowed_region_tiers=request.allowed_region_tiers,
            decision_epoch=(
                getattr(request.components, "epoch", None),
                getattr(self, "current_physical_place_id", None),
                getattr(self, "last_work_item_created", None),
                getattr(self, "last_portal_hypothesis_id", None),
            ),
        )
        self.last_frontier_decision = {
            "policy": "event_pareto",
            "category": decision.category,
            "candidate_count": len(decision.feasible) + len(decision.rejected),
            "feasible_count": len(decision.feasible),
            "pareto_front_count": len(decision.pareto_front),
            "selected_candidate_id": (
                None if decision.selected is None else decision.selected.candidate_id
            ),
            "decision_epoch": decision.decision_epoch,
        }
        if decision.selected is None:
            return None
        selected = decision.selected.route
        self.last_frontier_region_tier = decision.selected.region_tier
        self.last_selected_place_hops = selected[9] if len(selected) > 9 else None
        self.last_selected_viewpoint_retry = bool(
            len(selected) > 16 and selected[16]
        )
        publish = getattr(self, "publish_status", None)
        if publish is not None:
            publish(
                "frontier_action_selected",
                policy="event_pareto",
                category=decision.category,
                candidate_count=self.last_frontier_decision["candidate_count"],
                feasible_count=self.last_frontier_decision["feasible_count"],
                pareto_front_count=self.last_frontier_decision["pareto_front_count"],
                candidate_id=decision.selected.candidate_id,
                decision_epoch=decision.decision_epoch,
            )
        return selected

    def _select_portal_probe_candidate(
        self, request, context, probe_candidates,
    ):
        """Choose one full-method probe before legacy score-pool fallback."""
        if not probe_candidates or not getattr(
            self, "task_semantic_value_enabled", False
        ):
            return None
        ledger = getattr(self, "portal_probe_ledger", None)
        rejected = getattr(self, "frontier_is_rejected", None)
        selected, result = select_probe_route(
            probe_candidates,
            request,
            context,
            ledger=ledger,
            allowed_region_tiers=request.allowed_region_tiers,
            route_rejected=(
                None
                if rejected is None
                else lambda x, y: rejected(x, y, request.now)
            ),
        )
        report = selection_report(result)
        self.last_portal_probe_value_selection = report
        if selected is None:
            return None
        probe = selected[15]
        publish_status = getattr(self, "publish_status", None)
        if publish_status is not None:
            publish_status(
                "portal_probe_value_selected",
                action_category=report["action_category"],
                candidate_count=report["candidate_count"],
                feasible_count=report["feasible_count"],
                pareto_front_count=report["pareto_front_count"],
                probe_id=getattr(probe, "probe_id", None),
                goal=[round(float(selected[2]), 3), round(float(selected[3]), 3)],
            )
        self.last_frontier_region_tier = next(
            (
                pool_name
                for route, pool_name, _category in probe_candidates
                if route is selected
            ),
            "revisit",
        )
        self.last_selected_place_hops = selected[9]
        self.last_selected_viewpoint_retry = bool(
            len(selected) > 16 and selected[16]
        )
        return selected
