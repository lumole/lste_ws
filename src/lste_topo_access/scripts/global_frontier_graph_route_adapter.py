"""ROS-facing adapter for the pure durable graph route planner.

This boundary translates a discrete graph intent into the existing portal
selection/egress helpers.  It never computes geometry or publishes a velocity
command; Navfn and TEB remain the only motion authorities.
"""

from global_frontier_graph_route_planner import (
    ACTION_CROSS_PORTAL,
    ACTION_PROBE_PORTAL,
    PLAN_READY,
)
from global_frontier_graph_route_transaction import (
    GraphRouteActionTransaction,
    candidate_action,
    materialize_graph_route_action,
)


class GlobalFrontierGraphRouteAdapterMixin:
    """Materialize only the first edge of a durable graph route."""

    @staticmethod
    def _graph_snapshot_map_epoch(snapshot):
        """Read the immutable map epoch used by graph negative evidence."""
        for value in (
            getattr(
                getattr(getattr(snapshot, "map_context", None), "components", None),
                "epoch",
                None,
            ),
            getattr(
                getattr(getattr(snapshot, "route_graph", None), "validation", None),
                "epoch",
                None,
            ),
        ):
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        return None

    @staticmethod
    def _positive_ids(values):
        """Normalize a snapshot candidate-ID collection for lease checks."""
        if values is None:
            return None
        if isinstance(values, dict):
            values = values.keys()
        try:
            iterator = iter(values)
        except TypeError:
            iterator = iter((values,))
        result = set()
        for value in iterator:
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                result.add(value)
        return result

    def _graph_plan_can_reuse_with_candidates(
        self, plan, visible_work_item_ids=None, visible_probe_ids=None,
    ):
        """Keep a lease only while its identity remains in this snapshot.

        A route lease protects an obligation from transient SLAM churn.  It
        must not freeze a WorkItem that candidate collection has shown cannot
        be materialized while another same-place identity is executable.  An
        empty candidate set is deliberately inconclusive and keeps the lease
        alive for the next map event.
        """
        if plan is None:
            return False
        action = str(getattr(plan, "action", "") or "")
        if not self._graph_plan_probe_phase_matches_durable_state(plan):
            return False
        if action in {"bootstrap_observation", "observe_local_work", "retry_viewpoint"}:
            ids = self._positive_ids(visible_work_item_ids)
            obligation = self._positive_ids((getattr(plan, "obligation_id", None),))
            obligation = next(iter(obligation), None)
        elif action == "probe_portal":
            ids = self._positive_ids(visible_probe_ids)
            obligation = self._positive_ids((getattr(plan, "obligation_id", None),))
            obligation = next(iter(obligation), None)
        else:
            return True
        if ids is None or not ids:
            return True
        return obligation is not None and obligation in ids

    def _graph_plan_probe_phase_matches_durable_state(self, plan):
        """Reject a probe lease after its evidence transaction changes phase.

        A source-side probe finishing is a durable state transition, not map
        churn.  Reusing the old source plan after that transition makes the
        materializer construct a destination candidate against a source
        contract.  The resulting mismatch is avoidable and can cause an
        unnecessary controller handoff.  Keep legacy plans/ledgers without
        phase provenance permissive; production records carry the state needed
        to make the boundary explicit.
        """
        if plan is None or str(getattr(plan, "action", "") or "") != "probe_portal":
            return True
        if str(getattr(plan, "obligation_kind", "") or "") != "portal_probe":
            return True
        expected = str(getattr(plan, "portal_probe_phase", "") or "").strip().lower()
        if not expected:
            return True
        ledger = getattr(self, "portal_probe_ledger", None)
        get_record = getattr(ledger, "get", None)
        if not callable(get_record):
            return True
        try:
            record = get_record(getattr(plan, "obligation_id", None))
        except (TypeError, ValueError):
            return True
        if not isinstance(record, dict):
            return True
        state = str(record.get("state", "") or "").strip().lower()
        if expected == "source":
            # ``active`` is the source-side execution state.  A destination
            # attempt has its own explicit ``destination_active`` state.
            return state in {"pending", "active"}
        if expected == "destination":
            return state in {"source_arrived", "destination_active"}
        return True

    def _publish_graph_route_unavailable(self, plan, reason):
        """Publish a controller handoff when a ready graph route has no edge.

        ``graph_route_plan_lease_active`` protects the durable obligation from
        snapshot churn, but it does not own the already-running MoveBase
        action.  Once a ready plan has been checked against the current
        snapshot and no candidate can materialize, the controller must receive
        an explicit release record.  Keep the selected route identity in that
        record so a bridge cannot infer a substitute from the goal coordinates.
        """
        if plan is None or getattr(plan, "status", None) != PLAN_READY:
            return False
        # A pre-admitted successor is still a controller handoff path.  This
        # helper is reserved for the stronger case where the graph edge has no
        # materialized successor at all; let the normal terminal promotion
        # consume the cached successor instead.
        if getattr(self, "prefetched_frontier", None) is not None:
            return False
        try:
            route_id = max(0, int(getattr(self, "active_route_id", 0) or 0))
        except (TypeError, ValueError):
            route_id = 0
        if route_id <= 0:
            # There is no controller lease to release during startup or after
            # a mission has explicitly stopped. Keep this a planning-only
            # diagnostic in that state.
            return False
        signature = (
            plan.signature(),
            route_id,
            str(reason),
        )
        if signature == getattr(
            self, "last_graph_route_unavailable_signature", None
        ):
            return False
        self.last_graph_route_unavailable_signature = signature
        publish = getattr(self, "publish_status", None)
        if publish is None:
            return False
        publish(
            "frontier_route_unavailable",
            route_id=route_id,
            map_epoch=(
                None
                if getattr(getattr(self, "graph_route_action_transaction", None), "map_epoch", None)
                is None
                else getattr(self.graph_route_action_transaction, "map_epoch")
            ),
            successor_route_id=0,
            controller_lease="release",
            route_unavailable=True,
            action=str(plan.action),
            obligation_kind=str(plan.obligation_kind),
            obligation_id=plan.obligation_id,
            portal_path=list(plan.portal_path),
            first_portal_id=plan.first_portal_id,
            target_place_id=plan.target_place_id,
            reason=str(reason),
        )
        return True

    def _retain_graph_route_plan_lease(self, plan, reason="materialization_wait"):
        """Keep one ready durable action across transient map snapshots.

        A missing frontier projection is not evidence that the durable
        obligation changed.  The lease makes that distinction explicit and
        prevents the next planning cycle from selecting a different obligation
        merely because its cells happen to be visible first.
        """
        if plan is None or getattr(plan, "status", None) != PLAN_READY:
            return False
        self.graph_route_plan_lease_active = True
        self.graph_route_plan_lease_signature = plan.signature()
        return True

    def _clear_graph_route_plan_lease(self, reason=""):
        """Release the durable action lease at an explicit lifecycle boundary."""
        del reason
        self.graph_route_plan_lease_active = False
        self.graph_route_plan_lease_signature = None

    def _graph_route_plan_lease_matches_current(self, plan):
        """Reject a stale lease after a physical Place transition."""
        if plan is None:
            return False
        current = getattr(self, "current_physical_place_id", None)
        planned = getattr(plan, "current_place_id", None)
        if current is None or planned is None:
            return True
        try:
            return int(current) == int(planned)
        except (TypeError, ValueError):
            return False

    def _install_graph_route_preference(self, plan):
        """Install the selected identity without creating a new transaction."""
        self.graph_route_portal_id = None
        self.graph_route_probe_id = None
        if plan is None or plan.status != PLAN_READY:
            return
        first_id = plan.first_portal_id
        if first_id is None:
            return
        try:
            first_id = int(first_id)
        except (TypeError, ValueError):
            return
        if plan.action == ACTION_PROBE_PORTAL and plan.obligation_kind == "portal_probe":
            self.graph_route_probe_id = first_id
        else:
            self.graph_route_portal_id = first_id

    def _set_graph_route_preference(
        self, map_epoch=None, visible_work_item_ids=None,
        visible_probe_ids=None, stage_only=False,
    ):
        """Replan after ledger reconciliation and install one edge identity.

        Selection calls this hook only after the current frontier snapshot has
        reconciled its WorkItems and Portal probes.  Keeping the preference
        installation separate from geometry materialization gives the slow
        graph layer a consistent view of the durable ledgers.
        """
        plan = self._plan_durable_graph_route(
            map_epoch=map_epoch,
            visible_work_item_ids=visible_work_item_ids,
            visible_probe_ids=visible_probe_ids,
            commit=not stage_only,
        )
        if stage_only:
            # Do not let events emitted during candidate collection inherit a
            # completed route's stale graph plan.  The staged identity lives
            # only in the preference fields until the final pass commits it.
            self.last_graph_route_plan = None
            self.last_graph_route_plan_signature = None
            self.graph_route_action_transaction = None
        self._install_graph_route_preference(plan)
        return plan

    def _plan_durable_graph_route(
        self, map_epoch=None, visible_work_item_ids=None,
        visible_probe_ids=None, commit=True,
    ):
        if not getattr(self, "graph_route_planner_enabled", False):
            self.last_graph_route_plan = None
            self.graph_route_portal_id = None
            self.graph_route_probe_id = None
            self.graph_route_action_transaction = None
            return None
        planner = getattr(self, "graph_route_planner", None)
        current_place_id = getattr(self, "current_physical_place_id", None)
        if planner is None or current_place_id is None:
            return None
        work_ledger = getattr(self, "place_work_items", None)
        probe_ledger = getattr(self, "portal_probe_ledger", None)
        portal_ledger = getattr(self, "portal_hypothesis_ledger", None)
        target_work = getattr(self, "target_observation_work", None)
        if visible_work_item_ids is None:
            context = getattr(self, "last_selection_context", None)
            visible_work_item_ids = (
                None
                if context is None
                else getattr(context, "work_item_details", None)
            )
        target_work_pending = bool(
            target_work is not None
            and callable(getattr(target_work, "has_pending", None))
            and target_work.has_pending(current_place_id)
        )
        # The reinspection event is authoritative even while the target
        # WorkItem is still crossing the asynchronous GoalManager/global-
        # frontier boundary. Once bound, the ledger independently preserves
        # the same obligation across detector gaps.
        target_pending = bool(
            target_work_pending
            or getattr(self, "target_reinspection_pending", False)
        )
        plan = planner.plan(
            current_place_id,
            places=getattr(getattr(self, "region_memory", None), "regions", ()),
            portals=portal_ledger,
            work_items=work_ledger,
            probes=probe_ledger,
            branch_coverage=getattr(
                self, "directional_branch_coverage", None
            ),
            visible_work_item_ids=visible_work_item_ids,
            visible_probe_ids=visible_probe_ids,
            branch_first=bool(getattr(self, "branch_first_enabled", False)),
            target_observation_pending=target_pending,
            structural_boundary_first=bool(
                getattr(self, "branch_first_enabled", False)
            ),
            map_epoch=map_epoch,
        )
        if not commit:
            # This is the read-only first phase of snapshot planning.  It
            # supplies the candidate collector with one durable identity
            # preference, but leaves the previous committed plan/transaction
            # untouched until all current-snapshot candidates are reconciled.
            return plan
        self.last_graph_route_plan = plan
        transaction_id = int(
            getattr(self, "graph_route_action_transaction_sequence", 0)
        ) + 1
        self.graph_route_action_transaction_sequence = transaction_id
        self.graph_route_action_transaction = GraphRouteActionTransaction(
            transaction_id=transaction_id,
            plan=plan,
            map_epoch=map_epoch,
        )
        signature = plan.signature()
        if signature != getattr(self, "last_graph_route_plan_signature", None):
            self.last_graph_route_plan_signature = signature
            publish = getattr(self, "publish_status", None)
            if publish is not None:
                publish(
                    "graph_route_plan_selected",
                    **plan.as_dict(),
                    transaction_id=transaction_id,
                    map_epoch=map_epoch,
                )
        return plan

    def _prepare_graph_route_action(
        self, snapshot, *, visible_work_item_ids=None, visible_probe_ids=None,
        reuse_existing_plan=False, stage_only=False,
    ):
        """Reserve a planner-selected edge for the next selection pass.

        The returned candidate is only used for a durable reverse egress whose
        current map no longer exposes a normal frontier.  Forward edges and
        source-side probes retain a preferred identity and are materialized by
        the ordinary portal selector on the same snapshot.
        """
        map_epoch = getattr(
            getattr(getattr(snapshot, "map_context", None), "components", None),
            "epoch",
            None,
        )
        if map_epoch is None:
            map_epoch = getattr(
                getattr(getattr(snapshot, "route_graph", None), "validation", None),
                "epoch",
                None,
            )
        existing = getattr(self, "last_graph_route_plan", None)
        lease_identity_is_present = self._graph_plan_can_reuse_with_candidates(
            existing,
            visible_work_item_ids=visible_work_item_ids,
            visible_probe_ids=visible_probe_ids,
        )
        if (
            reuse_existing_plan
            and existing is not None
            and lease_identity_is_present
            # A cleared lease does not make a plan current.  Portal arrival
            # clears the old lease exactly when physical Place ownership
            # changes; reusing that plan would dispatch reverse transit from
            # the previous Place before its new local ledger is reconciled.
            and self._graph_route_plan_lease_matches_current(existing)
        ):
            plan = existing
            self._install_graph_route_preference(plan)
        else:
            if reuse_existing_plan:
                reason = (
                    "stale_or_missing_plan"
                    if lease_identity_is_present
                    else "leased_obligation_not_in_current_candidates"
                )
                self._clear_graph_route_plan_lease(reason)
            plan = self._set_graph_route_preference(
                map_epoch=map_epoch,
                visible_work_item_ids=visible_work_item_ids,
                visible_probe_ids=visible_probe_ids,
                stage_only=stage_only,
            )
        # Materialization can discover that the selected probe has no current
        # map projection and park it in the durable ledger.  That is a
        # structural change in obligation readiness: retaining the old ready
        # plan would make the lease wait forever.  Replan once so another
        # executable obligation may proceed, while the parked probe remains in
        # the completion ledger for a future physical observation.
        if (
            plan is not None
            and getattr(plan, "action", None) == ACTION_PROBE_PORTAL
            and getattr(plan, "obligation_kind", None) == "portal_probe"
        ):
            probe_ledger = getattr(self, "portal_probe_ledger", None)
            probe = (
                None
                if probe_ledger is None
                else probe_ledger.get(getattr(plan, "obligation_id", None))
            )
            if isinstance(probe, dict) and str(
                probe.get("state", "")
            ).strip().lower() == "awaiting_projection":
                self._clear_graph_route_plan_lease("probe_projection_parked")
                plan = self._set_graph_route_preference(
                    map_epoch=map_epoch,
                    visible_work_item_ids=visible_work_item_ids,
                    visible_probe_ids=visible_probe_ids,
                )
        if plan is None or plan.status != PLAN_READY:
            self._clear_graph_route_plan_lease("plan_not_ready")
            return plan, None
        # ``None`` is the deliberate dry-run used after frontier reconciliation
        # and before score-pool selection.  It installs the graph identity but
        # does not attempt to consume a map snapshot twice.
        if snapshot is None:
            return plan, None
        first_id = plan.first_portal_id
        if first_id is None:
            return plan, None
        try:
            first_id = int(first_id)
        except (TypeError, ValueError):
            return plan, None

        if plan.action == ACTION_PROBE_PORTAL and plan.obligation_kind == "portal_probe":
            probe_ledger = getattr(self, "portal_probe_ledger", None)
            probe = (
                None
                if probe_ledger is None
                else probe_ledger.get(first_id)
            )
            if probe is not None:
                self.graph_route_probe_id = first_id
                select_probe = getattr(self, "select_durable_portal_probe", None)
                candidate = None if select_probe is None else select_probe(snapshot)
                if candidate is not None:
                    if not self._commit_graph_route_candidate(
                        plan, candidate, source="durable_probe"
                    ):
                        return plan, None
                    self.graph_route_probe_id = None
                    self._clear_graph_route_plan_lease("probe_candidate_committed")
                    self._publish_graph_route_materialized(
                        self.last_graph_route_plan, candidate
                    )
                    return plan, candidate
                return plan, None

        # An unbound hypothesis or a crossed edge is selected by the portal
        # layer.  A certified unbound edge may be materialized directly from
        # physical Portal memory even when its frontier projection vanished;
        # reverse traversal remains owned by durable egress.
        self.graph_route_portal_id = first_id
        record = getattr(
            getattr(self, "portal_hypothesis_ledger", None), "get", lambda _id: None
        )(first_id)
        current = getattr(self, "current_physical_place_id", None)
        reverse = bool(
            isinstance(record, dict)
            and current is not None
            and record.get("destination_place_id") is not None
            and int(record.get("destination_place_id")) == int(current)
            and int(record.get("source_place_id", -1)) != int(current)
        )
        if reverse:
            select_egress = getattr(self, "select_durable_portal_egress", None)
            candidate = None if select_egress is None else select_egress(snapshot)
            if candidate is not None:
                if not self._commit_graph_route_candidate(
                    plan, candidate, source="durable_egress"
                ):
                    return plan, None
                self.graph_route_portal_id = None
                self._clear_graph_route_plan_lease("egress_candidate_committed")
                self._publish_graph_route_materialized(
                    self.last_graph_route_plan, candidate
                )
                return plan, candidate
        if (
            isinstance(record, dict)
            and record.get("destination_place_id") is None
            and str(record.get("state", "")).strip().lower()
            in ("certified", "selected")
        ):
            select_crossing = getattr(
                self, "select_durable_portal_crossing", None,
            )
            candidate = (
                None
                if select_crossing is None
                else select_crossing(snapshot)
            )
            if candidate is not None:
                if not self._commit_graph_route_candidate(
                    plan, candidate, source="durable_crossing"
                ):
                    return plan, None
                self.graph_route_portal_id = None
                self._clear_graph_route_plan_lease("crossing_candidate_committed")
                self._publish_graph_route_materialized(
                    self.last_graph_route_plan, candidate
                )
                return plan, candidate
            # The crossing materializer can prove that this physical Portal
            # cannot be represented in the current map/costmap snapshot. The
            # egress helper records that epoch-scoped negative evidence. Do
            # not return the stale ready plan to the caller, which would make
            # every planning wake retain the same lease indefinitely.
            unavailable = getattr(
                self, "last_durable_portal_crossing_unavailable", None
            )
            unavailable_epoch = (
                None if not isinstance(unavailable, dict)
                else unavailable.get("map_epoch")
            )
            current_epoch = self._graph_snapshot_map_epoch(snapshot)
            try:
                same_epoch = (
                    unavailable is not None
                    and int(unavailable.get("portal_id")) == int(first_id)
                    and unavailable_epoch is not None
                    and current_epoch is not None
                    and int(unavailable_epoch) == int(current_epoch)
                )
            except (TypeError, ValueError):
                same_epoch = (
                    isinstance(unavailable, dict)
                    and str(unavailable.get("portal_id")) == str(first_id)
                    and str(unavailable_epoch) == str(current_epoch)
                )
            if same_epoch:
                reason = str(
                    unavailable.get("reason") or
                    "portal_materialization_unavailable"
                )
                # Keep the state transition at the graph/action boundary as
                # well as in the geometry helper. Test adapters and older
                # materializers may only report the reason; marking here
                # guarantees that the immediate replan cannot select the same
                # Portal again on this map epoch.
                mark_rejected = getattr(
                    getattr(self, "portal_hypothesis_ledger", None),
                    "mark_unbound_rejected",
                    None,
                )
                if callable(mark_rejected):
                    mark_rejected(first_id, current_epoch, reason)
                self.graph_route_portal_id = None
                self._clear_graph_route_plan_lease(reason)
                replanned = self._set_graph_route_preference(
                    map_epoch=current_epoch,
                    visible_work_item_ids=visible_work_item_ids,
                    visible_probe_ids=visible_probe_ids,
                )
                self.last_durable_portal_crossing_unavailable = None
                publish = getattr(self, "publish_status", None)
                if publish is not None:
                    publish(
                        "graph_route_materialization_replanned",
                        previous_portal_id=first_id,
                        map_epoch=current_epoch,
                        action=(
                            None if replanned is None
                            else str(replanned.action)
                        ),
                        obligation_id=(
                            None if replanned is None
                            else replanned.obligation_id
                        ),
                        reason=reason,
                    )
                return replanned, None
        return plan, None

    def _commit_graph_route_candidate(self, plan, candidate, *, source="selector"):
        """Commit one candidate through the single graph-action boundary."""
        transaction = getattr(self, "graph_route_action_transaction", None)
        if transaction is None:
            return True
        validation_candidate = candidate
        # Older injected adapters returned a short tuple from the durable
        # probe/egress helpers.  Preserve that public tuple while supplying
        # the graph action to the transaction validator; production routes
        # already carry the explicit action fields.
        if candidate_action(candidate) is None and str(source).startswith("durable"):
            route = list(tuple(candidate or ()))
            while len(route) <= 18:
                route.append(None)
            route[17] = str(transaction.plan.action)
            route[18] = "legacy_durable_candidate"
            validation_candidate = tuple(route)
        elif candidate_action(candidate) == "cross_portal" and len(
            tuple(candidate or ())
        ) <= 19:
            # The live portal selector records the chosen durable ID even for
            # older route tuples that predate the trailing identity field.
            # Supply that identity to validation without changing the public
            # tuple returned to activation.
            portal_id = getattr(self, "last_portal_hypothesis_id", None)
            try:
                portal_id = int(portal_id)
            except (TypeError, ValueError):
                portal_id = None
            if portal_id is not None and portal_id > 0:
                route = list(tuple(candidate))
                while len(route) <= 18:
                    route.append(None)
                route.append(portal_id)
                validation_candidate = tuple(route)
        materialization = materialize_graph_route_action(
            plan,
            validation_candidate,
            branch_first=bool(getattr(self, "branch_first_enabled", False)),
        )
        if not materialization.accepted:
            publish = getattr(self, "publish_status", None)
            if publish is not None:
                publish(
                    "graph_route_plan_mismatch",
                    transaction_id=int(transaction.transaction_id),
                    map_epoch=transaction.map_epoch,
                    source=str(source),
                    planned_action=str(plan.action),
                    candidate_action=str(materialization.candidate_action),
                    planned_first_portal_id=plan.first_portal_id,
                    candidate_portal_id=materialization.candidate_portal_id,
                    planned_obligation_id=plan.obligation_id,
                    candidate_work_item_id=materialization.candidate_work_item_id,
                    reason=str(materialization.reason),
                )
            return False

        self.last_graph_route_plan = materialization.plan
        self.last_graph_route_plan_signature = materialization.plan.signature()
        self.graph_route_action_transaction = transaction.commit(materialization)
        self._clear_graph_route_plan_lease("candidate_committed")
        publish = getattr(self, "publish_status", None)
        if publish is not None:
            if materialization.reconciled:
                publish(
                    "graph_route_plan_reconciled",
                    transaction_id=int(transaction.transaction_id),
                    map_epoch=transaction.map_epoch,
                    source=str(source),
                    planned_action=str(plan.action),
                    committed_action=str(materialization.plan.action),
                    planned_obligation_id=plan.obligation_id,
                    committed_obligation_id=materialization.plan.obligation_id,
                    planned_first_portal_id=plan.first_portal_id,
                    committed_first_portal_id=materialization.plan.first_portal_id,
                    reason=str(materialization.reason),
                )
            publish(
                "graph_route_action_committed",
                transaction_id=int(transaction.transaction_id),
                map_epoch=transaction.map_epoch,
                source=str(source),
                action=str(materialization.plan.action),
                obligation_id=materialization.plan.obligation_id,
                portal_id=materialization.candidate_portal_id,
                reconciled=bool(materialization.reconciled),
                reason=str(materialization.reason),
            )
        return True

    def _publish_graph_route_materialized(self, plan, candidate):
        publish = getattr(self, "publish_status", None)
        if publish is None:
            return
        publish(
            "graph_route_edge_materialized",
            transaction_id=(
                None
                if getattr(self, "graph_route_action_transaction", None) is None
                else int(self.graph_route_action_transaction.transaction_id)
            ),
            map_epoch=(
                None
                if getattr(self, "graph_route_action_transaction", None) is None
                else self.graph_route_action_transaction.map_epoch
            ),
            status=plan.status,
            action=plan.action,
            target_place_id=plan.target_place_id,
            obligation_kind=plan.obligation_kind,
            obligation_id=plan.obligation_id,
            portal_path=list(plan.portal_path),
            first_portal_id=plan.first_portal_id,
            portal_probe_phase=str(
                getattr(plan, "portal_probe_phase", "") or ""
            ),
            route_kind=(candidate[10] if len(candidate) > 10 else None),
            goal=[round(float(candidate[2]), 3), round(float(candidate[3]), 3)],
        )


__all__ = ["GlobalFrontierGraphRouteAdapterMixin"]
