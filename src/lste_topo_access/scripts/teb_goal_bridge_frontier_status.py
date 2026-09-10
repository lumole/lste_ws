"""Frontier lifecycle observations for the TEB action bridge."""

import json
import math

import rospy


class TebGoalBridgeFrontierStatusMixin:
    def _clear_frontier_intent_after_release_locked(self):
        """Prevent the bridge timer from replaying a released frontier goal."""
        self.latest_goal = None
        self.last_dispatched_goal = None
        self.last_dispatch_identity = None
        self.last_terminal_goal = None
        self.latest_intent_source = "waiting_global_slam_frontier"
        self.latest_intent_priority = 0
        self.latest_route_kind = ""
        self.latest_mission_route_kind = ""
        self.latest_graph_transaction_id = 0
        self.latest_route_id = 0
        self.latest_intent_goal = None

    def _apply_controller_lease_release_locked(self, payload):
        """Apply an identity-checked terminal release from GlobalFrontier."""
        released = payload.get("released_controller_route")
        released = released if isinstance(released, dict) else {}
        if not bool(
            payload.get("terminal_received")
            or released.get("terminal_received")
        ) or not bool(
            payload.get("controller_pending")
            or released.get("controller_pending")
        ):
            return False
        try:
            route_id = max(
                0,
                int(
                    released.get(
                        "route_id",
                        payload.get(
                            "released_route_id", payload.get("route_id", 0)
                        ),
                    )
                    or 0
                ),
            )
        except (TypeError, ValueError):
            route_id = 0
        if route_id <= 0:
            return False
        if int(getattr(self, "active_route_id", 0) or 0) != route_id:
            # A newer route already owns the action. The monotonic route guard
            # below is the final protection against cancelling that successor.
            return False
        if str(
            getattr(self, "active_intent_source", "unknown") or "unknown"
        ).strip().lower() != "global_slam_frontier":
            return False
        release = getattr(
            self, "_release_frontier_controller_lease_locked", None
        )
        if not callable(release):
            return False
        pending_prepare = getattr(self, "pending_terminal_prepare", None)
        expected_contract = {}
        if isinstance(pending_prepare, dict):
            expected_contract = dict(
                pending_prepare.get("old_contract") or {}
            )
        if not expected_contract:
            active_contract = getattr(self, "active_action_contract", None)
            if isinstance(active_contract, dict):
                expected_contract = dict(active_contract)
        if not expected_contract:
            last_dispatch = getattr(self, "last_dispatch_identity", None)
            if isinstance(last_dispatch, dict):
                expected_contract = dict(last_dispatch)

        def contract_int(*values):
            for value in values:
                try:
                    return int(value or 0)
                except (TypeError, ValueError):
                    continue
            return 0

        incoming_lifecycle = contract_int(
            released.get("lifecycle_transaction_id"),
            payload.get("lifecycle_transaction_id"),
        )
        incoming_generation = contract_int(
            released.get("action_generation"),
            payload.get("action_generation"),
        )
        incoming_graph = contract_int(
            released.get("graph_transaction_id"),
            payload.get("graph_transaction_id"),
        )
        incoming_map = released.get("map_epoch")
        if incoming_map is None:
            incoming_map = payload.get("map_epoch")
        expected_generation = contract_int(
            expected_contract.get(
                "generation", expected_contract.get("action_generation", 0)
            )
        )
        expected_lifecycle = contract_int(
            expected_contract.get("lifecycle_transaction_id")
        )
        expected_graph = contract_int(expected_contract.get("graph_transaction_id"))
        expected_map = expected_contract.get("map_epoch")
        if expected_generation > 0 and incoming_generation != expected_generation:
            self.publish_bridge_status(
                "controller_lease_release_rejected",
                reason="action_generation_mismatch",
                route_id=route_id,
                expected_action_generation=expected_generation,
                received_action_generation=incoming_generation,
            )
            return False
        if expected_lifecycle > 0 and incoming_lifecycle != expected_lifecycle:
            self.publish_bridge_status(
                "controller_lease_release_rejected",
                reason="lifecycle_transaction_id_mismatch",
                route_id=route_id,
                expected_lifecycle_transaction_id=expected_lifecycle,
                received_lifecycle_transaction_id=incoming_lifecycle,
            )
            return False
        if expected_graph > 0 and incoming_graph != expected_graph:
            self.publish_bridge_status(
                "controller_lease_release_rejected",
                reason="graph_transaction_id_mismatch",
                route_id=route_id,
                expected_graph_transaction_id=expected_graph,
                received_graph_transaction_id=incoming_graph,
            )
            return False
        if expected_map is not None:
            try:
                if int(incoming_map) != int(expected_map):
                    raise ValueError
            except (TypeError, ValueError):
                self.publish_bridge_status(
                    "controller_lease_release_rejected",
                    reason="map_epoch_mismatch",
                    route_id=route_id,
                    expected_map_epoch=expected_map,
                    received_map_epoch=incoming_map,
                )
                return False
        released_route_kind = str(
            released.get("route_kind")
            or payload.get("released_route_kind")
            or getattr(self, "active_route_kind", "")
            or ""
        )
        release_reason = str(
            payload.get("release_reason", "controller_lease_released")
            or "controller_lease_released"
        )
        if (
            release_reason != "terminal_boundary"
            and not release_reason.startswith(
                ("terminal_boundary_", "endpoint_terminal_")
            )
        ):
            release_reason = "terminal_boundary_%s" % release_reason
        if not release(route_id, release_reason):
            return False
        self._clear_frontier_intent_after_release_locked()
        self.publish_bridge_status(
            "controller_lease_release_applied",
            released_route_id=route_id,
            released_route_kind=released_route_kind,
            terminal_received=True,
            controller_lease="released",
            next_owner="global_slam_frontier",
            lifecycle_transaction_id=incoming_lifecycle,
            action_generation=incoming_generation,
            new_action_generation=int(
                getattr(self, "action_generation", 0) or 0
            ),
            graph_transaction_id=incoming_graph,
            map_epoch=incoming_map,
        )
        self.publish_bridge_status(
            "lease_release_ack",
            released_route_id=route_id,
            released_route_kind=released_route_kind,
            lifecycle_transaction_id=incoming_lifecycle,
            action_generation=incoming_generation,
            new_action_generation=int(
                getattr(self, "action_generation", 0) or 0
            ),
            graph_transaction_id=incoming_graph,
            map_epoch=incoming_map,
            controller_lease="released",
            released_controller_route={
                "route_id": route_id,
                "route_kind": released_route_kind,
                "lifecycle_transaction_id": incoming_lifecycle,
                "action_generation": incoming_generation,
                "graph_transaction_id": incoming_graph,
                "map_epoch": incoming_map,
            },
        )
        # Keep the exact predecessor ACK local to the bridge.  The next
        # dispatch may consume it only for a newer frontier route; a target or
        # same-route replay must never be labeled as a successor.
        self.pending_successor_release_ack = {
            "route_id": route_id,
            "route_kind": released_route_kind,
            "lifecycle_transaction_id": incoming_lifecycle,
            "action_generation": incoming_generation,
            "new_action_generation": int(
                getattr(self, "action_generation", 0) or 0
            ),
            "graph_transaction_id": incoming_graph,
            "map_epoch": incoming_map,
        }
        return True

    def _remember_frontier_map_epoch_locked(self, payload):
        """Bind the newest frontier route to its graph snapshot epoch."""
        raw_route_id = payload.get("route_id")
        try:
            route_id = max(0, int(raw_route_id or 0))
        except (TypeError, ValueError):
            route_id = 0
        if route_id <= 0:
            return
        raw_epoch = payload.get("map_epoch")
        try:
            epoch = None if raw_epoch is None else int(raw_epoch)
        except (TypeError, ValueError):
            epoch = None
        previous_route_id = int(
            getattr(self, "latest_frontier_map_route_id", 0) or 0
        )
        try:
            graph_transaction_id = max(
                0, int(payload.get("graph_transaction_id", 0) or 0)
            )
        except (TypeError, ValueError):
            graph_transaction_id = 0
        if route_id > previous_route_id:
            self.latest_frontier_map_route_id = route_id
            self.latest_frontier_map_epoch = epoch
            self.latest_frontier_graph_transaction_id = graph_transaction_id
        elif route_id == previous_route_id and epoch is not None:
            self.latest_frontier_map_epoch = epoch
            if graph_transaction_id > 0:
                self.latest_frontier_graph_transaction_id = graph_transaction_id
        if epoch is not None and (
            not bool(getattr(self, "intent_seen", False))
            or str(getattr(self, "latest_intent_source", "") or "").strip().lower()
            == "global_slam_frontier"
            or route_id >= int(getattr(self, "latest_route_id", 0) or 0)
        ):
            self.latest_route_map_epoch = epoch
            if graph_transaction_id > 0:
                self.latest_graph_transaction_id = graph_transaction_id

    def _handle_frontier_route_unavailable_locked(self, payload):
        """Release an idle route, or defer while its controller still runs.

        ``frontier_route_unavailable`` describes a planning snapshot, not a
        physical failure of the currently executing route.  A durable graph
        obligation may be temporarily absent from that snapshot while the
        current endpoint or Portal transition is still making progress.  The
        controller lease therefore stays with that route until its terminal
        callback; explicit ``route_invalidated``/stall events remain the
        cancellation boundary.
        """
        try:
            successor_route_id = max(
                0, int(payload.get("successor_route_id", 0) or 0)
            )
        except (TypeError, ValueError):
            successor_route_id = 0
        if successor_route_id > 0:
            return
        reason = str(payload.get("reason", "graph_route_unavailable"))
        try:
            route_id = max(0, int(payload.get("route_id", 0) or 0))
        except (TypeError, ValueError):
            route_id = 0
        if route_id <= 0:
            route_id = max(
                int(getattr(self, "active_route_id", 0) or 0),
                int(getattr(self, "latest_route_id", 0) or 0),
            )
        if (
            self.latest_intent_priority >= 2
            or self.active_intent_priority >= 2
            or (
                self.persistent_execution
                and self.persistent_target_pending_transaction > 0
            )
        ):
            self.publish_bridge_status(
                "frontier_route_unavailable_ignored",
                reason="higher_priority_target_owns_controller",
                route_id=route_id,
                source_status=payload,
            )
            return
        frontier_owned = (
            self.active_intent_source == "global_slam_frontier"
            or self.latest_intent_source == "global_slam_frontier"
        )
        if not frontier_owned or not self._is_active_mode() or route_id <= 0:
            return
        active_route_id = int(getattr(self, "active_route_id", 0) or 0)
        released_route = payload.get("released_controller_route")
        if not isinstance(released_route, dict):
            released_route = {}
        try:
            released_route_id = int(released_route.get("route_id", 0) or 0)
        except (TypeError, ValueError):
            released_route_id = 0
        # GlobalFrontier publishes the endpoint terminal and then starts a
        # new planning transaction.  During that boundary the persistent
        # MoveBase action is intentionally still alive, but it no longer owns
        # the old endpoint.  Treating this as an ordinary active route keeps
        # ``terminal_hold_active`` set forever and prevents the next route from
        # being admitted.  The explicit route identity and terminal bit are
        # the only evidence that authorizes this release; a plain unavailable
        # event while a route is moving must remain deferred.
        terminal_boundary = bool(
            released_route.get("terminal_received")
            and released_route.get("controller_pending")
            and released_route_id == route_id
        )
        if (
            bool(getattr(self, "action_active", False))
            and active_route_id == route_id
            and str(
                getattr(self, "active_intent_source", "unknown") or "unknown"
            ).strip().lower() == "global_slam_frontier"
            and int(getattr(self, "active_intent_priority", 0) or 0) == 0
            and not terminal_boundary
        ):
            # Planning and control have separate leases. Keep the physical
            # route alive and let its terminal event trigger the next graph
            # decision; cancelling here would turn ordinary map churn into a
            # PREEMPTED action and an avoidable stop/restart cycle.
            self.publish_bridge_status(
                "frontier_route_unavailable_deferred",
                route_id=route_id,
                active_route_id=active_route_id,
                active_route_kind=str(
                    getattr(self, "active_route_kind", "") or ""
                ),
                action_active=True,
                controller_lease="held_until_terminal",
                reason=reason,
                source_status=payload,
            )
            return
        stale_goal = self.last_dispatched_goal
        release = getattr(
            self, "_release_frontier_controller_lease_locked", None
        )
        release_reason = reason
        if terminal_boundary and release_reason == "terminal_boundary":
            release_reason = "terminal_boundary"
        elif terminal_boundary and not release_reason.startswith(
            ("endpoint_terminal_", "terminal_boundary_")
        ):
            release_reason = "endpoint_terminal_%s" % release_reason
        if release is None or not release(route_id, release_reason):
            return
        self.prefetched_frontier_goal = None
        self.prefetched_frontier_route_id = 0
        self.prefetched_frontier_entry_yaw = None
        self.prefetched_frontier_entry_yaw_basis = ""
        self.latest_goal = None
        self.last_dispatched_goal = None
        self.last_dispatch_identity = None
        self.last_terminal_goal = None
        self.latest_intent_source = "waiting_global_slam_frontier"
        self.latest_intent_priority = 0
        self.latest_route_kind = ""
        self.latest_mission_route_kind = ""
        self.latest_graph_transaction_id = 0
        self.latest_route_id = 0
        self.latest_intent_goal = None
        self.publish_bridge_status(
            "frontier_route_unavailable",
            route_id=route_id,
            released_route_id=route_id,
            successor_route_id=0,
            controller_lease="released",
            reason=release_reason,
            stale_goal=(
                None
                if stale_goal is None
                else [
                    round(float(stale_goal.pose.position.x), 3),
                    round(float(stale_goal.pose.position.y), 3),
                ]
            ),
            source_status=payload,
        )

    def on_frontier_status(self, message):
        """Consume frontier lifecycle state without treating it as a goal."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        event = str(payload.get("event", "")).strip()
        with self.lock:
            self._remember_frontier_map_epoch_locked(payload)
            observe_watchdog = getattr(
                self, "_observe_route_lease_watchdog_status_locked", None
            )
            if callable(observe_watchdog):
                observe_watchdog(payload)
            if event == "controller_lease_released":
                self._apply_controller_lease_release_locked(payload)
                return
            if event == "frontier_route_unavailable":
                self._handle_frontier_route_unavailable_locked(payload)
                return
            if event == "place_graph_waiting_for_portal_frontier":
                self.frontier_portal_wait = True
                self.frontier_portal_wait_route_id = max(
                    self.frontier_portal_wait_route_id,
                    int(payload.get("route_id", 0) or 0),
                )
                self.prefetched_frontier_goal = None
                self.prefetched_frontier_route_id = 0
                self.prefetched_frontier_entry_yaw = None
                self.prefetched_frontier_entry_yaw_basis = ""
                self.publish_bridge_status(
                    "frontier_portal_wait",
                    waiting_route_id=int(self.frontier_portal_wait_route_id),
                    hop_limit=payload.get("hop_limit"),
                    skipped_candidates=payload.get("skipped_candidates"),
                )
                return
            if event == "frontier_prefetched":
                raw_goal = payload.get("pending_goal")
                try:
                    pending_route_id = max(
                        0, int(payload.get("pending_route_id", 0) or 0)
                    )
                    active_route_id = max(
                        0, int(payload.get("route_id", 0) or 0)
                    )
                    x, y = float(raw_goal[0]), float(raw_goal[1])
                except (TypeError, ValueError, IndexError):
                    return
                if (
                    pending_route_id == active_route_id + 1
                    and math.isfinite(x)
                    and math.isfinite(y)
                ):
                    self.prefetched_frontier_goal = (x, y)
                    self.prefetched_frontier_route_id = pending_route_id
                    raw_entry_yaw = payload.get("pending_entry_yaw")
                    try:
                        entry_yaw = float(raw_entry_yaw)
                        self.prefetched_frontier_entry_yaw = (
                            entry_yaw if math.isfinite(entry_yaw) else None
                        )
                    except (TypeError, ValueError):
                        self.prefetched_frontier_entry_yaw = None
                    self.prefetched_frontier_entry_yaw_basis = str(
                        payload.get("pending_entry_yaw_basis", "")
                    ).strip()
                    self.publish_bridge_status(
                        "frontier_prefetch_available",
                        active_route_id=active_route_id,
                        pending_route_id=pending_route_id,
                        pending_goal=[round(x, 3), round(y, 3)],
                        pending_entry_yaw=(
                            None
                            if self.prefetched_frontier_entry_yaw is None
                            else round(self.prefetched_frontier_entry_yaw, 4)
                        ),
                        pending_entry_yaw_basis=(
                            self.prefetched_frontier_entry_yaw_basis or None
                        ),
                    )
                return
            # A promotion, invalidation, or fresh non-pending route makes the
            # old endpoint's cached successor ineligible.
            if event in (
                "terminal_prefetch_promoted",
                "route_invalidated",
                "frontier_exhausted",
                "terminal_replan_requested",
                "replan_acknowledged",
            ):
                self.prefetched_frontier_goal = None
                self.prefetched_frontier_route_id = 0
                self.prefetched_frontier_entry_yaw = None
                self.prefetched_frontier_entry_yaw_basis = ""
            if event not in ("route_invalidated", "frontier_exhausted"):
                return
            reason = str(payload.get("reason", "unknown"))
            # Do not cancel the persistent action lease while a visual target
            # waits for validation or already owns the mission.
            if self.persistent_execution and (
                self.latest_intent_priority >= 2
                or self.active_intent_priority >= 2
                or self.persistent_target_pending_transaction > 0
            ):
                self.publish_bridge_status(
                    "frontier_invalidation_ignored",
                    reason=reason,
                    latest_transaction_id=int(self.latest_goal_transaction_id),
                    latest_priority=int(self.latest_intent_priority),
                    active_priority=int(self.active_intent_priority),
                    pending_target_transaction=int(
                        self.persistent_target_pending_transaction
                    ),
                )
                return
            frontier_owned = (
                self.active_intent_source == "global_slam_frontier"
                or self.latest_intent_source == "global_slam_frontier"
            )
            if not frontier_owned or not self._is_active_mode():
                return
            stale_goal = self.last_dispatched_goal
            self.cancel_locked("frontier_%s_%s" % (event, reason))
            # Wait for a validated replacement route; never redispatch this
            # stale source pose from the bridge timer.
            self.latest_goal = None
            self.last_dispatched_goal = None
            self.last_dispatch_identity = None
            self.last_terminal_goal = None
            self.latest_intent_source = "waiting_global_slam_frontier"
            self.latest_intent_priority = 0
            self.latest_route_kind = ""
            self.latest_graph_transaction_id = 0
            self.latest_route_id = 0
            self.latest_intent_goal = None
            self.publish_bridge_status(
                "frontier_%s" % event,
                reason=reason,
                stale_goal=(
                    None
                    if stale_goal is None
                    else [
                        round(float(stale_goal.pose.position.x), 3),
                        round(float(stale_goal.pose.position.y), 3),
                    ]
                ),
                source_status=payload,
            )
            rospy.logwarn(
                "TEB goal bridge released stale frontier action: reason=%s",
                reason,
            )
