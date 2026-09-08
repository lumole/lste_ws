"""Frontier lifecycle observations for the TEB action bridge."""

import json
import math

import rospy


class TebGoalBridgeFrontierStatusMixin:
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
        release_reason = (
            "endpoint_terminal_%s" % reason if terminal_boundary else reason
        )
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
