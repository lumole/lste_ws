"""Validation of terminal messages received from the TEB goal bridge."""

import copy
import math
import time

import rospy


class GlobalFrontierTerminalValidationMixin:
    """Accept only the terminal notification for the active route contract."""

    _DISPATCH_CONTRACT_WAIT_SECONDS = 3.0
    _PENDING_TERMINAL_LIMIT = 16

    @staticmethod
    def _terminal_identity(message):
        """Decode the immutable identity carried by one terminal message."""
        def integer(value, default=0):
            try:
                return int(value or default)
            except (TypeError, ValueError):
                return int(default)

        raw_map_epoch = getattr(message, "map_epoch", 0)
        return {
            "route_id": integer(getattr(message, "route_id", 0)),
            "map_epoch": integer(raw_map_epoch),
            "graph_transaction_id": integer(
                getattr(message, "graph_transaction_id", 0)
            ),
            "lifecycle_transaction_id": integer(
                getattr(message, "lifecycle_transaction_id", "")
            ),
            "action_generation": integer(
                getattr(message, "action_generation", 0)
            ),
            "route_kind": str(
                getattr(message, "route_kind", "") or ""
            ).strip().lower(),
            "mission_route_kind": str(
                getattr(message, "mission_route_kind", "") or ""
            ).strip().lower(),
        }

    def _reject_terminal(self, message, reason, **fields):
        """Record a terminal rejection without mutating route ownership."""
        identity = self._terminal_identity(message)
        publish = getattr(self, "publish_status", None)
        if callable(publish):
            publish(
                "stale_terminal_rejected",
                reason=str(reason),
                terminal_identity=identity,
                **fields
            )
        rospy.loginfo_throttle(
            2.0,
            "Global frontier rejected terminal route_id=%d reason=%s",
            identity["route_id"],
            reason,
        )
        return None

    @staticmethod
    def _pending_terminal_key(identity):
        """Build a stable key for one terminal waiting on dispatch evidence."""
        return (
            int(identity.get("route_id", 0) or 0),
            int(identity.get("action_generation", 0) or 0),
            int(identity.get("lifecycle_transaction_id", 0) or 0),
            int(identity.get("graph_transaction_id", 0) or 0),
            int(identity.get("map_epoch", 0) or 0),
            str(identity.get("route_kind", "") or ""),
            str(identity.get("mission_route_kind", "") or ""),
        )

    def _defer_terminal_until_dispatch(self, message):
        """Retain a valid terminal when its independent dispatch event is late."""
        identity = self._terminal_identity(message)
        if identity["route_id"] <= 0 or identity["action_generation"] <= 0:
            return self._reject_terminal(message, "dispatch_contract_missing")
        pending = getattr(self, "_pending_terminals_waiting_for_dispatch", None)
        if not isinstance(pending, dict):
            pending = {}
            self._pending_terminals_waiting_for_dispatch = pending
        key = self._pending_terminal_key(identity)
        if key not in pending and len(pending) >= self._PENDING_TERMINAL_LIMIT:
            oldest_key = next(iter(pending))
            oldest = pending.pop(oldest_key)
            self._reject_terminal(
                oldest["message"],
                "dispatch_contract_wait_queue_full",
            )
        pending[key] = {
            "message": copy.deepcopy(message),
            "identity": dict(identity),
            "received_monotonic": time.monotonic(),
        }
        publish = getattr(self, "publish_status", None)
        if callable(publish):
            publish(
                "terminal_waiting_for_dispatch",
                reason="dispatch_contract_missing",
                terminal_identity=identity,
                wait_seconds=self._DISPATCH_CONTRACT_WAIT_SECONDS,
            )
        return None

    def _expire_terminals_waiting_for_dispatch(self):
        """Bound the out-of-order grace period without inventing a failure."""
        pending = getattr(self, "_pending_terminals_waiting_for_dispatch", None)
        if not isinstance(pending, dict) or not pending:
            return 0
        now = time.monotonic()
        expired = []
        for key, entry in list(pending.items()):
            if now - float(entry.get("received_monotonic", now)) >= (
                self._DISPATCH_CONTRACT_WAIT_SECONDS
            ):
                expired.append((key, entry))
        for key, entry in expired:
            pending.pop(key, None)
            self._reject_terminal(
                entry["message"],
                "dispatch_contract_timeout",
                waited_seconds=round(
                    now - float(entry.get("received_monotonic", now)), 3
                ),
            )
        return len(expired)

    def _retry_terminals_after_dispatch(self, dispatch):
        """Replay only terminals whose complete identity matches this dispatch."""
        pending = getattr(self, "_pending_terminals_waiting_for_dispatch", None)
        process = getattr(self, "_process_execution_terminal_locked", None)
        if not isinstance(pending, dict) or not pending or not callable(process):
            return 0
        route_id = int(dispatch.get("route_id", 0) or 0)
        if route_id <= 0:
            return 0
        replayed = 0
        for key, entry in list(pending.items()):
            identity = entry["identity"]
            if identity["route_id"] != route_id:
                continue
            matches = all(
                identity[field] == dispatch.get(field)
                for field in (
                    "action_generation",
                    "route_kind",
                    "mission_route_kind",
                    "lifecycle_transaction_id",
                    "graph_transaction_id",
                )
            )
            dispatch_map_epoch = dispatch.get("map_epoch")
            if matches and dispatch_map_epoch is not None:
                matches = identity["map_epoch"] == dispatch_map_epoch
            if not matches:
                continue
            pending.pop(key, None)
            publish = getattr(self, "publish_status", None)
            if callable(publish):
                publish(
                    "terminal_dispatch_contract_reconciled",
                    route_id=route_id,
                    terminal_identity=identity,
                )
            process(entry["message"])
            replayed += 1
        return replayed

    def matching_execution_terminal(self, message):
        """Return a completion only for this exact route transaction.

        Adjacent frontier endpoints can be closer than any useful geometric
        acceptance radius. A pose-only terminal can therefore close the next
        action when ROS delivers the previous callback late. The bridge sends
        an immutable ``route_id`` contract; coordinate checks remain a
        diagnostic guard, never the primary identity proof.
        """
        identity = self._terminal_identity(message)
        terminal_route_id = identity["route_id"]
        if self.task_done:
            return self._reject_terminal(message, "task_already_done")
        if self.active_last_waypoint_map is None:
            return self._reject_terminal(message, "no_active_route_waypoint")
        if getattr(self, "active_frontier", None) is None:
            return self._reject_terminal(message, "predecessor_not_current_owner")
        if getattr(self, "active_terminal_received", False):
            return self._reject_terminal(message, "duplicate_terminal")
        if getattr(self, "awaiting_controller_lease_release_ack", False):
            return self._reject_terminal(
                message, "predecessor_release_ack_pending"
            )
        if terminal_route_id != int(self.active_route_id):
            return self._reject_terminal(
                message,
                "route_id_mismatch",
                active_route_id=int(self.active_route_id),
            )
        terminal_route_kind = identity["route_kind"]
        allowed_route_kinds = {
            str(getattr(self, "active_route_kind", "") or "")
            .strip()
            .lower(),
            str(getattr(self, "active_mission_route_kind", "") or "")
            .strip()
            .lower(),
        }
        allowed_route_kinds.discard("")
        if (
            terminal_route_kind
            and terminal_route_kind not in allowed_route_kinds
        ):
            return self._reject_terminal(
                message,
                "route_kind_mismatch",
                active_route_kind=str(getattr(self, "active_route_kind", "")),
            )
        terminal_epoch = identity["map_epoch"]
        active_epoch = getattr(self, "active_route_map_epoch", None)
        if active_epoch is not None:
            try:
                if terminal_epoch != int(active_epoch):
                    return self._reject_terminal(
                        message,
                        "map_epoch_mismatch",
                        active_map_epoch=int(active_epoch),
                    )
            except (TypeError, ValueError):
                return self._reject_terminal(message, "active_map_epoch_invalid")
        graph_transaction = getattr(
            self, "graph_route_action_transaction", None
        )
        expected_graph_transaction_id = int(
            getattr(graph_transaction, "graph_transaction_id", 0) or 0
        )
        terminal_graph_transaction_id = identity["graph_transaction_id"]
        if expected_graph_transaction_id > 0 and (
            terminal_graph_transaction_id != expected_graph_transaction_id
        ):
            return self._reject_terminal(
                message,
                "graph_transaction_id_mismatch",
                active_graph_transaction_id=expected_graph_transaction_id,
            )
        dispatch = getattr(self, "_bridge_dispatch_contracts", {}).get(
            terminal_route_id
        )
        if not isinstance(dispatch, dict):
            return self._defer_terminal_until_dispatch(message)
        current_lifecycle_id = int(
            getattr(
                getattr(self, "lifecycle_manager", None),
                "current_transaction_id",
                0,
            )
            or 0
        )
        if identity["lifecycle_transaction_id"] <= 0:
            return self._reject_terminal(
                message, "lifecycle_transaction_id_missing"
            )
        if (
            current_lifecycle_id > 0
            and identity["lifecycle_transaction_id"] != current_lifecycle_id
        ):
            return self._reject_terminal(
                message,
                "lifecycle_transaction_id_mismatch",
                active_lifecycle_transaction_id=current_lifecycle_id,
            )
        if identity["action_generation"] <= 0:
            return self._reject_terminal(message, "action_generation_missing")
        for field in (
            "action_generation",
            "lifecycle_transaction_id",
            "graph_transaction_id",
        ):
            expected = int(dispatch.get(field, 0) or 0)
            if expected > 0 and identity[field] != expected:
                return self._reject_terminal(
                    message,
                    "dispatch_contract_%s_mismatch" % field,
                    expected_identity=dispatch,
                )
        dispatch_map_epoch = dispatch.get("map_epoch")
        if dispatch_map_epoch is not None:
            try:
                if terminal_epoch != int(dispatch_map_epoch):
                    return self._reject_terminal(
                        message,
                        "dispatch_contract_map_epoch_mismatch",
                        expected_identity=dispatch,
                    )
            except (TypeError, ValueError):
                return self._reject_terminal(
                    message, "dispatch_contract_map_epoch_invalid"
                )
        dispatch_route_kind = str(dispatch.get("route_kind", "") or "")
        if dispatch_route_kind and terminal_route_kind != dispatch_route_kind:
            return self._reject_terminal(
                message,
                "dispatch_contract_route_kind_mismatch",
                expected_identity=dispatch,
            )
        dispatch_mission_kind = str(
            dispatch.get("mission_route_kind", "") or ""
        )
        if (
            dispatch_mission_kind
            and identity["mission_route_kind"]
            and identity["mission_route_kind"] != dispatch_mission_kind
        ):
            return self._reject_terminal(
                message,
                "dispatch_contract_mission_route_kind_mismatch",
                expected_identity=dispatch,
            )
        terminal_goal = getattr(message, "goal", None)
        if terminal_goal is None:
            return self._reject_terminal(message, "terminal_goal_missing")
        frame = (terminal_goal.header.frame_id or "").strip().lstrip("/")
        expected_frame = (
            "" if self.map_msg is None
            else (self.map_msg.header.frame_id or "map").strip().lstrip("/")
        )
        expected_xy = self.active_last_waypoint_map
        frozen_portal_odom = getattr(
            self, "active_portal_destination_odom_xy", None,
        )
        if (
            self.active_route_kind == "portal_transition"
            and frozen_portal_odom is not None
        ):
            # A portal is a physical graph edge. Its command is intentionally
            # frozen in odom, so the matching terminal must be checked against
            # that same endpoint rather than a later SLAM-corrected map goal.
            expected_frame = "odom"
            expected_xy = frozen_portal_odom
        if frame and expected_frame and frame != expected_frame:
            return self._reject_terminal(
                message,
                "terminal_frame_mismatch",
                expected_frame=expected_frame,
            )
        terminal_delta = math.hypot(
            float(terminal_goal.pose.position.x) - float(expected_xy[0]),
            float(terminal_goal.pose.position.y) - float(expected_xy[1]),
        )
        if terminal_delta > 0.25:
            return self._reject_terminal(
                message,
                "terminal_geometry_mismatch",
                terminal_delta=round(terminal_delta, 4),
            )
        self.active_terminal_contract = {
            "route_id": terminal_route_id,
            "route_kind": terminal_route_kind or dispatch_route_kind,
            "lifecycle_transaction_id": identity[
                "lifecycle_transaction_id"
            ],
            "action_generation": identity["action_generation"],
            "graph_transaction_id": identity["graph_transaction_id"],
            "map_epoch": terminal_epoch,
        }
        return self.active_frontier, terminal_delta
