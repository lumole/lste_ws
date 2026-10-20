"""Bind portal-terminal outcomes to physical gate-crossing evidence."""

from global_frontier_portal_crossing import (
    portal_crossing_depth,
    portal_crossing_evidence,
    portal_signed_distance,
)


class GlobalFrontierPortalCrossingLifecycleMixin:
    """Confirm a place-graph edge in the stable odom frame."""

    def active_portal_crossing_plane(self):
        """Return the gate and its doorway-own far-side direction point.

        The crossing test must measure against the doorway, not against where
        the route happens to end: ``active_portal_destination_odom_xy`` is the
        navigation endpoint, which can sit deep inside the far room (or off to
        the side), and a plane through that goal verifies a "crossing" the
        moment a route moves toward its goal -- without the base ever passing
        the gate. The certified hypothesis record stores the gate and a point
        one crossing-depth along the doorway's own wall normal; only their
        *direction* is used, with the odom gate as origin, so the two frames
        never mix.
        """
        gate_xy = getattr(self, "active_portal_gate_odom_xy", None)
        destination_xy = getattr(self, "active_portal_destination_odom_xy", None)
        ledger = getattr(self, "portal_hypothesis_ledger", None)
        getter = getattr(ledger, "get", None)
        if gate_xy is None or not callable(getter):
            return gate_xy, destination_xy
        portal_id = getattr(self, "last_portal_hypothesis_id", None)
        transaction = getattr(self, "portal_transaction", None)
        if transaction is not None:
            try:
                transaction_portal_id = transaction.snapshot().portal_id
            except AttributeError:
                transaction_portal_id = None
            if transaction_portal_id is not None:
                portal_id = transaction_portal_id
        try:
            record = getter(int(portal_id)) if portal_id else None
        except (TypeError, ValueError):
            record = None
        if not isinstance(record, dict):
            return gate_xy, destination_xy
        physical_gate = record.get("physical_gate")
        physical_destination = record.get("physical_destination")
        if not physical_gate or not physical_destination:
            return gate_xy, destination_xy
        direction_x = float(physical_destination[0]) - float(physical_gate[0])
        direction_y = float(physical_destination[1]) - float(physical_gate[1])
        if (direction_x * direction_x + direction_y * direction_y) <= 1e-12:
            return gate_xy, destination_xy
        return gate_xy, (
            float(gate_xy[0]) + direction_x,
            float(gate_xy[1]) + direction_y,
        )

    def active_portal_crossing_evidence(self):
        """Evaluate the active portal using frozen odom geometry only."""
        terminal_xy = (
            None
            if getattr(self, "pose_odom", None) is None
            else (float(self.pose_odom.x), float(self.pose_odom.y))
        )
        # Room-observation and controller endpoint radii intentionally do not
        # enter this test.  They are much larger than a door and made a robot
        # that had visibly passed through an opening wait indefinitely for an
        # unrelated room-scale depth.  This is a pure doorway geometry fact.
        minimum_depth = portal_crossing_depth(
            getattr(self, "clearance", 0.0)
        )
        transaction = getattr(self, "portal_transaction", None)
        transaction_snapshot = (
            None if transaction is None else transaction.snapshot()
        )
        gate_xy, destination_xy = self.active_portal_crossing_plane()
        # The route-local start can be at the doorway after a recovery retry.
        # Preserve the transaction-level source proof and feed the latest
        # signed distance into it as additional physical evidence.
        current_signed_distance = portal_signed_distance(
            terminal_xy,
            gate_xy,
            destination_xy,
        )
        if transaction is not None and current_signed_distance is not None:
            transaction.observe_source_side(
                current_signed_distance,
                route_id=getattr(self, "active_route_id", None),
                reason="source_side_observed_during_route",
            )
            transaction_snapshot = transaction.snapshot()
        return portal_crossing_evidence(
            getattr(self, "active_start_odom_xy", None),
            gate_xy,
            destination_xy,
            terminal_xy,
            getattr(self, "active_portal_gate_approached_at", None) is not None,
            minimum_depth,
            getattr(self, "active_portal_crossing_preobserved", False),
            source_side_proven=(
                False
                if transaction_snapshot is None
                else bool(transaction_snapshot.source_side_proven)
            ),
        )

    def confirm_active_portal_crossing(self):
        """Persist a verified crossing on the pending source-place transaction."""
        evidence = self.active_portal_crossing_evidence()
        if evidence.verified:
            transaction = getattr(self, "portal_transaction", None)
            if transaction is not None:
                standoff = transaction.destination_standoff(
                    self.active_route_id,
                    reason="physical_destination_standoff_reached",
                )
                if standoff is not None:
                    self.publish_status(
                        "portal_transaction_phase",
                        transaction_id=int(standoff.transaction_id),
                        route_id=int(standoff.route_id),
                        state=standoff.state,
                        transition=standoff.last_transition,
                        reason=standoff.last_reason,
                    )
                transition = transaction.crossing_verified(
                    self.active_route_id,
                    reason=evidence.reason,
                )
                if transition is not None:
                    self.publish_status(
                        "portal_transaction_phase",
                        transaction_id=int(transition.transaction_id),
                        route_id=int(transition.route_id),
                        state=transition.state,
                        transition=transition.last_transition,
                        reason=transition.last_reason,
                    )
            confirm = getattr(
                self.place_departure, "confirm_physical_gate_crossing", None,
            )
            if confirm is not None:
                confirm()
        return evidence
