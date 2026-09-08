"""Continuous physical evidence for a directed portal transaction."""


class GlobalFrontierPortalLiveCrossingMixin:
    """Commit a crossed graph edge before the local action terminal arrives."""

    def observe_active_portal_crossing(self, now=None):
        """Commit a physical portal crossing without waiting for TEB success.

        A portal is a graph-edge transaction, while a TEB terminal only says
        that its local action has stopped. Once odometry proves a crossing,
        retaining the old source place until a final few centimetres of TEB
        motion creates the repeated-room failure this graph prevents.
        """
        if (
            self.active_route_kind != "portal_transition"
            or self.active_frontier is None
            or getattr(self, "active_portal_crossing_observed", False)
            or getattr(self, "active_portal_crossing_rejected", False)
        ):
            return None
        crossing = self.active_portal_crossing_evidence()
        if not crossing.verified:
            return None
        return self.record_portal_place_arrival(
            self.active_frontier,
            now=now,
            evidence_source="continuous_odom",
            retry_on_unconfirmed=False,
        )
