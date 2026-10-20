"""Watchdogs governing the lifecycle of an active frontier route."""

import math

import rospy

from global_frontier_models import ActiveRouteWatchdogState
from global_frontier_route_lease import route_lease_failure_decision


class GlobalFrontierExecutionWatchdogMixin:
    """Decide whether active-route progress remains healthy."""

    def route_actuator_is_forwarding(self):
        """Return whether the actuator boundary carries this node's routes.

        Without a configured contract (no mux status in the composition) the
        answer is True, which keeps the historical watchdog authority.  See
        ``global_frontier_route_actuator`` for the ownership reasons that make
        it False.
        """
        status = getattr(self, "route_actuator", None)
        forwarding = getattr(status, "forwarding", None)
        if not callable(forwarding):
            return True
        return bool(forwarding())

    def route_actuator_diagnostic(self):
        """Return the actuator-boundary fields for a status payload."""
        status = getattr(self, "route_actuator", None)
        diagnostic = getattr(status, "diagnostic", None)
        if not callable(diagnostic):
            return {}
        return dict(diagnostic())

    def evaluate_active_route_watchdogs(
        self, x, y, robot_map, now, portal_transition, active_component,
        active_information,
    ):
        """Evaluate turn, recovery, region, and timeout watchdogs."""
        # A turn can be owned directly by the TEB adapter for endpoint, portal,
        # and local-egress actions; it is not always represented by the legacy
        # ``frontier_turn_connector`` route kind.  Treat a matching supervisor
        # TURNING state as physical progress.  Without this boundary, a valid
        # in-place rotation is judged by the translational stall clock and the
        # route is cancelled while the base is still acquiring its heading.
        supervisor_state = str(
            getattr(self, "turn_supervisor_state", "")
        ).strip().upper()
        supervisor_route_kind = str(
            getattr(self, "turn_supervisor_route_kind", "")
        ).strip().lower()
        supervisor_turn_active = (
            supervisor_state == "TURNING"
            and self.active_route_kind in (
                "frontier_turn_connector",
                "frontier_endpoint",
                "portal_transition",
                "local_egress",
            )
            and (
                not supervisor_route_kind
                or supervisor_route_kind == self.active_route_kind
            )
        )
        turn_phase_active = (
            (
                self.active_route_kind == "frontier_turn_connector"
                and not self.turn_connector_released
            )
            or supervisor_turn_active
        )
        recovery_pending = self.recovery_pending_route_id == self.active_route_id
        if turn_phase_active:
            self.active_progress_time = now
        (
            post_turn_goal_matches,
            post_turn_elapsed,
            post_turn_translation,
        ) = self.update_post_turn_watchdog(x, y, now)
        # Both elapsed clocks are read before the actuator gate freezes them,
        # so the suppressed observation stays available for the status record.
        observed_post_turn_stall = (
            not turn_phase_active
            and self.post_turn_stall_timeout is not None
            and post_turn_goal_matches
            and not self.active_turn_completed_launched
            and post_turn_elapsed >= self.post_turn_stall_timeout
            and now - self.active_progress_time >= self.post_turn_stall_timeout
        )
        observed_progress_stall = (
            not turn_phase_active
            and now - self.active_progress_time > self.stall_timeout
        )
        # Route progress is only attributable to this node while the actuator
        # boundary carries its commands.  Someone else on /cmd_vel (a human on
        # teleop, a navigation hold, a finished task, a slice reset) is not
        # this route failing, so the clock is frozen exactly as it is for an
        # owned turn: the route stays committed and resumes a full timeout
        # window once the boundary forwards its commands again.  A controller
        # terminal remains authoritative through ``recovery_pending``.
        actuator_forwarding = self.route_actuator_is_forwarding()
        # Time that accrued while the boundary held this route is not this
        # route's time -- including when no tick ran during the hold, which is
        # why the resume itself restarts the window.
        resuming_held_route = (
            actuator_forwarding
            and int(getattr(self, "route_actuator_held_route_id", 0))
            == int(self.active_route_id)
        )
        if not actuator_forwarding:
            self.route_actuator_held_route_id = int(self.active_route_id)
        elif resuming_held_route:
            self.route_actuator_held_route_id = 0
        suppress_elapsed_evidence = not actuator_forwarding or resuming_held_route
        if suppress_elapsed_evidence:
            self.active_progress_time = now
            # The portal-edge deadline is the same kind of elapsed evidence:
            # a crossing that was never driven has not run out of time.
            if self.active_portal_gate_approached_at is not None:
                self.active_portal_gate_approached_at = now

        post_turn_stalled = (
            observed_post_turn_stall and not suppress_elapsed_evidence
        )
        region_stagnant = self.active_region_is_stagnant(
            x,
            y,
            robot_map,
            now,
            portal_transition,
            turn_phase_active,
            recovery_pending,
            active_component,
            active_information,
        )
        # Observation stagnation describes a place-level fact, not an action
        # failure. In particular, it must not terminate or replace the
        # still-active move_base transaction. The terminal lifecycle closes
        # the place only after the bridge proves that this exact route ended.
        stalled = (
            recovery_pending
            or post_turn_stalled
            or (observed_progress_stall and not suppress_elapsed_evidence)
        )
        waypoint_distance = self.active_waypoint_distance(robot_map)
        waypoint_reached = (
            self.active_last_waypoint_map is None
            or waypoint_distance <= self.waypoint_release_radius
        )
        # A portal route represents one graph edge, not an open-ended local
        # search. Its lease starts at the physical doorway, rather than at a
        # remote route-selection point. This separates corridor approach from
        # the actual crossing without adding a controller-specific timeout.
        portal_edge_started_at = getattr(
            self, "active_portal_gate_approached_at", None,
        )
        portal_edge_expired = (
            portal_transition
            and portal_edge_started_at is not None
            and now - portal_edge_started_at > self.active_timeout
        )
        active_timeout = (
            now - self.active_since > self.active_timeout and stalled
        )
        lease_decision = route_lease_failure_decision(
            controller_owns_failure=bool(
                getattr(self, "controller_owned_route_failure", False)
            ),
            recovery_pending=recovery_pending,
            actuator_forwarding=actuator_forwarding,
            route_stalled=stalled,
            post_turn_stalled=post_turn_stalled,
            portal_edge_expired=portal_edge_expired,
            active_timeout=active_timeout,
        )
        expired = lease_decision.release
        suppressed_stall = (
            not actuator_forwarding
            and not recovery_pending
            and (observed_progress_stall or observed_post_turn_stall)
        )
        if (
            suppressed_stall
            and int(getattr(self, "last_suppressed_stall_route_id", 0))
            != int(self.active_route_id)
        ):
            # One record per route: a suppressed stall is a statement about
            # the boundary, not a repeated route event.
            self.last_suppressed_stall_route_id = int(self.active_route_id)
            actuator_diagnostic = self.route_actuator_diagnostic()
            publish = getattr(self, "publish_status", None)
            if callable(publish):
                publish(
                    "route_stall_not_attributable",
                    route_id=int(self.active_route_id),
                    route_kind=str(self.active_route_kind),
                    active_elapsed=round(float(now - self.active_since), 3),
                    observed_post_turn_stall=bool(observed_post_turn_stall),
                    **actuator_diagnostic
                )
            rospy.logwarn(
                "Global frontier withheld route failure: the actuator "
                "boundary does not forward this route route_id=%d "
                "selected_mode=%s block_reason=%s",
                self.active_route_id,
                actuator_diagnostic["route_actuator_selected_mode"],
                actuator_diagnostic["route_actuator_block_reason"],
            )
        if (
            bool(getattr(self, "controller_owned_route_failure", False))
            and stalled
            and not recovery_pending
            and int(getattr(self, "last_route_stagnation_reported_route_id", 0))
            != int(self.active_route_id)
        ):
            self.last_route_stagnation_reported_route_id = int(self.active_route_id)
            publish = getattr(self, "publish_status", None)
            if callable(publish):
                publish(
                    "route_stagnant_observed",
                    route_id=int(self.active_route_id),
                    route_kind=str(self.active_route_kind),
                    authority=lease_decision.authority,
                    reason=lease_decision.reason,
                    active_elapsed=round(float(now - self.active_since), 3),
                    last_progress_signal=str(
                        getattr(self, "active_last_progress_signal", "none")
                    ),
                )
            rospy.logwarn(
                "Global frontier observed route stagnation without releasing "
                "controller-owned lease route_id=%d",
                self.active_route_id,
            )
        return ActiveRouteWatchdogState(
            turn_phase_active=turn_phase_active,
            recovery_pending=recovery_pending,
            post_turn_goal_matches=post_turn_goal_matches,
            post_turn_elapsed=post_turn_elapsed,
            post_turn_translation=post_turn_translation,
            post_turn_stalled=post_turn_stalled,
            region_stagnant=region_stagnant,
            stalled=stalled,
            waypoint_distance=waypoint_distance,
            waypoint_reached=waypoint_reached,
            portal_edge_expired=portal_edge_expired,
            expired=expired,
            actuator_forwarding=bool(actuator_forwarding),
            stall_suppressed_by_actuator=bool(suppressed_stall),
        )

    def update_post_turn_watchdog(self, x, y, now):
        """Track whether an execution handoff actually launched translation."""
        goal_matches = (
            self.active_turn_completed_route_id == self.active_route_id
            and self.active_turn_completed_goal is not None
            and math.hypot(
                self.active_turn_completed_goal[0] - x,
                self.active_turn_completed_goal[1] - y,
            ) <= 0.10
        )
        elapsed = (
            0.0 if not goal_matches
            else max(0.0, now - self.active_turn_completed_wall)
        )
        translation = None
        if (
            goal_matches
            and self.active_turn_completed_odom_xy is not None
            and self.pose_odom is not None
        ):
            translation = math.hypot(
                float(self.pose_odom.x) - self.active_turn_completed_odom_xy[0],
                float(self.pose_odom.y) - self.active_turn_completed_odom_xy[1],
            )
            self.active_turn_completed_translation = max(
                self.active_turn_completed_translation,
                translation,
            )
            if (
                not self.active_turn_completed_launched
                and self.active_turn_completed_translation >= self.progress_epsilon
            ):
                # A map->odom correction can increase map-space goal distance
                # while the base follows a doorway detour.  Odom launch is the
                # invariant proof that this route is alive.
                self.active_turn_completed_launched = True
                self.active_progress_time = now
                self.active_last_progress_signal = "post_turn_odom_launch"
                self.publish_status(
                    "post_turn_progress_watchdog_released",
                    route_id=int(self.active_route_id),
                    goal=[round(float(x), 3), round(float(y), 3)],
                    odom_translation=round(
                        float(self.active_turn_completed_translation), 3
                    ),
                )
        return goal_matches, elapsed, translation

    def active_region_is_stagnant(
        self, x, y, robot_map, now, portal_transition, turn_phase_active,
        recovery_pending, active_component, active_information,
    ):
        """Report no-gain dwell without changing the route or place state.

        A planning tick can observe a finished room while TEB still owns the
        endpoint action. This method is deliberately read-only: turning a
        place dormant here would make the next timer choose a successor before
        the active action has emitted its matching terminal. Terminal-time
        observation handling owns the eventual place closure.
        """
        if (
            portal_transition
            or turn_phase_active
            or recovery_pending
            # Direct target evidence explicitly owns its structural place.
            # Ordinary coverage completion must not override that higher
            # level mission claim merely because detector frames are sparse.
            or self.target_region_claim_active
            or self.active_observation_session_started_at is None
        ):
            return False
        region = self.region_memory.stagnant(
            x,
            y,
            robot_map[0],
            robot_map[1],
            now,
            component=active_component,
            region_id=self.active_frontier_region_id,
        )
        if region is None:
            return False
        rospy.loginfo_throttle(
            5.0,
            "Global frontier observed no information gain for %.1fs in "
            "region id=%d; retaining active route_id=%d until its terminal",
            now - region["last_gain"],
            region["id"],
            self.active_route_id,
        )
        return True

    def active_waypoint_distance(self, robot_map):
        """Return the distance to the bridge's latest short waypoint."""
        if self.active_last_waypoint_map is None:
            return float("inf")
        return math.hypot(
            self.active_last_waypoint_map[0] - robot_map[0],
            self.active_last_waypoint_map[1] - robot_map[1],
        )

    def active_frontier_observation_ready(self, endpoint_distance):
        """Require the controller's physical arrival before starting dwell."""
        return (
            self.active_last_waypoint_map is not None
            and self.active_route_kind != "frontier_turn_connector"
            and float(endpoint_distance) <= self.waypoint_release_radius
        )
