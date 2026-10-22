#!/usr/bin/env python3
"""Regression tests for controller-owned route lease failure authority."""

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from global_frontier_route_lease import (  # noqa: E402
    AUTHORITY_CONTROLLER_TERMINAL,
    AUTHORITY_GLOBAL_WATCHDOG,
    route_lease_failure_decision,
)


class RouteLeaseAuthorityTest(unittest.TestCase):
    def test_persistent_stall_is_diagnostic_only(self):
        decision = route_lease_failure_decision(
            controller_owns_failure=True,
            route_stalled=True,
            post_turn_stalled=True,
            portal_edge_expired=True,
            active_timeout=True,
        )

        self.assertFalse(decision.release)
        self.assertEqual(decision.authority, AUTHORITY_CONTROLLER_TERMINAL)
        self.assertEqual(decision.reason, "controller_owns_failure_terminal")

    def test_controller_terminal_releases_persistent_lease(self):
        decision = route_lease_failure_decision(
            controller_owns_failure=True,
            recovery_pending=True,
        )

        self.assertTrue(decision.release)
        self.assertEqual(decision.authority, AUTHORITY_CONTROLLER_TERMINAL)
        self.assertEqual(decision.reason, "controller_terminal_failure")

    def test_legacy_watchdog_retains_historical_failure_semantics(self):
        decision = route_lease_failure_decision(
            controller_owns_failure=False,
            route_stalled=True,
        )

        self.assertTrue(decision.release)
        self.assertEqual(decision.authority, AUTHORITY_GLOBAL_WATCHDOG)
        self.assertEqual(decision.reason, "stall")

    def test_healthy_route_does_not_release(self):
        decision = route_lease_failure_decision()

        self.assertFalse(decision.release)
        self.assertEqual(decision.reason, "active_route_healthy")

    def test_foreign_actuator_suppresses_every_elapsed_evidence_path(self):
        """A route nobody is driving cannot run out of time."""
        for evidence in (
            {"route_stalled": True},
            {"active_timeout": True, "route_stalled": True},
            {"post_turn_stalled": True},
            {"portal_edge_expired": True},
        ):
            with self.subTest(evidence=evidence):
                decision = route_lease_failure_decision(
                    controller_owns_failure=False,
                    actuator_forwarding=False,
                    **evidence
                )

                self.assertFalse(decision.release)
                self.assertEqual(decision.reason, "actuator_not_forwarding")
                self.assertEqual(
                    decision.authority, AUTHORITY_GLOBAL_WATCHDOG
                )

    def test_controller_terminal_outranks_a_foreign_actuator(self):
        """The controller's own verdict survives the actuator gate."""
        decision = route_lease_failure_decision(
            controller_owns_failure=False,
            recovery_pending=True,
            actuator_forwarding=False,
        )

        self.assertTrue(decision.release)
        self.assertEqual(decision.reason, "controller_terminal_failure")

    def test_forwarding_actuator_keeps_legacy_stall_semantics(self):
        decision = route_lease_failure_decision(
            actuator_forwarding=True,
            route_stalled=True,
        )

        self.assertTrue(decision.release)
        self.assertEqual(decision.reason, "stall")

    def test_hard_stall_breaks_controller_owned_persistent_lease(self):
        """Hard controller stall timeout breaks persistent lease even when controller owns failure."""
        decision = route_lease_failure_decision(
            controller_owns_failure=True,
            hard_stalled=True,
            actuator_forwarding=True,
        )

        self.assertTrue(decision.release)
        self.assertEqual(decision.authority, AUTHORITY_GLOBAL_WATCHDOG)
        self.assertEqual(decision.reason, "hard_controller_stall_timeout")

    def test_hard_stall_suppressed_by_foreign_actuator(self):
        """Foreign actuator still suppresses hard stall (e.g. manual teleop or hold)."""
        decision = route_lease_failure_decision(
            controller_owns_failure=True,
            hard_stalled=True,
            actuator_forwarding=False,
        )

        self.assertFalse(decision.release)
        self.assertEqual(decision.reason, "actuator_not_forwarding")


if __name__ == "__main__":
    unittest.main()
