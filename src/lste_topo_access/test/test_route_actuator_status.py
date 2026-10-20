#!/usr/bin/env python3
"""Regression tests for actuator-boundary attribution of route progress."""

import json
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from global_frontier_route_actuator import RouteActuatorStatus  # noqa: E402


def mux_status(source="teb", mode="teb", active=True, reason="none"):
    return json.dumps(
        {
            "event": "actuator_command",
            "sequence": 7,
            "source": source,
            "selected_mode": mode,
            "active": active,
            "block_reason": reason,
        },
        sort_keys=True,
    )


class RouteActuatorStatusTest(unittest.TestCase):
    def test_missing_contract_keeps_historical_authority(self):
        """A composition without the mux must not suppress every stall."""
        status = RouteActuatorStatus(source="teb")

        self.assertTrue(status.forwarding())
        self.assertFalse(status.diagnostic()["route_actuator_received"])

    def test_selected_teb_mode_forwards_routes(self):
        status = RouteActuatorStatus(source="teb")

        self.assertTrue(status.apply(mux_status()))
        self.assertTrue(status.forwarding())

    def test_human_on_teleop_does_not_own_this_routes_progress(self):
        status = RouteActuatorStatus(source="teb")

        status.apply(mux_status(source="teb", mode="teleop", active=False,
                                reason="source_not_selected"))

        self.assertFalse(status.forwarding())
        diagnostic = status.diagnostic()
        self.assertEqual(diagnostic["route_actuator_selected_mode"], "teleop")
        self.assertEqual(
            diagnostic["route_actuator_block_reason"], "source_not_selected"
        )

    def test_a_mode_switch_record_is_enough_to_restore_forwarding(self):
        """The mux announces its new mode before TEB publishes again."""
        status = RouteActuatorStatus(source="teb")
        status.apply(mux_status(mode="teleop", reason="source_not_selected"))
        self.assertFalse(status.forwarding())

        status.apply(
            mux_status(source="none", mode="teb", active=False,
                       reason="controller_mode_switch")
        )

        self.assertTrue(status.forwarding())

    def test_mission_holds_suppress_elapsed_time_evidence(self):
        for reason in ("navigation_hold", "task_complete", "hard_reset",
                       "hard_reset_hold"):
            with self.subTest(block_reason=reason):
                status = RouteActuatorStatus(source="teb")
                status.apply(
                    mux_status(source="none", mode="teb", active=False,
                               reason=reason)
                )

                self.assertFalse(status.forwarding())

    def test_rejected_own_command_keeps_the_historical_watchdog(self):
        """A refused command contract is evidence about our own route."""
        status = RouteActuatorStatus(source="teb")

        status.apply(
            mux_status(source="teb", mode="teb", active=False,
                       reason="route_id_mismatch")
        )

        self.assertTrue(status.forwarding())

    def test_malformed_status_does_not_change_the_decision(self):
        status = RouteActuatorStatus(source="teb")
        status.apply(mux_status(mode="teleop", reason="source_not_selected"))

        for payload in ("not json", "[]", json.dumps({"event": "other"})):
            with self.subTest(payload=payload):
                self.assertFalse(status.apply(payload))
                self.assertFalse(status.forwarding())

    def test_actuator_source_selects_the_mode_that_carries_routes(self):
        status = RouteActuatorStatus(source="sappo")

        status.apply(mux_status(source="teb", mode="teb", active=True))

        self.assertFalse(status.forwarding())
        self.assertTrue(
            status.apply(mux_status(source="sappo", mode="sappo", active=True))
        )
        self.assertTrue(status.forwarding())

    def test_bridge_dispatch_hold_is_not_this_nodes_route_failure(self):
        """A goal the TEB bridge never dispatched has not failed.

        The live two-room crossing was convicted as stalled 12.2 s in while
        the bridge was still holding its dispatch for a controller-lease
        release, which aborted the portal transaction.  The bridge says so
        explicitly on its status topic, and that is an actuator-boundary fact
        like any other.
        """
        status = RouteActuatorStatus(source="teb")
        status.apply(mux_status(source="teb", mode="teb", active=True))
        self.assertTrue(status.forwarding())

        status.observe_dispatch({
            "event": "dispatch_deferred_controller_lease_release",
            "route_id": 3,
            "reason": "pending_controller_lease_release",
            "deferred_reason": "coalesced_global_goal",
        })

        self.assertTrue(status.dispatch_is_deferred())
        self.assertFalse(status.forwarding())
        diagnostic = status.diagnostic()
        self.assertTrue(diagnostic["route_actuator_dispatch_deferred"])
        self.assertEqual(
            diagnostic["route_actuator_dispatch_deferred_route_id"], 3,
        )

    def test_bridge_dispatch_restores_authority_when_it_actually_dispatches(self):
        status = RouteActuatorStatus(source="teb")
        status.observe_dispatch({
            "event": "dispatch_deferred_controller_lease_release",
            "route_id": 3,
        })
        self.assertFalse(status.forwarding())

        status.observe_dispatch({"event": "dispatch", "route_id": 4})

        self.assertFalse(status.dispatch_is_deferred())
        self.assertTrue(status.forwarding())

    def test_dispatch_hold_expires_with_the_bridge_own_escape_window(self):
        """Past the bridge's own watchdog the silence is evidence again."""
        import global_frontier_route_actuator as module

        status = RouteActuatorStatus(source="teb")
        status.observe_dispatch({
            "event": "dispatch_deferred_controller_lease_release",
            "route_id": 3,
        })
        self.assertFalse(status.forwarding())

        record = status._dispatch_deferred
        record["observed_at"] = (
            record["observed_at"]
            - module.BRIDGE_DISPATCH_DEFERRAL_WINDOW_SECONDS
            - 0.01
        )

        self.assertFalse(status.dispatch_is_deferred())
        self.assertTrue(status.forwarding())

    def test_unrelated_bridge_events_do_not_change_forwarding(self):
        status = RouteActuatorStatus(source="teb")
        status.apply(mux_status(source="teb", mode="teb", active=True))

        for payload in ({}, [], "nope", {"event": "active"}):
            with self.subTest(payload=payload):
                status.observe_dispatch(payload)
                self.assertTrue(status.forwarding())


if __name__ == "__main__":
    unittest.main()
