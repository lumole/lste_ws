#!/usr/bin/env python3
"""Regression tests for actuator attribution of a route's elapsed clocks.

A demo or a mission may leave the command mux forwarding someone else (a human
on teleop, a navigation hold, a finished task, a slice reset).  While that is
true the frontier's progress clock must not convict its own route: the vehicle
moving -- or standing still -- says nothing about a route whose commands the
boundary is discarding.
"""

import ast
import json
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SOURCE = SCRIPTS_DIR / "global_frontier_execution_watchdogs.py"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import math  # noqa: E402
import rospy  # noqa: E402
from global_frontier_models import ActiveRouteWatchdogState  # noqa: E402
from global_frontier_route_actuator import RouteActuatorStatus  # noqa: E402
from global_frontier_route_lease import (  # noqa: E402
    route_lease_failure_decision,
)


EXTRACTED = (
    "route_actuator_is_forwarding",
    "route_actuator_diagnostic",
    "evaluate_active_route_watchdogs",
    "update_post_turn_watchdog",
    "active_waypoint_distance",
)


def load_methods():
    """Extract the watchdog methods without importing the ROS node."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    mixin = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GlobalFrontierExecutionWatchdogMixin"
    )
    selected = [
        node for node in mixin.body
        if isinstance(node, ast.FunctionDef) and node.name in EXTRACTED
    ]
    namespace = {
        "math": math,
        "rospy": rospy,
        "ActiveRouteWatchdogState": ActiveRouteWatchdogState,
        "route_lease_failure_decision": route_lease_failure_decision,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"
        ),
        namespace,
    )
    return namespace


METHODS = load_methods()


class WatchdogExplorer:
    route_actuator_is_forwarding = METHODS["route_actuator_is_forwarding"]
    route_actuator_diagnostic = METHODS["route_actuator_diagnostic"]
    evaluate_active_route_watchdogs = METHODS["evaluate_active_route_watchdogs"]
    update_post_turn_watchdog = METHODS["update_post_turn_watchdog"]
    active_waypoint_distance = METHODS["active_waypoint_distance"]

    def __init__(self, mux_payload):
        self.route_actuator = RouteActuatorStatus(source="teb")
        self.route_actuator.apply(mux_payload)
        self.route_actuator_source = "teb"
        self.active_route_id = 412
        self.active_route_kind = "frontier_endpoint"
        self.active_since = 0.0
        self.active_progress_time = 10.0
        self.active_last_progress_signal = "route_progress"
        self.active_portal_gate_approached_at = None
        self.stall_timeout = 12.0
        self.active_timeout = 45.0
        self.post_turn_stall_timeout = None
        self.active_turn_completed_route_id = 0
        self.active_turn_completed_goal = None
        self.active_turn_completed_wall = 0.0
        self.active_turn_completed_odom_xy = None
        self.active_turn_completed_translation = 0.0
        self.active_turn_completed_launched = False
        self.pose_odom = None
        self.progress_epsilon = 0.05
        self.controller_owned_route_failure = False
        self.recovery_pending_route_id = 0
        self.turn_connector_released = True
        self.turn_supervisor_state = ""
        self.turn_supervisor_route_kind = ""
        self.waypoint_release_radius = 0.8
        self.active_last_waypoint_map = None
        self.events = []

    def active_region_is_stagnant(self, *args, **kwargs):
        """Region dwell is a place fact and is not under test here."""
        return False

    def publish_status(self, event, **payload):
        self.events.append((event, payload))

    def tick(self, now, x=1.0, y=0.0):
        return self.evaluate_active_route_watchdogs(
            x, y, (0.0, 0.0), now, False, None, 0.0,
        )


def teleop_mux_status():
    return json.dumps(
        {
            "event": "actuator_command",
            "source": "teb",
            "selected_mode": "teleop",
            "active": False,
            "block_reason": "source_not_selected",
        }
    )


def teb_mux_status():
    return json.dumps(
        {
            "event": "actuator_command",
            "source": "teb",
            "selected_mode": "teb",
            "active": True,
            "block_reason": "none",
        }
    )


class RouteStallAttributionTest(unittest.TestCase):
    def test_a_human_on_teleop_cannot_fail_this_nodes_route(self):
        explorer = WatchdogExplorer(teleop_mux_status())

        # The clock has run out twice over while the human drove.
        watchdog = explorer.tick(now=40.0)

        self.assertFalse(watchdog.actuator_forwarding)
        self.assertFalse(watchdog.stalled)
        self.assertFalse(watchdog.expired)
        self.assertTrue(watchdog.stall_suppressed_by_actuator)
        self.assertEqual(
            [event for event, _ in explorer.events],
            ["route_stall_not_attributable"],
        )

    def test_the_suppressed_route_resumes_a_full_window_when_teb_returns(self):
        explorer = WatchdogExplorer(teleop_mux_status())
        explorer.tick(now=40.0)

        # The human hands /cmd_vel back at t=100: the route has not failed,
        # and it gets the same stall window it would have had on dispatch.
        explorer.route_actuator.apply(teb_mux_status())
        recovered = explorer.tick(now=100.0)
        self.assertTrue(recovered.actuator_forwarding)
        self.assertFalse(recovered.stalled)

        self.assertFalse(explorer.tick(now=111.0).expired)
        self.assertTrue(explorer.tick(now=113.0).expired)

    def test_a_forwarding_teb_boundary_keeps_the_historical_watchdog(self):
        explorer = WatchdogExplorer(teb_mux_status())

        watchdog = explorer.tick(now=40.0)

        self.assertTrue(watchdog.actuator_forwarding)
        self.assertTrue(watchdog.stalled)
        self.assertTrue(watchdog.expired)
        self.assertFalse(watchdog.stall_suppressed_by_actuator)
        self.assertEqual(explorer.events, [])

    def test_a_controller_terminal_still_fails_a_suppressed_route(self):
        explorer = WatchdogExplorer(teleop_mux_status())
        explorer.recovery_pending_route_id = explorer.active_route_id

        watchdog = explorer.tick(now=40.0)

        self.assertTrue(watchdog.stalled)
        self.assertTrue(watchdog.expired)
        self.assertFalse(watchdog.stall_suppressed_by_actuator)

    def test_the_suppression_is_reported_once_per_route(self):
        explorer = WatchdogExplorer(teleop_mux_status())

        explorer.tick(now=40.0)
        explorer.tick(now=41.0)

        self.assertEqual(
            [event for event, _ in explorer.events],
            ["route_stall_not_attributable"],
        )
        _, payload = explorer.events[0]
        self.assertEqual(payload["route_actuator_selected_mode"], "teleop")
        self.assertEqual(
            payload["route_actuator_block_reason"], "source_not_selected"
        )
        self.assertEqual(payload["route_id"], 412)

    def test_a_portal_edge_deadline_does_not_expire_undriven(self):
        explorer = WatchdogExplorer(teleop_mux_status())
        explorer.active_route_kind = "portal_transition"
        explorer.active_portal_gate_approached_at = 20.0

        watchdog = explorer.tick(now=90.0, x=0.0, y=0.0)

        self.assertFalse(watchdog.portal_edge_expired)
        self.assertFalse(watchdog.expired)

    def test_no_mux_contract_keeps_the_legacy_watchdog(self):
        explorer = WatchdogExplorer(teleop_mux_status())
        explorer.route_actuator = SimpleNamespace()

        watchdog = explorer.tick(now=40.0)

        self.assertTrue(watchdog.actuator_forwarding)
        self.assertTrue(watchdog.expired)


if __name__ == "__main__":
    unittest.main()
