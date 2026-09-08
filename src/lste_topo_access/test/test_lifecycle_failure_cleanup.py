#!/usr/bin/env python3
"""Regression tests for execution cleanup on lifecycle failure."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
DEVEL_PYTHON = Path(__file__).resolve().parents[3] / "devel/lib/python3/dist-packages"
if DEVEL_PYTHON.is_dir() and str(DEVEL_PYTHON) not in sys.path:
    sys.path.insert(0, str(DEVEL_PYTHON))

from lifecycle_manager import State
from teb_goal_bridge_lifecycle import TebGoalBridgeLifecycleMixin
from teb_turn_supervisor_lifecycle import TebTurnSupervisorLifecycleMixin


class LifecycleFailureCleanupTest(unittest.TestCase):
    def test_bridge_failure_cancels_active_action(self):
        bridge = TebGoalBridgeLifecycleMixin()
        bridge.lifecycle_manager = SimpleNamespace(current_transaction_id=7)
        bridge.active_action = True
        bridge.cancelled = []
        bridge.cancel_locked = lambda reason: bridge.cancelled.append(reason)
        bridge.statuses = []
        bridge.publish_bridge_status = lambda name, **fields: bridge.statuses.append(
            (name, fields)
        )

        bridge._on_lifecycle_transition(State.DISPATCHED, State.FAILED, None)

        self.assertEqual(bridge.cancelled, ["lifecycle_failed"])

    def test_turn_failure_releases_turning_action(self):
        supervisor = TebTurnSupervisorLifecycleMixin()
        supervisor.lifecycle_manager = SimpleNamespace(current_transaction_id=8)
        supervisor.state = "TURNING"
        supervisor.active_action = True
        supervisor.released = []
        supervisor._release_turn_locked = lambda reason, completed=False: (
            supervisor.released.append((reason, completed))
        )
        supervisor.statuses = []
        supervisor.publish_status_locked = lambda name, **fields: supervisor.statuses.append(
            (name, fields)
        )

        supervisor._on_lifecycle_transition(State.DISPATCHED, State.FAILED, None)

        self.assertEqual(supervisor.released, [("lifecycle_failed", False)])
        self.assertFalse(supervisor.active_action)


if __name__ == "__main__":
    unittest.main()
