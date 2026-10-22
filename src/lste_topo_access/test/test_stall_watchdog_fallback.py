#!/usr/bin/env python3
"""Regression tests for stall watchdog fallback mechanisms.

Verifies:
1. Level 1: Motion / Controller Execution Stall:
   - When controller_owned_route_failure=True, ordinary stall waits for controller terminal.
   - When hard_stall_timeout is exceeded with actuator forwarding, hard_stalled breaks the
     lease with 'hard_controller_stall_timeout'.
   - When actuator is not forwarding (teleop/hold), hard stall is suppressed.
   - Deactivating a hard-stalled route marks the active WorkItem as dead corner and does
     not block on awaiting controller release ACK.
2. Level 2: Executive Planning Stall Watchdog:
   - When active_frontier is None and unresolved WorkItems exist, exceeding
     planning_stall_timeout triggers watchdog recovery, reconciles orphaned attempts,
     clears exclusions, and requests a decision wake.
   - Repeated planning stalls resolve the stuck WorkItem as dead corner.
"""

from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from global_frontier_execution_watchdogs import GlobalFrontierExecutionWatchdogMixin
from global_frontier_execution_resolution import GlobalFrontierExecutionResolutionMixin
from global_frontier_planning_runtime import GlobalFrontierPlanningRuntimeMixin
from global_frontier_route_lease import (
    AUTHORITY_GLOBAL_WATCHDOG,
    AUTHORITY_CONTROLLER_TERMINAL,
    route_lease_failure_decision,
)
from global_frontier_work_items import (
    PlaceWorkItemLedger,
    WORK_RESOLVED,
    WORK_UNRESOLVED,
)


class MockExplorer(
    GlobalFrontierExecutionWatchdogMixin,
    GlobalFrontierExecutionResolutionMixin,
    GlobalFrontierPlanningRuntimeMixin,
):
    def __init__(self):
        self.active_route_id = 42
        self.active_route_kind = "frontier_endpoint"
        self.active_since = 100.0
        self.active_progress_time = 100.0
        self.stall_timeout = 8.0
        self.hard_stall_timeout = 14.0
        self.planning_stall_timeout = 6.0
        self.active_timeout = 30.0
        self.post_turn_stall_timeout = 4.0
        self.rejected_timeout = 60.0
        self.active_best_path_distance = None
        self.active_best_goal_distance = None
        self.active_best_detour_odom_distance = 0.0
        self.active_visited_odom_cells = set()
        self.active_last_progress_signal = "none"
        self.recovery_pending_route_id = 0
        self.recovery_pending_reason = ""
        self.turn_supervisor_state = "IDLE"
        self.turn_connector_released = True
        self.active_turn_completed_route_id = 0
        self.active_turn_completed_goal = None
        self.active_turn_completed_launched = True
        self.active_turn_completed_translation = 0.0
        self.active_portal_gate_approached_at = None
        self.target_region_claim_active = False
        self.active_observation_session_started_at = None
        self.active_frontier_region_id = 1
        self.active_last_waypoint_map = (1.0, 1.0)
        self.waypoint_release_radius = 0.5
        self.controller_owned_route_failure = True
        self.persistent_execution = True
        self.mission_endpoint_only = True
        self.last_suppressed_stall_route_id = 0
        self.last_route_stagnation_reported_route_id = 0
        self.route_actuator = SimpleNamespace(
            forwarding=lambda: True,
            diagnostic=lambda: {
                "route_actuator_selected_mode": "autonomous",
                "route_actuator_block_reason": "none",
            },
        )
        self.region_memory = SimpleNamespace(
            stagnant=lambda *args, **kwargs: None,
            retain_for_local_egress=lambda *args, **kwargs: {"id": 1},
        )
        self.rejected_frontiers = []
        self.events = []
        self.place_work_items = PlaceWorkItemLedger(0.5)
        self.active_work_item_id = None
        self.active_work_item_attempt_id = None
        self.active_frontier = None
        self.prefetched_frontier = None
        self.costmap_cleared = False
        self.decision_wake_scheduler = SimpleNamespace(
            request=lambda reason: self.events.append(("wake_requested", reason))
        )

    def publish_status(self, event, **fields):
        self.events.append((event, fields))

    def clear_move_base_costmaps(self):
        self.costmap_cleared = True
        return True

    def record_failed_frontier_region(self, *args, **kwargs):
        return {"id": 1, "state": "failed", "failures": 1}

    def prepare_local_egress(self, *args, **kwargs):
        return False

    def release_active_frontier(self, **kwargs):
        self.events.append(("release_active_frontier", kwargs))
        self.active_frontier = None
        self.prefetched_frontier = None
        self.active_route_id = 0


class StallWatchdogFallbackTest(unittest.TestCase):
    def test_ordinary_stall_vs_hard_stall_evaluation(self):
        explorer = MockExplorer()
        now = 109.0  # elapsed 9.0s > stall_timeout (8.0s), but < hard_stall_timeout (14.0s)
        state = explorer.evaluate_active_route_watchdogs(
            x=2.0, y=2.0, robot_map=(0.0, 0.0), now=now,
            portal_transition=False, active_component=None, active_information=0.0
        )
        self.assertTrue(state.stalled)
        self.assertFalse(state.hard_stalled)
        self.assertFalse(state.expired)

        # Now advance past hard_stall_timeout (elapsed 15.0s > 14.0s)
        now = 115.0
        state = explorer.evaluate_active_route_watchdogs(
            x=2.0, y=2.0, robot_map=(0.0, 0.0), now=now,
            portal_transition=False, active_component=None, active_information=0.0
        )
        self.assertTrue(state.stalled)
        self.assertTrue(state.hard_stalled)
        self.assertTrue(state.expired)
        self.assertEqual(state.hard_stall_reason, "hard_controller_stall_timeout")

    def test_hard_stall_suppressed_when_actuator_not_forwarding(self):
        explorer = MockExplorer()
        explorer.route_actuator = SimpleNamespace(
            forwarding=lambda: False,
            diagnostic=lambda: {
                "route_actuator_selected_mode": "teleop",
                "route_actuator_block_reason": "source_not_selected",
            },
        )
        now = 120.0  # 20s elapsed
        state = explorer.evaluate_active_route_watchdogs(
            x=2.0, y=2.0, robot_map=(0.0, 0.0), now=now,
            portal_transition=False, active_component=None, active_information=0.0
        )
        self.assertFalse(state.hard_stalled)
        self.assertFalse(state.expired)
        self.assertTrue(state.stall_suppressed_by_actuator)

    def test_deactivate_failed_active_frontier_hard_stall_recovery(self):
        explorer = MockExplorer()
        now = 120.0
        item_id = explorer.place_work_items.dispatch(
            1, (5.0, 5.0), now=100.0
        )
        item = explorer.place_work_items.get(item_id)
        attempt_id = item["active_attempt_id"]
        explorer.active_work_item_id = item_id
        explorer.active_work_item_attempt_id = attempt_id

        watchdog = SimpleNamespace(
            recovery_pending=False,
            hard_stalled=True,
            hard_stall_reason="hard_controller_stall_timeout",
            post_turn_stalled=False,
            post_turn_goal_matches=False,
            post_turn_elapsed=0.0,
            post_turn_translation=0.0,
            stalled=True,
        )
        reason = explorer.active_route_failure_reason(watchdog)
        self.assertEqual(reason, "hard_controller_stall_timeout")

        explorer.deactivate_failed_active_frontier(
            x=5.0, y=5.0, distance=3.0, now=now,
            portal_transition=False, active_component=None, route_distance=4.0,
            reason=reason, watchdog=watchdog
        )

        # WorkItem should be resolved as dead corner
        resolved_item = explorer.place_work_items.get(item_id)
        self.assertEqual(resolved_item["state"], WORK_RESOLVED)
        self.assertTrue(
            "dead_corner" in str(resolved_item.get("last_reason", ""))
        )
        # Costmaps should be cleared
        self.assertTrue(explorer.costmap_cleared)
        # release_active_frontier called with await_controller_release_ack=False
        release_calls = [c for c in explorer.events if c[0] == "release_active_frontier"]
        self.assertTrue(len(release_calls) > 0)
        self.assertFalse(release_calls[0][1].get("await_controller_release_ack", True))

    def test_planning_stall_watchdog_auto_recovery(self):
        explorer = MockExplorer()
        explorer.active_frontier = None
        now = 100.0

        # Create an unresolved WorkItem with a dangling attempt
        item_id = explorer.place_work_items.dispatch(
            1, (3.0, 3.0), now=now
        )
        item = explorer.place_work_items.get(item_id)
        self.assertIsNotNone(item["active_attempt_id"])

        # Initial call sets baseline wall clock
        explorer._evaluate_planning_stall_watchdog(now)
        self.assertEqual(len([e for e in explorer.events if e[0] == "planning_stall_watchdog_triggered"]), 0)

        # Advance time by 7.0s (> planning_stall_timeout=6.0s)
        explorer._last_active_frontier_seen_wall = time.monotonic() - 7.0
        explorer._evaluate_planning_stall_watchdog(now + 7.0)

        # Watchdog should trigger!
        triggered = [e for e in explorer.events if e[0] == "planning_stall_watchdog_triggered"]
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0][1]["stall_count"], 1)

        # Dangling attempt on WorkItem should be reconciled
        updated_item = explorer.place_work_items.get(item_id)
        self.assertIsNone(updated_item["active_attempt_id"])
        self.assertEqual(updated_item["state"], WORK_UNRESOLVED)

        # Costmaps cleared and decision wake scheduler requested
        self.assertTrue(explorer.costmap_cleared)
        wakes = [e for e in explorer.events if e[0] == "wake_requested"]
        self.assertEqual(len(wakes), 1)
        self.assertEqual(wakes[0][1], "planning_stall_watchdog_recovery")

    def test_repeated_planning_stall_resolves_stuck_item(self):
        explorer = MockExplorer()
        explorer.active_frontier = None
        now = 100.0

        item = explorer.place_work_items.register_target_work_item(
            1, (3.0, 3.0), now
        )
        item_id = item["id"]

        # First trigger
        explorer._last_active_frontier_seen_wall = time.monotonic() - 7.0
        explorer._evaluate_planning_stall_watchdog(now + 7.0)
        self.assertEqual(explorer.place_work_items.get(item_id)["state"], WORK_UNRESOLVED)

        # Second trigger (repeated stall count >= 2)
        explorer._last_active_frontier_seen_wall = time.monotonic() - 7.0
        explorer._evaluate_planning_stall_watchdog(now + 14.0)

        # Second trigger should resolve the stuck item as dead corner to break deadlock
        resolved_item = explorer.place_work_items.get(item_id)
        self.assertEqual(resolved_item["state"], WORK_RESOLVED)
        self.assertEqual(
            resolved_item.get("last_reason"),
            "dead_corner_planning_stall_exhausted",
        )


if __name__ == "__main__":
    unittest.main()
