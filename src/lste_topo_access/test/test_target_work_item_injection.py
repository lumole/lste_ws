#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests verifying that visual target detection injects high-priority Target WorkItems."""

import unittest
from unittest.mock import patch
from pathlib import Path
import sys
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from global_frontier_work_items import PlaceWorkItemLedger, WORK_UNRESOLVED, WORK_RESOLVED
from global_frontier_work_item_tour import order_work_items_open_tsp, is_target_work_item
from global_frontier_graph_route_planner import GraphRoutePlanner, ACTION_REINSPECT_TARGET
from global_frontier_graph_route_gate import candidate_matches_graph_plan
from goal_manager_input_callbacks import GoalManagerInputCallbacksMixin
from goal_manager_target_completion import GoalManagerTargetCompletionMixin
from goal_manager_target_follow import GoalManagerTargetFollowMixin
from goal_manager_modes import (
    CATCH_CTX_MODE,
    CATCH_TARGET_MODE,
    EXPLORE_PASS_MODE,
    STATE_LOCKED,
    STATE_SUSPICIOUS,
)
from lste_msgs.msg import LsteState


class TargetWorkItemInjectionTest(unittest.TestCase):
    def setUp(self):
        self.ledger = PlaceWorkItemLedger(merge_radius=0.4, density_radius=0.8)

    def test_register_and_update_target_work_item(self):
        # Register a target work item
        item = self.ledger.register_target_work_item(
            place_id=1,
            anchor_xy=(5.0, 3.0),
            now=100.0,
            target_track_id="cup_1",
        )
        self.assertIsNotNone(item)
        self.assertTrue(item.get("is_target"))
        self.assertEqual(item.get("kind"), "target_observation")
        self.assertEqual(item.get("anchor_xy"), (5.0, 3.0))
        self.assertEqual(item.get("target_track_id"), "cup_1")
        self.assertEqual(item.get("spawn_distance"), 0.0)
        self.assertEqual(item.get("state"), WORK_UNRESOLVED)

        # Update target work item with refined coordinates (e.g. from triangulation)
        updated = self.ledger.register_target_work_item(
            place_id=1,
            anchor_xy=(5.2, 3.1),
            now=101.0,
            target_track_id="cup_1",
        )
        self.assertEqual(item["id"], updated["id"])
        self.assertEqual(updated.get("anchor_xy"), (5.2, 3.1))
        self.assertEqual(len(self.ledger._items), 1)

    def test_target_work_item_immunity_to_suppression_and_dead_corners(self):
        # Register target work item
        target_item = self.ledger.register_target_work_item(
            place_id=1,
            anchor_xy=(2.0, 2.0),
            now=100.0,
        )
        # Register a regular frontier work item nearby
        frontier_item = self.ledger._new_item(
            place_id=1,
            support_cells=frozenset([(10, 10), (10, 11)]),
            now=100.0,
            anchor_xy=(2.1, 2.0),
        )

        # Spatial suppression must NOT merge or suppress the target item
        suppressed = self.ledger.suppress_dense_unresolved_items(place_id=1)
        self.assertEqual(target_item["state"], WORK_UNRESOLVED)

        # Dead corner marking near target item must NOT resolve the target item
        self.ledger.mark_dead_corner((2.0, 2.0))
        self.assertEqual(target_item["state"], WORK_UNRESOLVED)

        # Resolving target work items upon completion resolves it
        resolved = self.ledger.resolve_target_work_items(now=105.0, reason="task_done")
        self.assertIn(target_item["id"], resolved)
        self.assertEqual(target_item["state"], WORK_RESOLVED)

    def test_open_tsp_strict_target_precedence(self):
        robot_pose = (0.0, 0.0, 0.0)
        # Regular frontier work items
        frontier_w1 = {"id": 1, "anchor_xy": (1.0, 0.0), "spawn_distance": 1.0}
        frontier_w2 = {"id": 2, "anchor_xy": (2.0, 0.0), "spawn_distance": 2.0}
        frontier_w3 = {"id": 3, "anchor_xy": (3.0, 0.0), "spawn_distance": 3.0}

        # Target work item located further away (e.g. 5.0m)
        target_wt = {
            "id": 99,
            "anchor_xy": (5.0, 0.0),
            "is_target": True,
            "kind": "target_observation",
            "spawn_distance": 0.0,
        }

        # Regardless of distance, Target WorkItem MUST be placed at index 0!
        work_items = [frontier_w1, frontier_w2, frontier_w3, target_wt]
        tour = order_work_items_open_tsp(robot_pose, work_items)
        self.assertEqual(tour[0]["id"], 99)
        self.assertTrue(is_target_work_item(tour[0]))

    @patch("rospy.Time.now", return_value=SimpleNamespace(to_sec=lambda: 100.0))
    def test_goal_manager_suspicion_breaks_context_hold_and_replan(self, _mock_now):
        class DummyGoalManager(GoalManagerInputCallbacksMixin):
            def __init__(self):
                self.current_state = 0
                self.current_subtype = ""
                self.latest_state_msg = None
                self.locked_enter_time = None
                self.target_done_require_locked = False
                self.base_mode = CATCH_CTX_MODE
                self.effective_mode = CATCH_CTX_MODE
                self.ctx_state_hold_until = 9999.0  # Context hold active
                self.context_state_hold_time = 3.0
                self.last_state = None
                self.replan_requests = []
                self.target_hypothesis_xy = (6.0, 2.5)
                self.target_track_id = "yellow_cup:1"

            def target_tracking_active(self, _now):
                return False  # Target tracking not yet active across frames

            def mode_from_state(self, state, subtype):
                if state == STATE_LOCKED:
                    return CATCH_TARGET_MODE
                if state == STATE_SUSPICIOUS and subtype == "Sus-A":
                    return CATCH_TARGET_MODE
                if state == STATE_SUSPICIOUS and subtype == "Sus-B":
                    return CATCH_CTX_MODE
                return EXPLORE_PASS_MODE

            def reset_target_close_confirmation(self):
                pass

            def request_global_frontier_replan(self, reason, **fields):
                self.replan_requests.append((reason, fields))
                return len(self.replan_requests)

        manager = DummyGoalManager()
        state_msg = LsteState(state=STATE_SUSPICIOUS, subtype="Sus-A", task_id="yellow_cup")
        manager.apply_state(state_msg)

        # Context hold must be cleared to 0.0 immediately
        self.assertEqual(manager.ctx_state_hold_until, 0.0)
        # Effective mode must switch to CATCH_TARGET_MODE
        self.assertEqual(manager.effective_mode, CATCH_TARGET_MODE)
        # Replan request must be issued with target coordinates
        self.assertEqual(len(manager.replan_requests), 1)
        reason, fields = manager.replan_requests[0]
        self.assertEqual(reason, "target_suspicion_preemption")
        self.assertTrue(fields.get("target_pursuit"))
        self.assertEqual(fields.get("semantic_hint_map"), [6.0, 2.5])

    def test_reconcile_orphaned_attempts_unblocks_planning(self):
        # Create an item and dispatch an attempt
        item = self.ledger._new_item(
            place_id=1,
            support_cells=frozenset([(1, 1)]),
            now=100.0,
            anchor_xy=(1.0, 1.0),
        )
        self.ledger.dispatch_existing(
            item_id=item["id"],
            now=100.0,
            viewpoint_xy=(0.5, 0.5),
        )
        self.assertIsNotNone(item.get("active_attempt_id"))
        self.assertEqual(len([it for it in self.ledger.snapshot() if it.get("active_attempt_id") is not None]), 1)

        # Preemption occurs: route is cancelled, no active route executing item
        reconciled = self.ledger.reconcile_orphaned_attempts(now=101.0, active_work_item_id=None)
        self.assertIn(item["id"], reconciled)
        self.assertIsNone(item.get("active_attempt_id"))

        # In snapshot(), active_attempt_id is None, so GraphRoutePlanner is not blocked
        snap = self.ledger.snapshot()
        self.assertTrue(all(it.get("active_attempt_id") is None for it in snap))

    def test_locked_target_work_item_plan_and_gate_matching(self):
        # Register a target item and regular frontier items
        target_item = self.ledger.register_target_work_item(
            place_id=1,
            anchor_xy=(19.9, -3.6),
            now=100.0,
            target_track_id="cup:1",
        )
        frontier_item = self.ledger._new_item(
            place_id=1,
            support_cells=frozenset([(1, 1)]),
            now=100.0,
            anchor_xy=(10.0, -2.0),
        )
        local_work = [target_item, frontier_item]
        planner = GraphRoutePlanner()
        plan = planner.plan(
            current_place_id=1,
            places=[{"id": 1, "state": "observed"}],
            portals=[],
            probes=[],
            work_items=local_work,
            branch_first=True,
        )
        # Plan MUST be for reinspect_target with obligation_id == target_item["id"]
        self.assertEqual(plan.action, ACTION_REINSPECT_TARGET)
        self.assertEqual(plan.obligation_id, target_item["id"])
        self.assertEqual(plan.work_item_tour, (target_item["id"],))

        # Test candidate matching: regular candidate must NOT match the plan
        regular_route = (
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,  # [9] hops
            "frontier_endpoint",  # [10]
            0,  # [11]
            frontier_item["id"],  # [12] work_item_id
            0, 0, None, 0,
            "observe_local_work",  # [17] action
        )
        target_route = (
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,  # [9] hops
            "frontier_endpoint",  # [10]
            0,  # [11]
            target_item["id"],  # [12] work_item_id
            0, 0, None, 0,
            "reinspect_target",  # [17] action
        )
        self.assertFalse(candidate_matches_graph_plan(plan, SimpleNamespace(route=regular_route)))
        self.assertTrue(candidate_matches_graph_plan(plan, SimpleNamespace(route=target_route)))

    @patch("rospy.Time.now", return_value=SimpleNamespace(to_sec=lambda: 100.0))
    def test_locked_state_confirms_target_and_bypasses_room_search(self, _mock_now):
        class CombinedGoalManager(GoalManagerInputCallbacksMixin, GoalManagerTargetCompletionMixin, GoalManagerTargetFollowMixin):
            def __init__(self):
                self.current_state = 0
                self.current_subtype = ""
                self.latest_state_msg = None
                self.locked_enter_time = None
                self.target_done_require_locked = False
                self.target_track_id = "yellow_cup:1"
                self.target_approach_track_id = ""
                self.target_follow_confirmed = False
                self.base_mode = CATCH_TARGET_MODE
                self.effective_mode = CATCH_TARGET_MODE
                self.ctx_state_hold_until = 0.0
                self.context_state_hold_time = 3.0
                self.last_state = None
                self.replan_requests = []
                self.target_hypothesis_xy = (19.9, -3.6)
                self.target_last_goal = None
                self.target_candidate_room_claim_requested = True
                self.fresh_frontier_called = False

            def target_tracking_active(self, _now):
                return True

            def mode_from_state(self, state, subtype):
                return CATCH_TARGET_MODE

            def reset_target_close_confirmation(self):
                pass

            def request_global_frontier_replan(self, reason, **fields):
                self.replan_requests.append((reason, fields))
                return len(self.replan_requests)

            def fresh_global_frontier_goal(self, now):
                self.fresh_frontier_called = True
                return None

            def _target_parallax_observation_goal(self, now):
                return None

        mgr = CombinedGoalManager()
        state_msg = LsteState(state=STATE_LOCKED, subtype="", task_id="yellow_cup")
        mgr.apply_state(state_msg)

        # 1. Target follow must be confirmed immediately upon LOCKED
        self.assertTrue(mgr.target_follow_confirmed)
        self.assertEqual(mgr.target_approach_track_id, "yellow_cup:1")

        # 2. Target close completion must be eligible
        self.assertTrue(mgr.target_close_completion_eligible())

        # 3. Unconfirmed target candidate handler must NOT trigger room search
        handled, goal = mgr._handle_unconfirmed_target_candidate(100.0)
        self.assertFalse(handled)
        self.assertIsNone(goal)
        self.assertFalse(mgr.fresh_frontier_called)


if __name__ == "__main__":
    unittest.main()
