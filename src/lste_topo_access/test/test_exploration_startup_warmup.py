#!/usr/bin/env python3
"""Unit tests for the online exploration startup warmup mechanism.

Validates that during the initial startup warmup window:
1. No WorkItems are minted (preventing early truncated-scan SLAM phantom artifacts).
2. No exploration routes are dispatched.
3. Status is published as 'startup_warmup_active' with remaining time.
4. Once warmup expires, normal WorkItem minting and route selection resume.
"""

from collections import deque
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

rospy_stub = sys.modules.setdefault("rospy", SimpleNamespace())
if not hasattr(rospy_stub, "loginfo"):
    rospy_stub.loginfo = lambda *args, **kwargs: None
if not hasattr(rospy_stub, "logwarn"):
    rospy_stub.logwarn = lambda *args, **kwargs: None
if not hasattr(rospy_stub, "logerr"):
    rospy_stub.logerr = lambda *args, **kwargs: None
if not hasattr(rospy_stub, "get_param"):
    rospy_stub.get_param = lambda name, default=None: default

from global_frontier_runtime_state import GlobalFrontierRuntimeStateMixin
from global_frontier_route_state import GlobalFrontierRouteStateMixin
from global_frontier_selection_context import GlobalFrontierSelectionContextMixin
from global_frontier_selection_planner import GlobalFrontierSelectionPlannerMixin
from global_frontier_planning_selection import GlobalFrontierPlanningSelectionMixin
from global_frontier_planning_runtime import GlobalFrontierPlanningRuntimeMixin
from global_frontier_work_items import PlaceWorkItemLedger


class RuntimeStateProbe(GlobalFrontierRuntimeStateMixin):
    def __init__(self, warmup_seconds=8.0):
        self.exploration_startup_warmup_seconds = float(warmup_seconds)
        self.first_map_received_time = None
        self.last_startup_warmup_status_wall = 0.0
        self.active_frontier = None


class SelectionContextProbe(
    GlobalFrontierRuntimeStateMixin,
    GlobalFrontierSelectionContextMixin,
):
    def __init__(self, warmup_seconds=8.0):
        self.exploration_startup_warmup_seconds = float(warmup_seconds)
        self.first_map_received_time = 100.0
        self.last_startup_warmup_status_wall = 0.0
        self.active_frontier = None
        self.place_work_items = PlaceWorkItemLedger(1.25)
        self.work_item_memory_enabled = True
        self.target_region_claim_active = False
        self.region_memory = SimpleNamespace(
            by_id=lambda _id: {"id": 1, "state": "open", "endpoint_observations": 1},
            dormant_component_labels=lambda _epoch: set(),
            observed_open_component_labels=lambda _epoch: set(),
            candidate_tier=lambda *args, **kwargs: ("open", {"id": 1, "state": "open"}),
        )
        self.current_physical_place_id = 1
        self.last_work_item_created = 0
        self.last_work_item_inherited = 0
        self.last_work_item_resolved_descendant_skips = 0
        self.last_work_item_density_absorbed = 0
        self.last_sealed_portal_reentry_skips = 0

    @staticmethod
    def cell_xy(_message, row, col):
        return float(col) * 0.1, float(row) * 0.1

    @staticmethod
    def component_at_map_position(_message, _components, _known_free, _x, _y):
        return {"label": 1, "center_x": 0.0, "center_y": 0.0}

    @staticmethod
    def covered_place_observation_footprint(*args, **kwargs):
        return np.zeros((10, 10), dtype=bool), 0


class PlannerProbe(
    GlobalFrontierRuntimeStateMixin,
    GlobalFrontierSelectionPlannerMixin,
):
    def __init__(self, warmup_seconds=8.0):
        self.exploration_startup_warmup_seconds = float(warmup_seconds)
        self.first_map_received_time = 100.0
        self.last_startup_warmup_status_wall = 0.0
        self.active_frontier = None


class PlanningSelectionProbe(
    GlobalFrontierRuntimeStateMixin,
    GlobalFrontierPlanningSelectionMixin,
):
    def __init__(self, warmup_seconds=8.0):
        self.exploration_startup_warmup_seconds = float(warmup_seconds)
        self.first_map_received_time = 100.0
        self.last_startup_warmup_status_wall = 0.0
        self.active_frontier = None


class TimerRuntimeProbe(
    GlobalFrontierRuntimeStateMixin,
    GlobalFrontierPlanningRuntimeMixin,
):
    def __init__(self, warmup_seconds=8.0):
        self.exploration_startup_warmup_seconds = float(warmup_seconds)
        self.first_map_received_time = 100.0
        self.last_startup_warmup_status_wall = 0.0
        self.active_frontier = None
        self.published_events = []
        self.snapshot_built = False
        self.route_dispatched = False
        self.mock_now = 102.0

    def planning_pose_snapshot(self):
        msg = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            info=SimpleNamespace(width=50, height=50, resolution=0.1),
        )
        return msg, (1.0, 1.0), 0.0, self.mock_now

    def publish_status(self, event, **fields):
        self.published_events.append({"event": event, **fields})

    def build_frontier_planning_snapshot(self, *args, **kwargs):
        self.snapshot_built = True
        return None

    def active_route_planning_is_deferred(self, *args, **kwargs):
        return False


class ExplorationStartupWarmupTest(unittest.TestCase):
    def test_warmup_predicate_lifecycle(self):
        """Warmup is active only from first map receipt until warmup_seconds elapse."""
        probe = RuntimeStateProbe(warmup_seconds=8.0)

        # Before any map is received, warmup is active
        self.assertTrue(probe.is_startup_warmup_active(now=10.0))

        # First map arrives at t = 100.0
        probe.first_map_received_time = 100.0

        # During warmup window [100.0, 108.0)
        self.assertTrue(probe.is_startup_warmup_active(now=100.0))
        self.assertTrue(probe.is_startup_warmup_active(now=103.5))
        self.assertTrue(probe.is_startup_warmup_active(now=107.99))

        # At and after warmup expiration (t >= 108.0)
        self.assertFalse(probe.is_startup_warmup_active(now=108.0))
        self.assertFalse(probe.is_startup_warmup_active(now=115.0))

    def test_warmup_disabled_when_zero_seconds(self):
        """When warmup_seconds is configured to 0.0, warmup is never active."""
        probe = RuntimeStateProbe(warmup_seconds=0.0)
        probe.first_map_received_time = 100.0
        self.assertFalse(probe.is_startup_warmup_active(now=100.0))
        self.assertFalse(probe.is_startup_warmup_active(now=101.0))

    def test_warmup_inactive_if_active_frontier_already_exists(self):
        """Warmup is strictly a startup gate; an active frontier suppresses it."""
        probe = RuntimeStateProbe(warmup_seconds=8.0)
        probe.first_map_received_time = 100.0
        probe.active_frontier = (0, 0, 1.0, 2.0)
        self.assertFalse(probe.is_startup_warmup_active(now=102.0))

    def test_first_map_receipt_records_timestamp(self):
        """apply_map sets first_map_received_time only on the first valid map."""
        probe = GlobalFrontierRouteStateMixin()
        probe.first_map_received_time = None

        map_msg = SimpleNamespace(
            info=SimpleNamespace(width=100, height=100, resolution=0.05)
        )
        with patch("global_frontier_route_state.now_for", return_value=50.25):
            probe.apply_map(map_msg)

        self.assertEqual(probe.first_map_received_time, 50.25)

        # Subsequent maps do not overwrite the initial timestamp
        with patch("global_frontier_route_state.now_for", return_value=55.0):
            probe.apply_map(map_msg)

        self.assertEqual(probe.first_map_received_time, 50.25)

    def test_selection_context_prohibits_work_item_minting_during_warmup(self):
        """_prepare_frontier_selection_context must not mint WorkItems while warmup is active."""
        probe = SelectionContextProbe(warmup_seconds=8.0)

        # Synthetic map grid with unknown frontier
        grid_size = 20
        message = SimpleNamespace(
            info=SimpleNamespace(width=grid_size, height=grid_size, resolution=0.1)
        )
        steps = np.ones((grid_size, grid_size), dtype=np.int32) * 5
        frontier = np.zeros((grid_size, grid_size), dtype=bool)
        frontier[10, 10:15] = True
        unknown = np.zeros((grid_size, grid_size), dtype=bool)
        unknown[11:, :] = True
        occupied = np.zeros((grid_size, grid_size), dtype=bool)
        components = SimpleNamespace(
            epoch=1,
            labels=np.ones((grid_size, grid_size), dtype=np.int32),
        )

        # Call during warmup (now=102.0, first_map=100.0, warmup=8.0)
        context = probe._prepare_frontier_selection_context(
            message=message,
            steps=steps,
            frontier=frontier,
            unknown=unknown,
            occupied=occupied,
            route_anchor_xy=(0.5, 0.5),
            components=components,
            now=102.0,
        )

        # Must have zero WorkItems minted
        self.assertEqual(len(context.work_item_cells), 0)
        self.assertEqual(len(context.work_item_details), 0)
        self.assertEqual(probe.last_work_item_created, 0)
        self.assertEqual(len(probe.place_work_items._items), 0)

        # After warmup expires (now=109.0)
        context_after = probe._prepare_frontier_selection_context(
            message=message,
            steps=steps,
            frontier=frontier,
            unknown=unknown,
            occupied=occupied,
            route_anchor_xy=(0.5, 0.5),
            components=components,
            now=109.0,
        )

        # WorkItems are now minted normally
        self.assertGreater(len(context_after.work_item_details), 0)
        self.assertGreater(probe.last_work_item_created, 0)
        self.assertGreater(len(probe.place_work_items._items), 0)

    def test_choose_frontier_returns_none_during_warmup(self):
        """choose_frontier returns None during warmup without running selection."""
        probe = PlannerProbe(warmup_seconds=8.0)
        result = probe.choose_frontier(
            message=SimpleNamespace(),
            steps=np.zeros((10, 10)),
            frontier=np.ones((10, 10), dtype=bool),
            unknown=np.zeros((10, 10), dtype=bool),
            occupied=np.zeros((10, 10), dtype=bool),
            now=103.0,  # during warmup [100, 108)
        )
        self.assertIsNone(result)

    def test_select_next_active_frontier_returns_warmup_wait(self):
        """select_next_active_frontier returns startup_warmup_wait during warmup."""
        probe = PlanningSelectionProbe(warmup_seconds=8.0)
        snapshot = SimpleNamespace(now=104.0)  # during warmup [100, 108)
        candidate, mode, wait = probe.select_next_active_frontier(snapshot)
        self.assertIsNone(candidate)
        self.assertEqual(mode, "startup_warmup_wait")
        self.assertTrue(wait)

    def test_on_timer_publishes_status_and_suppresses_dispatch_during_warmup(self):
        """_on_timer publishes startup_warmup_active and does not build snapshot or dispatch."""
        probe = TimerRuntimeProbe(warmup_seconds=8.0)
        probe.mock_now = 103.0  # elapsed 3.0s, remaining 5.0s

        probe._on_timer(None)

        # Snapshot was not built and route was not dispatched
        self.assertFalse(probe.snapshot_built)
        self.assertFalse(probe.route_dispatched)

        # Status event was published
        warmup_events = [
            e for e in probe.published_events
            if e.get("event") == "startup_warmup_active"
        ]
        self.assertEqual(len(warmup_events), 1)
        event = warmup_events[0]
        self.assertEqual(event.get("status"), "startup_warmup_active")
        self.assertEqual(event.get("warmup_seconds"), 8.0)
        self.assertAlmostEqual(event.get("remaining_seconds"), 5.0, places=1)


if __name__ == "__main__":
    unittest.main()
