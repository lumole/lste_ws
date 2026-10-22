#!/usr/bin/env python3

"""Tests for WorkItem open generation, density control (spatial NMS), and dead corner abandonment."""

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

sys.modules.setdefault(
    "rospy",
    SimpleNamespace(
        logwarn=lambda *args, **kwargs: None,
        logwarn_throttle=lambda *args, **kwargs: None,
        Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 10.0)),
    ),
)

from global_frontier_work_item_components import (
    ObservationSupport,
    check_anchor_clearance_and_corner,
    observation_supports,
    suppress_dense_observation_supports,
)
from global_frontier_work_items import (
    PlaceWorkItemLedger,
    WORK_UNRESOLVED,
    WORK_RESOLVED,
)
from global_frontier_candidate_work_items import GlobalFrontierCandidateWorkItemMixin


class WorkItemDensityAndCornerRejectionTest(unittest.TestCase):
    """Verify open generation, spatial density suppression, and dead corner abandonment."""

    def test_corner_check_admits_open_frontier(self):
        """Open frontier facing unmapped space is admitted."""
        rows, cols = 40, 40
        resolution = 0.05
        occupied = np.zeros((rows, cols), dtype=bool)
        unknown = np.ones((rows, cols), dtype=bool)
        known_free = np.zeros((rows, cols), dtype=bool)

        # Free space in lower half
        known_free[20:, :] = True
        unknown[20:, :] = False

        arc = frozenset((20, c) for c in range(15, 25))

        rejected, reason, metrics = check_anchor_clearance_and_corner(
            arc,
            unknown,
            known_free,
            occupied,
            resolution=resolution,
        )

        self.assertFalse(rejected)
        self.assertEqual(reason, "admitted")

    def test_corner_check_admits_frontier_in_corner(self):
        """Under open generation, a frontier in a wall corner is admitted upfront.

        Abandonment is deferred to runtime if the vehicle cannot reach it.
        """
        rows, cols = 40, 40
        resolution = 0.05
        occupied = np.zeros((rows, cols), dtype=bool)
        unknown = np.zeros((rows, cols), dtype=bool)
        known_free = np.zeros((rows, cols), dtype=bool)

        # Enclose corner walls
        occupied[0:5, :] = True
        occupied[:, 0:5] = True
        unknown[0:15, 0:15] = True
        unknown[occupied] = False
        known_free[15:, 15:] = True
        known_free[~occupied & ~unknown] = True

        # Arc near corner but outside solid obstacle
        arc = frozenset({(6, 6), (6, 7), (7, 6)})

        rejected, reason, metrics = check_anchor_clearance_and_corner(
            arc,
            unknown,
            known_free,
            occupied,
            resolution=resolution,
        )

        self.assertFalse(rejected)
        self.assertEqual(reason, "admitted")

    def test_corner_check_rejects_anchor_in_solid_obstacle(self):
        """An arc whose anchor centroid is embedded in a solid obstacle is rejected."""
        rows, cols = 40, 40
        resolution = 0.05
        occupied = np.zeros((rows, cols), dtype=bool)
        unknown = np.zeros((rows, cols), dtype=bool)
        known_free = np.zeros((rows, cols), dtype=bool)

        # Solid wall
        occupied[5:15, 5:15] = True
        arc = frozenset({(10, 10), (10, 11)})

        rejected, reason, metrics = check_anchor_clearance_and_corner(
            arc,
            unknown,
            known_free,
            occupied,
            resolution=resolution,
        )

        self.assertTrue(rejected)
        self.assertEqual(reason, "anchor_in_obstacle")

    def test_corner_check_rejects_empty_arc(self):
        """Empty arc is rejected."""
        rows, cols = 40, 40
        rejected, reason, _ = check_anchor_clearance_and_corner(
            frozenset(),
            np.zeros((rows, cols), dtype=bool),
            np.zeros((rows, cols), dtype=bool),
            np.zeros((rows, cols), dtype=bool),
            resolution=0.05,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "empty_arc")

    def test_spatial_nms_absorbs_dense_cluster_supports(self):
        """Dense cluster of supports within 0.8m is merged into 1 dominant support."""
        dominant = ObservationSupport(
            frontier_cells=frozenset({(10, 10), (10, 11), (10, 12)}),
            support_cells=frozenset({(20, 20), (20, 21), (20, 22), (20, 23)}),
            anchor_xy=(2.0, 2.0),
            normal_xy=(1.0, 0.0),
        )
        small_1 = ObservationSupport(
            frontier_cells=frozenset({(10, 14)}),
            support_cells=frozenset({(20, 24)}),
            anchor_xy=(2.15, 2.10),
            normal_xy=(0.95, 0.31),
        )
        small_2 = ObservationSupport(
            frontier_cells=frozenset({(10, 8)}),
            support_cells=frozenset({(20, 19)}),
            anchor_xy=(1.85, 1.95),
            normal_xy=(0.99, -0.1),
        )

        supports = (dominant, small_1, small_2)
        filtered = suppress_dense_observation_supports(supports, density_separation_m=0.8)

        self.assertEqual(len(filtered), 1)
        result = filtered[0]
        self.assertEqual(result.anchor_xy, (2.0, 2.0))
        self.assertEqual(result.normal_xy, (1.0, 0.0))
        self.assertEqual(
            result.frontier_cells,
            frozenset(dominant.frontier_cells | small_1.frontier_cells | small_2.frontier_cells),
        )

    def test_spatial_nms_keeps_distant_supports_separate(self):
        """Supports separated by >1.0m are not merged."""
        supp_1 = ObservationSupport(
            frontier_cells=frozenset((1, c) for c in range(10)),
            support_cells=frozenset((10, c) for c in range(15)),
            anchor_xy=(1.0, 1.0),
            normal_xy=(1.0, 0.0),
        )
        supp_2 = ObservationSupport(
            frontier_cells=frozenset((5, c) for c in range(10)),
            support_cells=frozenset((20, c) for c in range(15)),
            anchor_xy=(2.3, 1.0),  # 1.3m away (>1.0m)
            normal_xy=(1.0, 0.0),
        )

        filtered = suppress_dense_observation_supports((supp_1, supp_2), density_separation_m=1.0)
        self.assertEqual(len(filtered), 2)

    def test_spatial_nms_absorbs_tiny_fragment_within_extended_reach(self):
        """A tiny fragment (<=4 cells) within 1.5 * density_separation is absorbed."""
        dominant = ObservationSupport(
            frontier_cells=frozenset((1, c) for c in range(10)),
            support_cells=frozenset((10, c) for c in range(20)),
            anchor_xy=(1.0, 1.0),
            normal_xy=(1.0, 0.0),
        )
        fragment = ObservationSupport(
            frontier_cells=frozenset({(5, 5)}),
            support_cells=frozenset({(20, 20)}),
            anchor_xy=(2.2, 1.0),  # 1.2m away (<= 1.5 * 1.0m)
            normal_xy=(1.0, 0.0),
        )

        filtered = suppress_dense_observation_supports((dominant, fragment), density_separation_m=1.0)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].anchor_xy, (1.0, 1.0))
        self.assertIn((20, 20), filtered[0].support_cells)

    def test_ledger_reconcile_absorbs_nearby_unresolved_work_item(self):
        """PlaceWorkItemLedger absorbs incoming supports within density_radius into existing item."""
        ledger = PlaceWorkItemLedger(merge_radius=0.5, density_radius=0.8)
        first = ObservationSupport(
            frontier_cells=frozenset({(1, 1)}),
            support_cells=frozenset({(10, 10)}),
            anchor_xy=(1.0, 1.0),
            normal_xy=(1.0, 0.0),
        )
        first_cells, first_details = ledger.reconcile(4, (first,), now=1.0)
        first_id = first_cells[(1, 1)]
        self.assertEqual(first_details[first_id]["match"], "created")

        second = ObservationSupport(
            frontier_cells=frozenset({(1, 3)}),
            support_cells=frozenset({(10, 12)}),
            anchor_xy=(1.3, 1.1),  # dist ~ 0.316m <= 0.8m
            normal_xy=(0.0, 1.0),
        )

        second_cells, second_details = ledger.reconcile(4, (second,), now=2.0)
        self.assertEqual(second_cells[(1, 3)], first_id)
        self.assertEqual(second_details[first_id]["match"], "density_absorbed")
        self.assertEqual(len(ledger._items), 1)

    def test_ledger_suppress_dense_unresolved_items(self):
        """suppress_dense_unresolved_items consolidates redundant items within density_radius."""
        ledger = PlaceWorkItemLedger(merge_radius=0.3, density_radius=0.8)
        it1 = ledger._new_item(
            place_id=7,
            support_cells=frozenset({(10, 10), (10, 11)}),
            now=1.0,
            anchor_xy=(1.0, 1.0),
            normal_xy=(1.0, 0.0),
        )
        it2 = ledger._new_item(
            place_id=7,
            support_cells=frozenset({(10, 12)}),
            now=1.0,
            anchor_xy=(1.3, 1.1),
            normal_xy=(0.0, 1.0),
        )
        self.assertEqual(ledger.unresolved_count(7), 2)

        suppressed = ledger.suppress_dense_unresolved_items(place_id=7)
        self.assertEqual(suppressed, 1)
        self.assertEqual(ledger.unresolved_count(7), 1)
        self.assertEqual(it1["state"], WORK_UNRESOLVED)
        self.assertEqual(it2["state"], WORK_RESOLVED)

    def test_dead_corner_blacklist_prevents_recreation(self):
        """Dead corner abandonment blacklists anchor and prevents recreation on future reconcile."""
        ledger = PlaceWorkItemLedger(merge_radius=0.5, density_radius=1.0)
        item = ledger._new_item(
            place_id=1,
            support_cells=frozenset({(5, 5)}),
            now=1.0,
            anchor_xy=(3.0, 3.0),
            normal_xy=(1.0, 0.0),
        )
        item_id = item["id"]

        # Settle as abandoned dead corner
        resolved = ledger.resolve(item_id, now=2.0, reason="dead_corner_abandoned_controller_stall")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["state"], WORK_RESOLVED)

        # Verify anchor is blacklisted
        self.assertTrue(ledger.is_dead_corner((3.0, 3.0)))
        self.assertTrue(ledger.is_dead_corner((3.2, 3.1), radius=0.6))
        self.assertFalse(ledger.is_dead_corner((4.5, 4.5), radius=0.6))

        # New incoming frontier support at the same dead corner
        new_support = ObservationSupport(
            frontier_cells=frozenset({(5, 6)}),
            support_cells=frozenset({(20, 20)}),
            anchor_xy=(3.1, 3.0),  # 0.1m away from dead corner
            normal_xy=(1.0, 0.0),
        )

        cells, details = ledger.reconcile(1, (new_support,), now=3.0)
        # Must be completely ignored: no items created or matched
        self.assertEqual(len(cells), 0)
        self.assertEqual(len(details), 0)
        self.assertEqual(ledger.unresolved_count(1), 0)

    def test_candidate_work_items_abandons_unreachable_viewpoints_after_consecutive_failures(self):
        """A WorkItem whose viewpoints are unreachable across consecutive ticks is abandoned."""
        class MockOrigin:
            position = type("Pos", (), {"x": 0.0, "y": 0.0})()

        class MockInfo:
            resolution = 0.05
            origin = MockOrigin()

        class MockMessage:
            info = MockInfo()

        class MockRequest:
            message = MockMessage()
            # steps < 0 makes all cells unreachable
            steps = -np.ones((50, 50), dtype=np.int32)
            validation = None
            map_to_physical_xy = None

        class MockContext:
            source_place_id = 1
            known_free = np.ones((50, 50), dtype=bool)

        class CandidateTester(GlobalFrontierCandidateWorkItemMixin):
            def __init__(self):
                self.place_work_items = PlaceWorkItemLedger(merge_radius=0.5, density_radius=0.8)
                self.graph_route_work_item_filter = None
                self.graph_route_work_item_id = None
                self.portal_probe_ledger = None
                self.planning_should_preempt = lambda: False
                # Reassociate always returns cell where steps < 0
                self.nearest_reachable_cell = lambda msg, steps, x, y: (10, 10)
                self.candidate_costmap_distance = lambda val, x, y: None
                self.cell_xy = lambda msg, r, c: (c * 0.05, r * 0.05)

        tester = CandidateTester()
        it = tester.place_work_items._new_item(
            place_id=1,
            support_cells=frozenset({(1, 1)}),
            now=1.0,
            anchor_xy=(0.5, 0.5),
            normal_xy=(1.0, 0.0),
        )
        item_id = it["id"]

        req = MockRequest()
        ctx = MockContext()

        # Candidates cannot be generated right now (e.g. outside local costmap),
        # but item must REMAIN unresolved and NOT be prematurely abandoned.
        candidates1 = tester.pending_local_work_item_candidates(req, ctx, None, None)
        self.assertEqual(len(candidates1), 0)
        self.assertEqual(it["state"], WORK_UNRESOLVED)

        # Subsequent evaluation: item still remains unresolved, waiting for robot to get closer
        candidates2 = tester.pending_local_work_item_candidates(req, ctx, None, None)
        self.assertEqual(len(candidates2), 0)
        self.assertEqual(it["state"], WORK_UNRESOLVED)
        self.assertEqual(tester.place_work_items.unresolved_count(1), 1)

    def test_candidate_work_items_spatial_nms_suppression(self):
        """Candidates within density_separation distance are suppressed, keeping the denser one."""
        class MockOrigin:
            position = type("Pos", (), {"x": 0.0, "y": 0.0})()

        class MockInfo:
            resolution = 0.05
            origin = MockOrigin()

        class MockMessage:
            info = MockInfo()

        class MockRequest:
            message = MockMessage()
            steps = np.ones((50, 50), dtype=np.int32)
            validation = None
            map_to_physical_xy = None

        class MockContext:
            source_place_id = 1
            known_free = np.ones((50, 50), dtype=bool)

        class CandidateTester(GlobalFrontierCandidateWorkItemMixin):
            def __init__(self):
                self.place_work_items = PlaceWorkItemLedger(merge_radius=0.2, density_radius=0.0)
                self.work_item_density_radius = 0.8
                self.graph_route_work_item_filter = None
                self.graph_route_work_item_id = None
                self.portal_probe_ledger = None
                self.planning_should_preempt = lambda: False
                self.nearest_reachable_cell = lambda msg, steps, x, y: (
                    int(round(y / 0.05)), int(round(x / 0.05))
                )
                self.candidate_costmap_distance = lambda val, x, y: 1.0
                self.cell_xy = lambda msg, r, c: (c * 0.05, r * 0.05)

        tester = CandidateTester()
        it1 = tester.place_work_items._new_item(
            place_id=1,
            support_cells=frozenset((i, i) for i in range(30)),
            now=1.0,
            anchor_xy=(1.0, 1.0),
            normal_xy=(1.0, 0.0),
        )
        it1["support_cell_count"] = 30

        it2 = tester.place_work_items._new_item(
            place_id=1,
            support_cells=frozenset((i, i) for i in range(5)),
            now=1.0,
            anchor_xy=(1.2, 1.0),
            normal_xy=(1.0, 0.0),
        )
        it2["support_cell_count"] = 5

        req = MockRequest()
        ctx = MockContext()
        candidates = tester.pending_local_work_item_candidates(req, ctx, None, None)

        self.assertEqual(len(candidates), 1)
        accepted_cand = candidates[0][2]
        self.assertEqual(accepted_cand.work_item_id, it1["id"])
        self.assertEqual(accepted_cand.work_item_support_cells, 30)

        reasons = tester.last_local_work_item_rehydration_report["rejection_reasons"]
        self.assertIn("suppressed_by_denser_candidate", reasons)


if __name__ == "__main__":
    unittest.main()
