#!/usr/bin/env python3
"""Unit tests for Geodesic Dijkstra distance field and Spawn-Constrained Open TSP."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from global_frontier_grid import (
    compute_geodesic_distance_to_work_item,
    geodesic_dijkstra,
    get_or_compute_spawn_geodesic_field,
)
from global_frontier_work_item_tour import (
    compute_open_tour_time,
    extract_work_item_spawn_distance,
    nearest_neighbor_kinematic_open_tsp,
    optimize_kinematic_open_tsp,
    order_work_items_open_tsp,
    WorkItemTourSolver,
)
from global_frontier_work_items import (
    PlaceWorkItemLedger,
    WORK_UNRESOLVED,
)


class SpawnConstrainedTspTest(unittest.TestCase):
    """Test suite covering geodesic distance fields, ledger caching, and spawn-penalized TSP."""

    def test_8_connected_diagonal_vs_orthogonal_cost(self):
        """Orthogonal edge is 1.0 * res, diagonal edge is sqrt(2) * res."""
        free = np.ones((5, 5), dtype=bool)
        res = 0.5
        dist = geodesic_dijkstra(free, (0, 0), resolution=res)

        self.assertAlmostEqual(dist[0, 0], 0.0, places=5)
        # Orthogonal neighbor (0, 1)
        self.assertAlmostEqual(dist[0, 1], 1.0 * res, places=5)
        # Diagonal neighbor (1, 1)
        self.assertAlmostEqual(dist[1, 1], math.sqrt(2.0) * res, places=5)
        # Two diagonal steps (2, 2)
        self.assertAlmostEqual(dist[2, 2], 2.0 * math.sqrt(2.0) * res, places=5)

    def test_through_wall_geometric_deception(self):
        """Through-wall Euclidean proximity vs true obstacle-avoiding detour distance."""
        # Grid of 20 rows x 20 cols, res = 1.0m
        free = np.ones((20, 20), dtype=bool)
        # Wall at col 10 from row 0 to row 14
        free[0:15, 10] = False  # Wall dividing left and right sides
        # Doorway / opening is at rows 15-19, col 10

        spawn_xy = (2, 5)  # Left room (row 2, col 5)
        target_behind_wall = (2, 14)  # Right room (row 2, col 14)
        target_in_same_room = (8, 5)  # Left room corridor (row 8, col 5)

        # Straight-line Euclidean distances
        euc_wall = math.hypot(target_behind_wall[0] - spawn_xy[0], target_behind_wall[1] - spawn_xy[1])
        euc_same = math.hypot(target_in_same_room[0] - spawn_xy[0], target_in_same_room[1] - spawn_xy[1])
        # Target behind wall is straight-line: hypot(0, 9) = 9.0m
        # Target in same room is straight-line: hypot(6, 0) = 6.0m
        # Now let's test with a closer through-wall point
        target_close_behind_wall = (2, 12)  # hypot(0, 7) = 7.0m vs (9, 5) = hypot(7, 0) = 7.0m

        field = get_or_compute_spawn_geodesic_field(free, spawn_xy, resolution=1.0)
        dist_behind_wall = compute_geodesic_distance_to_work_item(
            free, spawn_xy, target_behind_wall, resolution=1.0, geodesic_field=field
        )
        dist_same_room = compute_geodesic_distance_to_work_item(
            free, spawn_xy, target_in_same_room, resolution=1.0, geodesic_field=field
        )

        # In same room: direct straight path is 6 cells = 6.0m
        self.assertAlmostEqual(dist_same_room, 6.0, places=2)

        # Behind wall: must go down to row 15 (at least delta_row = 13), across col 10 (delta_col = 5),
        # then back up to row 2 (delta_row = 13), delta_col = 4. Total geodesic distance > 25m!
        self.assertGreater(dist_behind_wall, 25.0)
        self.assertGreater(dist_behind_wall, dist_same_room * 3.0)

    def test_ledger_spawn_distance_cached_once_permanently(self):
        """WorkItem spawn_distance is computed upon creation/reconciliation and locked."""
        ledger = PlaceWorkItemLedger(merge_radius=0.5)
        spawn_xy = (0.0, 0.0)
        ledger.spawn_odom_xy = spawn_xy

        # Mock a support with anchor at (3.0, 4.0) -> Euclidean distance 5.0
        support = MagicMock()
        support.support_cells = frozenset([(10, 10), (10, 11)])
        support.frontier_cells = [(10, 10)]
        support.anchor_xy = (3.0, 4.0)
        support.normal_xy = (1.0, 0.0)

        cells, details = ledger.reconcile(1, (support,), now=1.0)
        item_id = next(iter(details.keys()))
        item = ledger._items[item_id]

        self.assertIsNotNone(item.get("spawn_distance"))
        self.assertAlmostEqual(item["spawn_distance"], 5.0, places=3)

        # Manually alter spawn_distance to a sentinel value to check if reconcile overwrites it
        item["spawn_distance"] = 999.0
        ledger.spawn_odom_xy = (100.0, 100.0)  # Changed spawn to test immutability

        # Second reconciliation tick
        cells2, details2 = ledger.reconcile(1, (support,), now=2.0)
        # Must retain 999.0 because spawn_distance is immutable once set!
        self.assertEqual(item["spawn_distance"], 999.0)

    def test_spawn_penalty_formula_values(self):
        """Verify mathematical formula of initial proximity and backtracking penalty."""
        # Unoriented poses (theta = None) to test pure linear translation + spawn penalty
        start_pose = (0.0, 0.0, None)
        poses = [(2.0, 0.0, None), (8.0, 0.0, None)]
        spawns = [2.0, 8.0]

        # Tour [0, 1] (near to far):
        # Step 0: start -> pose[0]: dist 2.0. Kin time: 2.0 / 0.4 = 5.0s. Initial penalty: beta * 2.0 = 0.8 * 2.0 = 1.6s.
        # Step 1: pose[0] -> pose[1]: dist 6.0. Kin time: 6.0 / 0.4 = 15.0s. Backtrack: max(0, 2 - 8) = 0.0.
        # Total: 5.0 + 1.6 + 15.0 + 0 = 21.6s.
        cost_fwd = compute_open_tour_time(
            start_pose, poses, [0, 1], v_max=0.4, omega_max=0.8, spawn_distances=spawns, alpha=1.5, beta=0.8
        )
        self.assertAlmostEqual(cost_fwd, 21.6, places=4)

        # Tour [1, 0] (far to near, backtracking):
        # Step 0: start -> pose[1]: dist 8.0. Kin time: 8.0 / 0.4 = 20.0s. Initial penalty: beta * 8.0 = 0.8 * 8.0 = 6.4s.
        # Step 1: pose[1] -> pose[0]: dist 6.0. Kin time: 6.0 / 0.4 = 15.0s. Backtrack: alpha * (8.0 - 2.0) = 1.5 * 6.0 = 9.0s.
        # Total: 20.0 + 6.4 + 15.0 + 9.0 = 50.4s.
        cost_rev = compute_open_tour_time(
            start_pose, poses, [1, 0], v_max=0.4, omega_max=0.8, spawn_distances=spawns, alpha=1.5, beta=0.8
        )
        self.assertAlmostEqual(cost_rev, 50.4, places=4)
        self.assertGreater(cost_rev, cost_fwd * 2.0)

    def test_backtracking_penalty_prevents_rash_exploration_and_reverse(self):
        """Robot closer to far item still prefers near item to avoid massive backtracking penalty."""
        # Robot is at x = 5.5, y = 0.0
        robot_pose = (5.5, 0.0, 0.0)
        # Item 1 is near spawn at x = 1.0 (distance to robot = 4.5m)
        item_near_spawn = {"id": 1, "x": 1.0, "y": 0.0, "spawn_distance": 1.0}
        # Item 2 is far at x = 9.0 (distance to robot = 3.5m)
        item_far_spawn = {"id": 2, "x": 9.0, "y": 0.0, "spawn_distance": 9.0}

        items = [item_far_spawn, item_near_spawn]

        # Case 1: Pure kinematic (alpha=0, beta=0) -> robot goes to Item 2 first because 3.5m < 4.5m
        ordered_pure = order_work_items_open_tsp(
            robot_pose, items, v_max=1.0, omega_max=1.0, alpha=0.0, beta=0.0
        )
        self.assertEqual([it["id"] for it in ordered_pure], [2, 1])

        # Case 2: Spawn-constrained (alpha=1.5, beta=0.8) -> penalty eliminates rash detour to Item 2
        ordered_penalized = order_work_items_open_tsp(
            robot_pose, items, v_max=1.0, omega_max=1.0, alpha=1.5, beta=0.8
        )
        self.assertEqual([it["id"] for it in ordered_penalized], [1, 2])

    def test_open_tsp_ordering_outward_progression(self):
        """A series of items ordered with spawn penalty preserves smooth outward progression."""
        robot_pose = (0.0, 0.0, 0.0)
        items = [
            {"id": 30, "x": 12.0, "y": 0.0, "spawn_distance": 12.0},
            {"id": 10, "x": 2.0, "y": 0.0, "spawn_distance": 2.0},
            {"id": 20, "x": 6.0, "y": 0.0, "spawn_distance": 6.0},
        ]
        ordered = order_work_items_open_tsp(robot_pose, items, alpha=1.5, beta=0.8)
        self.assertEqual([it["id"] for it in ordered], [10, 20, 30])

    def test_tour_solver_class_with_spawn_distances(self):
        """WorkItemTourSolver cleanly exposes configurable alpha and beta."""
        solver = WorkItemTourSolver(v_max=0.5, omega_max=1.0, alpha=1.5, beta=0.8)
        robot_pose = (0.0, 0.0, 0.0)
        items = [
            {"id": 2, "x": 10.0, "y": 0.0, "spawn_distance": 10.0},
            {"id": 1, "x": 1.0, "y": 0.0, "spawn_distance": 1.0},
        ]
        planned = solver.plan_tour(robot_pose, items)
        self.assertEqual([it["id"] for it in planned], [1, 2])

    def test_through_wall_deception_combined_with_tsp_ordering(self):
        """Through-wall Euclidean proximity vs geodesic distance alters TSP sequence."""
        # Grid: 30x30, resolution 0.1m
        free = np.ones((30, 30), dtype=bool)
        # Wall at col 15, rows 0 to 24 (doorway only at bottom rows 25-29)
        free[0:25, 15] = False

        res = 0.1
        # Spawn is at row 5, col 5 -> (x=0.5m, y=0.5m)
        spawn_xy = (0.5, 0.5)

        # Target A (Behind Wall): row 5, col 20 -> (x=2.0m, y=0.5m)
        # Straight-line Euclidean distance: 1.5m
        target_a_xy = (2.0, 0.5)

        # Target B (In Corridor): row 15, col 5 -> (x=0.5m, y=1.5m)
        # Straight-line Euclidean distance: 1.0m
        # But let's place Target B at row 18, col 5 -> (x=0.5m, y=1.8m)
        # Euclidean: 1.3m (longer than 1.5m? No, 1.3m is shorter).
        # Let's place Target A at row 5, col 18 -> (x=1.8m, y=0.5m). Euc dist = 1.3m.
        # Target B at row 20, col 5 -> (x=0.5m, y=2.0m). Euc dist = 1.5m.
        # Notice: Straight-line Euclidean distance to Target A (1.3m) < Target B (1.5m)!
        target_a_behind_wall = {"id": "A_wall", "anchor_xy": (1.8, 0.5)}
        target_b_corridor = {"id": "B_corridor", "anchor_xy": (0.5, 2.0)}

        origin = (0.0, 0.0)
        field = get_or_compute_spawn_geodesic_field(
            free, spawn_xy, resolution=res, origin_xy=origin
        )

        dist_a = compute_geodesic_distance_to_work_item(
            free, spawn_xy, target_a_behind_wall, resolution=res, origin_xy=origin, geodesic_field=field
        )
        dist_b = compute_geodesic_distance_to_work_item(
            free, spawn_xy, target_b_corridor, resolution=res, origin_xy=origin, geodesic_field=field
        )

        target_a_behind_wall["spawn_distance"] = dist_a
        target_b_corridor["spawn_distance"] = dist_b

        # Geodesic distance to A must go down to row 25 around the wall:
        # > (25 - 5) * 0.1 * 2 = 4.0m!
        self.assertGreater(dist_a, 4.0)
        # Geodesic distance to B is straight down corridor: (20 - 5) * 0.1 = 1.5m
        self.assertAlmostEqual(dist_b, 1.5, places=2)

        # Now plan Open TSP with robot at spawn:
        robot_pose = (0.5, 0.5, 0.0)
        items = [target_a_behind_wall, target_b_corridor]

        # Case 1: If someone foolishly used straight-line Euclidean distance:
        euc_a = math.hypot(1.8 - 0.5, 0.5 - 0.5)  # 1.3m
        euc_b = math.hypot(0.5 - 0.5, 2.0 - 0.5)  # 1.5m
        self.assertLess(euc_a, euc_b)  # Euclidean thinks A is closer!

        # Case 2: With geodesic Dijkstra spawn distances, Target B is chosen FIRST!
        ordered = order_work_items_open_tsp(robot_pose, items, v_max=0.5, omega_max=1.0, alpha=1.5, beta=0.8)
        self.assertEqual([it["id"] for it in ordered], ["B_corridor", "A_wall"])

    def test_robot_deep_in_corridor_still_prioritizes_spawn_near_work_item_decisively(self):
        """When robot has moved close to a far work item, decisive alpha/beta forces near-spawn item first."""
        # Realistic scenario from live run:
        # Robot is at (12.0, -1.0) right next to W8 (only ~1m away), while W16/W12 are > 10m away near spawn
        robot_pose = (12.0, -1.0, 0.0)
        items = [
            {"id": 8, "anchor_xy": (12.88, -1.39), "spawn_distance": 12.95},
            {"id": 10, "anchor_xy": (8.68, 2.93), "spawn_distance": 9.16},
            {"id": 11, "anchor_xy": (12.12, -2.91), "spawn_distance": 12.46},
            {"id": 12, "anchor_xy": (4.19, 4.61), "spawn_distance": 6.23},
            {"id": 15, "anchor_xy": (12.0, -5.4), "spawn_distance": 13.16},
            {"id": 16, "anchor_xy": (3.25, 3.72), "spawn_distance": 4.94},
        ]
        # With default alpha=4.0, beta=3.0:
        # Near-spawn items W16 (spawn_dist 4.94) and W12 (spawn_dist 6.23) MUST be chosen first
        ordered = order_work_items_open_tsp(robot_pose, items)
        ordered_ids = [it["id"] for it in ordered]
        self.assertIn(ordered_ids[0], [16, 12])
        self.assertIn(ordered_ids[1], [16, 12])
        # Distant items 8, 11, 15 must come after 16, 12, 10
        self.assertGreater(ordered_ids.index(8), ordered_ids.index(12))
        self.assertGreater(ordered_ids.index(11), ordered_ids.index(12))
        self.assertGreater(ordered_ids.index(15), ordered_ids.index(12))


if __name__ == "__main__":
    unittest.main()
