"""Unit tests for WorkItem Open TSP tour solver and route integration."""

import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from global_frontier_graph_route_planner import (  # noqa: E402
    ACTION_OBSERVE_LOCAL_WORK,
    GraphRoutePlanner,
    PLAN_READY,
)
from global_frontier_work_item_tour import (  # noqa: E402
    WorkItemTourSolver,
    compute_open_tour_time,
    euclidean_distance,
    extract_work_item_coordinates,
    extract_work_item_pose,
    kinematic_time,
    nearest_neighbor_kinematic_open_tsp,
    nearest_neighbor_open_tsp,
    normalize_angle,
    optimize_kinematic_open_tsp,
    order_work_items_open_tsp,
    total_open_tour_length,
    total_open_tour_time,
    two_opt_open_tsp,
)


def make_work_item(
    item_id,
    x,
    y,
    theta=None,
    normal_xy=None,
    place_id=1,
    state="unresolved",
):
    """Helper to build a WorkItem dictionary with coordinates and optional heading."""
    item = {
        "id": item_id,
        "place_id": place_id,
        "x": float(x),
        "y": float(y),
        "anchor_xy": (float(x), float(y)),
        "state": state,
        "active_attempt_id": None,
    }
    if theta is not None:
        item["theta"] = float(theta)
    if normal_xy is not None:
        item["normal_xy"] = (float(normal_xy[0]), float(normal_xy[1]))
    return item


def make_place(place_id, covered=True):
    return {
        "id": place_id,
        "state": "open",
        "endpoint_observations": 1 if covered else 0,
        "work_items": [],
        "portals": [],
    }


class WorkItemTourSolverTest(unittest.TestCase):
    """Test the pure Open TSP algorithms: NN, 2-opt, coordinate parsing, lengths."""

    def setUp(self):
        self.solver = WorkItemTourSolver()

    def test_extract_coordinates_from_diverse_formats(self):
        # Dictionary with anchor_xy
        self.assertEqual(
            extract_work_item_coordinates({"anchor_xy": (1.5, 2.5)}),
            (1.5, 2.5),
        )
        # Dictionary with x, y
        self.assertEqual(
            extract_work_item_coordinates({"x": 3.0, "y": 4.0}),
            (3.0, 4.0),
        )
        # Tuple or list
        self.assertEqual(
            extract_work_item_coordinates([5.0, 6.0]),
            (5.0, 6.0),
        )

        # Object with attributes
        class DummyPose:
            x = 7.0
            y = 8.0

        self.assertEqual(
            extract_work_item_coordinates(DummyPose()),
            (7.0, 8.0),
        )

        # Invalid / None
        self.assertIsNone(extract_work_item_coordinates(None))
        self.assertIsNone(extract_work_item_coordinates({}))
        self.assertIsNone(extract_work_item_coordinates({"x": float("nan"), "y": 1.0}))

    def test_empty_and_single_work_item(self):
        # Empty list
        self.assertEqual(order_work_items_open_tsp((0.0, 0.0), []), [])
        # Single item
        single = [make_work_item(1, 10.0, 10.0)]
        self.assertEqual(order_work_items_open_tsp((0.0, 0.0), single), single)
        # None robot pose
        items = [make_work_item(1, 1.0, 0.0), make_work_item(2, 2.0, 0.0)]
        self.assertEqual(order_work_items_open_tsp(None, items), items)

    def test_backtrack_elimination_1d(self):
        """Eliminate back-and-forth zigzagging along a line."""
        robot_pose = (0.0, 0.0)
        # Scattered out of order along positive X axis: 10, 2, 8, 4
        items = [
            make_work_item(1, 10.0, 0.0),
            make_work_item(2, 2.0, 0.0),
            make_work_item(3, 8.0, 0.0),
            make_work_item(4, 4.0, 0.0),
        ]
        ordered = order_work_items_open_tsp(robot_pose, items)
        ordered_ids = [item["id"] for item in ordered]
        # Shortest open path starting at 0 must monotonically increase: 2 -> 4 -> 8 -> 10
        self.assertEqual(ordered_ids, [2, 4, 3, 1])

        # Verify total tour length is exactly 10.0
        self.assertAlmostEqual(
            total_open_tour_length(robot_pose, ordered), 10.0, places=5
        )

    def test_crossing_elimination_2d(self):
        """2-opt eliminates path intersections in 2D."""
        robot_pose = (0.0, 0.0)
        # Form an hourglass / cross:
        # P1=(0, 2), P2=(2, 0), P3=(0, 0.5), P4=(2, 2.5)
        # Connecting (0,0) -> P4 -> P1 -> P2 -> P3 crosses severely
        points = [(2.0, 2.0), (0.0, 2.0), (2.0, 0.0), (0.0, 1.0)]
        initial_bad_tour = [0, 1, 2, 3]
        bad_length = total_open_tour_length(
            robot_pose, [points[i] for i in initial_bad_tour]
        )

        optimized_indices = two_opt_open_tsp(
            robot_pose, points, initial_tour=initial_bad_tour
        )
        opt_length = total_open_tour_length(
            robot_pose, [points[i] for i in optimized_indices]
        )

        # 2-opt must strictly reduce the tour length
        self.assertLess(opt_length, bad_length)

    def test_items_with_missing_coordinates_appended_gracefully(self):
        robot_pose = (0.0, 0.0)
        item_valid_1 = make_work_item(1, 10.0, 0.0)
        item_valid_2 = make_work_item(2, 2.0, 0.0)
        item_no_coords = {"id": 99, "place_id": 1, "state": "unresolved"}

        ordered = order_work_items_open_tsp(
            robot_pose, [item_valid_1, item_no_coords, item_valid_2]
        )
        # Valid items ordered first by TSP (2 then 1), invalid item kept at end
        self.assertEqual([item["id"] for item in ordered], [2, 1, 99])


class WorkItemExecutionAndDynamicsTest(unittest.TestCase):
    """Test step-by-step execution, dynamic addition/removal, and obstacle fallback."""

    def setUp(self):
        self.planner = GraphRoutePlanner()
        self.places = [make_place(1, covered=True)]

    def test_step_by_step_execution(self):
        """Simulate sequential visit of all WorkItems in Open TSP order."""
        robot_pose = (0.0, 0.0)
        items = [
            make_work_item(10, 0.0, 6.0),
            make_work_item(20, 0.0, 2.0),
            make_work_item(30, 0.0, 4.0),
        ]

        # Step 1: from (0,0), next closest is item 20 at (0,2)
        plan1 = self.planner.plan(
            1, places=self.places, portals=(), work_items=items, robot_pose=robot_pose
        )
        self.assertEqual(plan1.status, PLAN_READY)
        self.assertEqual(plan1.action, ACTION_OBSERVE_LOCAL_WORK)
        self.assertEqual(plan1.obligation_id, 20)
        self.assertEqual(plan1.work_item_tour, (20, 30, 10))

        # Step 2: Robot reaches (0, 2), item 20 is resolved
        robot_pose = (0.0, 2.0)
        remaining_items = [items[0], items[2]]  # items 10 and 30
        plan2 = self.planner.plan(
            1, places=self.places, portals=(), work_items=remaining_items, robot_pose=robot_pose
        )
        self.assertEqual(plan2.obligation_id, 30)
        self.assertEqual(plan2.work_item_tour, (30, 10))

        # Step 3: Robot reaches (0, 4), item 30 is resolved
        robot_pose = (0.0, 4.0)
        remaining_items = [items[0]]  # item 10
        plan3 = self.planner.plan(
            1, places=self.places, portals=(), work_items=remaining_items, robot_pose=robot_pose
        )
        self.assertEqual(plan3.obligation_id, 10)
        self.assertEqual(plan3.work_item_tour, (10,))

    def test_dynamic_work_item_addition_stability(self):
        """Dynamically adding a nearby WorkItem smoothly adjusts visit sequence."""
        robot_pose = (0.0, 0.0)
        items = [
            make_work_item(1, 0.0, 10.0),
            make_work_item(2, 0.0, 20.0),
        ]
        plan_before = self.planner.plan(
            1, places=self.places, portals=(), work_items=items, robot_pose=robot_pose
        )
        self.assertEqual(plan_before.work_item_tour, (1, 2))

        # While moving towards item 1, robot is at (0, 3).
        # A new item 3 is discovered right nearby at (0, 4).
        robot_pose = (0.0, 3.0)
        items.append(make_work_item(3, 0.0, 4.0))
        plan_after = self.planner.plan(
            1, places=self.places, portals=(), work_items=items, robot_pose=robot_pose
        )
        # Item 3 is right ahead and visited first before 1 and 2
        self.assertEqual(plan_after.obligation_id, 3)
        self.assertEqual(plan_after.work_item_tour, (3, 1, 2))

    def test_dynamic_obstacle_viewpoint_fallback(self):
        """If the primary TSP candidate cannot form a viewpoint, automatically fall back to the next."""
        robot_pose = (0.0, 0.0)
        items = [
            make_work_item(1, 0.0, 2.0),
            make_work_item(2, 0.0, 5.0),
            make_work_item(3, 0.0, 8.0),
        ]

        # Case A: Normal unblocked case -> chooses item 1
        plan_normal = self.planner.plan(
            1, places=self.places, portals=(), work_items=items, robot_pose=robot_pose
        )
        self.assertEqual(plan_normal.obligation_id, 1)

        # Case B: Item 1 blocked by local obstacle (excluded via excluded_work_item_ids)
        # Should smoothly advance to item 2
        plan_fallback = self.planner.plan(
            1,
            places=self.places,
            portals=(),
            work_items=items,
            excluded_work_item_ids=(1,),
            robot_pose=robot_pose,
        )
        self.assertEqual(plan_fallback.status, PLAN_READY)
        self.assertEqual(plan_fallback.action, ACTION_OBSERVE_LOCAL_WORK)
        self.assertEqual(plan_fallback.obligation_id, 2)
        self.assertEqual(plan_fallback.work_item_tour, (2, 3))

        # Case C: Item 1 and Item 2 both blocked -> advances to item 3
        plan_fallback2 = self.planner.plan(
            1,
            places=self.places,
            portals=(),
            work_items=items,
            excluded_work_item_ids=(1, 2),
            robot_pose=robot_pose,
        )
        self.assertEqual(plan_fallback2.obligation_id, 3)
        self.assertEqual(plan_fallback2.work_item_tour, (3,))

    def test_visible_work_item_ids_filters_candidates_and_preserves_tsp_order(self):
        """visible_work_item_ids acts as the snapshot reachability barrier."""
        robot_pose = (0.0, 0.0)
        items = [
            make_work_item(10, 0.0, 2.0),
            make_work_item(20, 0.0, 4.0),
            make_work_item(30, 0.0, 6.0),
        ]
        # Suppose costmap only formed valid viewpoints for items 20 and 30 (item 10 unreachable)
        plan = self.planner.plan(
            1,
            places=self.places,
            portals=(),
            work_items=items,
            visible_work_item_ids=(20, 30),
            robot_pose=robot_pose,
        )
        self.assertEqual(plan.obligation_id, 20)
        self.assertEqual(plan.work_item_tour, (20, 30))

    def test_backward_compatibility_without_robot_pose(self):
        """When robot_pose is None, falls back to original ID ascending order."""
        items = [
            make_work_item(50, 0.0, 1.0),
            make_work_item(10, 0.0, 9.0),
            make_work_item(30, 0.0, 5.0),
        ]
        plan = self.planner.plan(
            1, places=self.places, portals=(), work_items=items, robot_pose=None
        )
        self.assertEqual(plan.obligation_id, 10)

    def test_negative_coordinates_four_quadrants(self):
        """Open TSP operates correctly across all 4 Cartesian quadrants."""
        robot_pose = (0.0, 0.0)
        items = [
            make_work_item(1, -5.0, -5.0),
            make_work_item(2, -1.0, -1.0),
            make_work_item(3, -10.0, -10.0),
        ]
        ordered = order_work_items_open_tsp(robot_pose, items)
        self.assertEqual([item["id"] for item in ordered], [2, 1, 3])

    def test_adapter_integration_records_work_item_tour(self):
        """Adapter fixture stores the active TSP tour in preference state."""
        from types import SimpleNamespace
        sys.modules.setdefault(
            "rospy",
            SimpleNamespace(
                loginfo=lambda *_args, **_kwargs: None,
                logwarn=lambda *_args, **_kwargs: None,
            ),
        )
        from global_frontier_graph_route_adapter import (
            GlobalFrontierGraphRouteAdapterMixin,
        )

        class MockLedger:
            def __init__(self, records):
                self.records = list(records)

            def snapshot(self):
                return list(self.records)

        class TestAdapter(GlobalFrontierGraphRouteAdapterMixin):
            def __init__(self, work_items, robot_xy):
                self.graph_route_planner_enabled = True
                self.graph_route_planner = GraphRoutePlanner()
                self.current_physical_place_id = 1
                self.branch_first_enabled = False
                self.region_memory = SimpleNamespace(
                    regions=[{"id": 1, "state": "open", "endpoint_observations": 1}]
                )
                self.portal_hypothesis_ledger = MockLedger([])
                self.portal_probe_ledger = MockLedger([])
                self.place_work_items = MockLedger(work_items)
                self.target_observation_work = None
                self.last_graph_route_plan = None
                self.last_graph_route_plan_signature = None
                self.graph_route_portal_id = None
                self.graph_route_probe_id = None
                self.graph_route_work_item_id = None
                self.graph_route_work_item_filter = None
                self.graph_route_work_item_tour = ()
                self.robot_map = robot_xy

            def publish_status(self, event, **fields):
                pass

        work_items = [
            make_work_item(101, 0.0, 8.0),
            make_work_item(102, 0.0, 2.0),
            make_work_item(103, 0.0, 5.0),
        ]
        adapter = TestAdapter(work_items, robot_xy=(0.0, 0.0))
        plan = adapter._set_graph_route_preference()

        self.assertIsNotNone(plan)
        self.assertEqual(plan.action, ACTION_OBSERVE_LOCAL_WORK)
        self.assertEqual(plan.obligation_id, 102)
        self.assertEqual(adapter.graph_route_work_item_id, 102)
        self.assertEqual(adapter.graph_route_work_item_tour, (102, 103, 101))

    def test_w13_and_w14_preferred_over_distant_w10_by_tsp(self):
        """Robot near gate in Place 2 must plan closest W13 first, then W14, avoiding distant W10."""
        robot_pose = (6.61996, 2.62568)
        planner = GraphRoutePlanner()
        places = [
            {"id": 1, "state": "open", "endpoint_observations": 1},
            {"id": 2, "state": "open", "endpoint_observations": 1},
        ]
        work_items = [
            make_work_item(10, 12.11, -2.42, place_id=1),   # 7.46m away in room 1
            make_work_item(12, 12.36, -3.50, place_id=2),   # 8.39m away in room 2
            make_work_item(13, 8.71, 2.94, place_id=2),     # 2.11m away in room 2 (closest!)
            make_work_item(14, 3.36, 3.81, place_id=2),     # 3.48m away in room 2 (second closest!)
        ]

        plan = planner.plan(
            2,
            places=places,
            portals=[],
            work_items=work_items,
            probes=[],
            visible_work_item_ids=(10, 12, 13, 14),
            robot_pose=robot_pose,
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.action, ACTION_OBSERVE_LOCAL_WORK)
        # Optimal Open TSP tour sweeping the room without backtracking: W14 -> W13 -> W12 (16.29m vs 19.12m)
        self.assertEqual(plan.obligation_id, 14)
        self.assertEqual(plan.work_item_tour, (14, 13, 12))
        self.assertNotIn(10, plan.work_item_tour)

    def test_candidate_selection_locks_onto_planned_work_item_over_high_score_w10(self):
        """Candidate selector must commit the planned TSP WorkItem (W13) rather than reconciling to distant W10."""
        from types import SimpleNamespace
        from global_frontier_selection_planner import GlobalFrontierSelectionPlannerMixin
        from global_frontier_graph_route_planner import GraphRoutePlan

        class SelectorFixture(GlobalFrontierSelectionPlannerMixin):
            def __init__(self):
                self.frontier_action_policy = "event_pareto"
                self.last_graph_selected_candidate = None

            def _select_frontier_decision_candidate(self, candidates, order, request):
                # Return the candidate's route
                return candidates[0].route

        fixture = SelectorFixture()
        plan = GraphRoutePlan(
            status="ready",
            action="observe_local_work",
            obligation_kind="work_item",
            obligation_id=13,
            current_place_id=2,
            target_place_id=2,
            work_item_tour=(13, 14),
        )

        # Candidate for W10 (with huge score) and candidate for W13 (planned)
        route_w10 = [0] * 20
        route_w10[10] = "frontier_endpoint"
        route_w10[12] = 10
        route_w10[17] = "observe_local_work"

        route_w13 = [0] * 20
        route_w13[10] = "frontier_endpoint"
        route_w13[12] = 13
        route_w13[17] = "observe_local_work"

        cand_w10 = SimpleNamespace(
            work_item_id=10,
            place_id=2,
            route=tuple(route_w10),
            graph_action="observe_local_work",
        )
        cand_w13 = SimpleNamespace(
            work_item_id=13,
            place_id=2,
            route=tuple(route_w13),
            graph_action="observe_local_work",
        )

        # Even if cand_w10 comes first in the candidate list
        selected = fixture._select_graph_plan_candidate(
            [cand_w10, cand_w13], plan, SimpleNamespace()
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected[12], 13)
        self.assertEqual(fixture.last_graph_selected_candidate.work_item_id, 13)


class KinematicWorkItemTourTest(unittest.TestCase):
    """Test kinematic time-optimal Open TSP metric, ATSP solver, and adapter integration."""

    def test_normalize_angle(self):
        self.assertAlmostEqual(normalize_angle(0.0), 0.0)
        self.assertAlmostEqual(normalize_angle(math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(3 * math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(-3 * math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(math.pi / 2), math.pi / 2)
        self.assertAlmostEqual(normalize_angle(-math.pi / 2), -math.pi / 2)

    def test_extract_work_item_pose(self):
        # Dict with anchor_xy and theta
        self.assertEqual(
            extract_work_item_pose({"anchor_xy": (1.0, 2.0), "theta": 0.5}),
            (1.0, 2.0, 0.5),
        )
        # Dict with x, y, yaw
        self.assertEqual(
            extract_work_item_pose({"x": 3.0, "y": 4.0, "yaw": -1.2}),
            (3.0, 4.0, -1.2),
        )
        # Dict with normal_xy
        pose = extract_work_item_pose({"x": 0.0, "y": 0.0, "normal_xy": (0.0, 1.0)})
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose[0], 0.0)
        self.assertAlmostEqual(pose[1], 0.0)
        self.assertAlmostEqual(pose[2], math.pi / 2)

        # Dict with viewpoint_yaw
        self.assertEqual(
            extract_work_item_pose({"x": 1.0, "y": 2.0, "viewpoint_yaw": 3.14}),
            (1.0, 2.0, 3.14),
        )
        # Tuple (x, y, theta)
        self.assertEqual(extract_work_item_pose((5.0, 6.0, 1.57)), (5.0, 6.0, 1.57))
        # Tuple (x, y) without heading
        self.assertEqual(extract_work_item_pose((5.0, 6.0)), (5.0, 6.0, None))
        # Invalid inputs
        self.assertIsNone(extract_work_item_pose(None))
        self.assertIsNone(extract_work_item_pose({}))
        self.assertIsNone(extract_work_item_pose({"x": float("nan"), "y": 1.0}))

    def test_kinematic_time_calculation(self):
        # 1. Straight forward motion (no turns)
        p1 = (0.0, 0.0, 0.0)
        p2 = (2.0, 0.0, 0.0)
        t = kinematic_time(p1, p2, v_max=0.4, omega_max=0.8)
        # dist=2.0, seg_th=0.0 -> delta_th1=0, delta_th2=0 -> 2.0 / 0.4 = 5.0s
        self.assertAlmostEqual(t, 5.0, places=5)

        # 2. 180 degree turnaround: moving backwards along -X while facing +X
        p3 = (-1.0, 0.0, 0.0)
        t_turn = kinematic_time(p1, p3, v_max=0.4, omega_max=0.8)
        # dist=1.0, seg_th=pi -> delta_th1=pi, delta_th2=pi -> 1.0/0.4 + 2*pi/0.8
        expected = 1.0 / 0.4 + (2.0 * math.pi) / 0.8
        self.assertAlmostEqual(t_turn, expected, places=5)

        # 3. Pure in-place rotation (dist near 0)
        p4 = (1.0, 1.0, 0.0)
        p5 = (1.0, 1.0, math.pi / 2)
        t_inplace = kinematic_time(p4, p5, v_max=0.4, omega_max=0.8)
        self.assertAlmostEqual(t_inplace, (math.pi / 2) / 0.8, places=5)

        # 4. Asymmetric property: T(A -> B) != T(B -> A) due to heading
        pA = (0.0, 0.0, 0.0)  # at origin facing +X
        pB = (1.0, 1.0, 0.0)  # at (1,1) facing +X
        # A -> B: seg_th = pi/4 -> delta_th1=pi/4, delta_th2=pi/4 -> sum=pi/2
        # B -> A: seg_th = -3pi/4 -> delta_th1 = 3pi/4, delta_th2 = 3pi/4 -> sum=3pi/2
        t_AB = kinematic_time(pA, pB, v_max=0.4, omega_max=0.8)
        t_BA = kinematic_time(pB, pA, v_max=0.4, omega_max=0.8)
        self.assertNotAlmostEqual(t_AB, t_BA)
        self.assertLess(t_AB, t_BA)

    def test_kinematic_smooth_forward_vs_awkward_backtrack(self):
        """Forward target (1.0m) prioritized over backward target (0.5m) to avoid 180-deg turnaround."""
        robot_pose = (0.0, 0.0, 0.0)  # facing +X
        # Item 1: 1.0m ahead along +X (straight forward)
        item_forward = make_work_item(1, 1.0, 0.0, theta=0.0)
        # Item 2: 0.5m behind along -X (closer in Euclidean distance, but requires 180° turnaround!)
        item_behind = make_work_item(2, -0.5, 0.0, theta=math.pi)

        # Pure Euclidean distance would choose item 2 (0.5m) first over item 1 (1.0m)
        self.assertLess(
            euclidean_distance((0.0, 0.0), (item_behind["x"], item_behind["y"])),
            euclidean_distance((0.0, 0.0), (item_forward["x"], item_forward["y"])),
        )

        # Kinematic time Open TSP must choose item 1 first to avoid awkward 180° turnaround
        ordered = order_work_items_open_tsp(
            robot_pose, [item_behind, item_forward], v_max=0.4, omega_max=0.8
        )
        self.assertEqual([item["id"] for item in ordered], [1, 2])

        # Verify total open tour time is lower for [1, 2] than [2, 1]
        time_1_then_2 = total_open_tour_time(
            robot_pose, [item_forward, item_behind], v_max=0.4, omega_max=0.8
        )
        time_2_then_1 = total_open_tour_time(
            robot_pose, [item_behind, item_forward], v_max=0.4, omega_max=0.8
        )
        self.assertLess(time_1_then_2, time_2_then_1)

    def test_kinematic_asymmetric_heading_preference(self):
        """Robot heading dynamically shifts preference between diagonal targets."""
        item_diag_pos = make_work_item(10, 1.0, 1.0, theta=math.pi / 4)
        item_diag_neg = make_work_item(20, 1.0, -1.0, theta=-math.pi / 4)
        items = [item_diag_pos, item_diag_neg]

        # Robot facing +pi/4: item 10 is straight ahead
        robot_pose_pos = (0.0, 0.0, math.pi / 4)
        ordered_pos = order_work_items_open_tsp(robot_pose_pos, items)
        self.assertEqual([item["id"] for item in ordered_pos], [10, 20])

        # Robot facing -pi/4: item 20 is straight ahead
        robot_pose_neg = (0.0, 0.0, -math.pi / 4)
        ordered_neg = order_work_items_open_tsp(robot_pose_neg, items)
        self.assertEqual([item["id"] for item in ordered_neg], [20, 10])

    def test_planner_respects_kinematic_time_and_robot_yaw(self):
        """GraphRoutePlanner dispatches the forward item when yaw is aligned."""
        planner = GraphRoutePlanner()
        places = [make_place(1, covered=True)]
        item_forward = make_work_item(1, 1.0, 0.0, theta=0.0)
        item_behind = make_work_item(2, -0.5, 0.0, theta=math.pi)
        items = [item_behind, item_forward]

        # Case A: robot facing +X (yaw = 0.0) -> item 1 first
        plan_fwd = planner.plan(
            1,
            places=places,
            portals=(),
            work_items=items,
            robot_pose=(0.0, 0.0, 0.0),
            v_max=0.4,
            omega_max=0.8,
        )
        self.assertEqual(plan_fwd.obligation_id, 1)
        self.assertEqual(plan_fwd.work_item_tour, (1, 2))

        # Case B: robot facing -X (yaw = pi) -> item 2 is now straight ahead!
        plan_rev = planner.plan(
            1,
            places=places,
            portals=(),
            work_items=items,
            robot_pose=(0.0, 0.0, math.pi),
            v_max=0.4,
            omega_max=0.8,
        )
        self.assertEqual(plan_rev.obligation_id, 2)
        self.assertEqual(plan_rev.work_item_tour, (2, 1))

    def test_adapter_passes_robot_yaw_and_speeds_to_planner(self):
        """Adapter correctly enriches robot_pose with robot_yaw and passes tsp_v_max/tsp_omega_max."""
        from types import SimpleNamespace
        sys.modules.setdefault(
            "rospy",
            SimpleNamespace(
                loginfo=lambda *_args, **_kwargs: None,
                logwarn=lambda *_args, **_kwargs: None,
                logerr=lambda *_args, **_kwargs: None,
            ),
        )
        from global_frontier_graph_route_adapter import (
            GlobalFrontierGraphRouteAdapterMixin,
        )

        class MockLedger:
            def __init__(self, records):
                self.records = list(records)

            def snapshot(self):
                return list(self.records)

        class TestKinematicAdapter(GlobalFrontierGraphRouteAdapterMixin):
            def __init__(self, work_items, robot_xy, robot_yaw=0.0):
                self.graph_route_planner_enabled = True
                self.graph_route_planner = GraphRoutePlanner()
                self.current_physical_place_id = 1
                self.branch_first_enabled = False
                self.region_memory = SimpleNamespace(
                    regions=[{"id": 1, "state": "open", "endpoint_observations": 1}]
                )
                self.portal_hypothesis_ledger = MockLedger([])
                self.portal_probe_ledger = MockLedger([])
                self.place_work_items = MockLedger(work_items)
                self.target_observation_work = None
                self.last_graph_route_plan = None
                self.last_graph_route_plan_signature = None
                self.graph_route_portal_id = None
                self.graph_route_probe_id = None
                self.graph_route_work_item_id = None
                self.graph_route_work_item_filter = None
                self.graph_route_work_item_tour = ()
                self.robot_map = robot_xy
                self.robot_yaw_map = robot_yaw
                self.tsp_v_max = 0.4
                self.tsp_omega_max = 0.8

            def publish_status(self, event, **fields):
                pass

        item_forward = make_work_item(1, 1.0, 0.0, theta=0.0)
        item_behind = make_work_item(2, -0.5, 0.0, theta=math.pi)
        work_items = [item_behind, item_forward]

        # Robot yaw = 0.0 -> forward item 1 chosen
        adapter_fwd = TestKinematicAdapter(work_items, robot_xy=(0.0, 0.0), robot_yaw=0.0)
        plan_fwd = adapter_fwd._set_graph_route_preference()
        self.assertEqual(plan_fwd.obligation_id, 1)
        self.assertEqual(adapter_fwd.graph_route_work_item_tour, (1, 2))

        # Robot yaw = pi -> item 2 chosen
        adapter_rev = TestKinematicAdapter(work_items, robot_xy=(0.0, 0.0), robot_yaw=math.pi)
        plan_rev = adapter_rev._set_graph_route_preference()
        self.assertEqual(plan_rev.obligation_id, 2)
        self.assertEqual(adapter_rev.graph_route_work_item_tour, (2, 1))


if __name__ == "__main__":
    unittest.main()

