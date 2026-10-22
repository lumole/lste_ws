"""Unit tests for safe, clearance-backed WorkItem viewpoint generation."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from global_frontier_candidate_work_items import GlobalFrontierCandidateWorkItemMixin
from global_frontier_portal_probe_ledger import PortalProbeLedger
from global_frontier_work_items import PlaceWorkItemLedger
from global_frontier_planning_routes import GlobalFrontierPlanningRouteMixin
from goal_manager_goal_output import GoalManagerGoalOutputMixin


class GoalManagerHarness(GoalManagerGoalOutputMixin):
    goal_costmap_max_age = 3.0
    goal_costmap_clearance = 0.52
    latest_costmap_received_at = 0.0
    goal_source = "frontier_replan"

    def __init__(self, message):
        self.latest_costmap = message
        self.latest_costmap_received_at = 0.0
        self.events = []

    @staticmethod
    def _pose_in_frame(goal, _target_frame):
        return goal

    @staticmethod
    def _frame_name(frame):
        return (frame or "map").strip().lstrip("/") or "map"

    def publish_goal_arbitration(self, event, **fields):
        self.events.append((event, fields))


class SafeViewpointFixture(
    GlobalFrontierCandidateWorkItemMixin,
    GlobalFrontierPlanningRouteMixin,
):
    def __init__(self):
        self.place_work_items = PlaceWorkItemLedger(merge_radius=0.35)
        self.portal_probe_ledger = PortalProbeLedger(match_radius=0.35)
        self.frontier_approach_distance = 1.0
        self.work_item_approach_distance = 0.55
        self.active_reassociation_radius = 1.0
        self.clearance = 0.52
        self.completed_radius = 1.25
        self.events = []
        self.disable_inverse_tf = False

    @staticmethod
    def planar_xy_projector(target_frame, source_frame):
        del target_frame, source_frame
        return lambda x, y: (float(x), float(y))

    @staticmethod
    def cell_xy(message, row, col):
        res = float(message.info.resolution)
        ox = float(message.info.origin.position.x)
        oy = float(message.info.origin.position.y)
        return ox + col * res, oy + row * res

    def candidate_costmap_distance(self, validation, x, y):
        if validation is None:
            return None
        costmap_msg, costmap_data, costmap_steps = validation
        res = float(costmap_msg.info.resolution)
        ox = float(costmap_msg.info.origin.position.x)
        oy = float(costmap_msg.info.origin.position.y)
        col = int(round((x - ox) / res))
        row = int(round((y - oy) / res))
        if not (0 <= row < costmap_data.shape[0] and 0 <= col < costmap_data.shape[1]):
            return None
        if costmap_data[row, col] != 0:
            return None
        return float(costmap_steps[row, col]) * res

    def publish_status(self, event, **fields):
        self.events.append((event, fields))


class SafeWorkItemViewpointTest(unittest.TestCase):

    def test_w18_scenario_avoids_obstacle_inflation_and_selects_safe_viewpoint(self):
        """Simulate W18 scenario: desired viewpoint hugs obstacle at y=3.5, safe cell at y<=3.0 is selected."""
        fixture = SafeViewpointFixture()

        # WorkItem Anchor near table edge at y=2.90, normal points to y < 2.9 (into unknown)
        # Normal = (0, -1), so desired viewpoint along ladder is at anchor - depth * normal = y=3.45 (table is at y=3.50)
        item = fixture.place_work_items._new_item(
            1,
            frozenset({(29, 30)}),
            1.0,
            anchor_xy=(3.05, 2.90),
            normal_xy=(0.0, -1.0),
        )

        res = 0.05
        # Grid: 100x100, origin at (0.0, 0.0) -> covers [0, 5.0] x [0, 5.0]
        rows, cols = 100, 100
        message = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            info=SimpleNamespace(
                resolution=res,
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0, z=0.0)
                ),
            ),
        )

        # Occupied: table is located at row >= 70 (y >= 3.50)
        occupied = np.zeros((rows, cols), dtype=bool)
        occupied[70:, :] = True

        # Preferred mask (strict_free): clearance >= 0.52m from y=3.50 -> row <= 59 (y <= 2.95)
        preferred_mask = np.zeros((rows, cols), dtype=bool)
        preferred_mask[:60, 20:80] = True

        # Reachable steps: all cells with y < 3.50 are free & reachable
        steps = np.full((rows, cols), -1, dtype=np.int32)
        steps[:70, :] = 10

        # Costmap: inflation zone extends from y=3.50 down to y=3.00 (rows 60 to 70 have cost > 0)
        # Below row 60 (y < 3.00), cost == 0
        costmap_data = np.zeros((rows, cols), dtype=np.int16)
        costmap_data[70:, :] = 254
        costmap_data[60:70, :] = 16  # Inflation layer
        costmap_steps = np.full((rows, cols), 10, dtype=np.int32)

        validation = (message, costmap_data, costmap_steps)

        req = SimpleNamespace(
            message=message,
            steps=steps,
            preferred_mask=preferred_mask,
            occupied=occupied,
            validation=validation,
            map_to_physical_xy=lambda x, y: (float(x), float(y)),
        )
        ctx = SimpleNamespace(source_place_id=1)

        candidates = fixture.pending_local_work_item_candidates(
            req, ctx, frontier=np.zeros((rows, cols), dtype=bool), approach_cells=1,
        )

        self.assertEqual(len(candidates), 1)
        _row, _col, candidate = candidates[0]

        # The candidate must be at y <= 2.95 (in preferred_mask, outside inflation layer)
        self.assertLessEqual(candidate.y, 2.95)
        # The candidate must have cost == 0 in costmap
        cand_row = int(round(candidate.y / res))
        cand_col = int(round(candidate.x / res))
        self.assertEqual(costmap_data[cand_row, cand_col], 0)
        # Distance to anchor must be <= limit
        dist = math.hypot(candidate.x - 3.05, candidate.y - 2.90)
        self.assertLessEqual(dist, 1.55)

    def test_line_of_sight_blocks_viewpoint_behind_wall(self):
        """Viewpoints that have an obstacle blocking the direct line to the anchor must not be chosen."""
        fixture = SafeViewpointFixture()

        # Anchor at (2.0, 2.0)
        item = fixture.place_work_items._new_item(
            1,
            frozenset({(20, 20)}),
            1.0,
            anchor_xy=(2.0, 2.0),
            normal_xy=(1.0, 0.0),
        )

        res = 0.05
        rows, cols = 80, 80
        message = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            info=SimpleNamespace(
                resolution=res,
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0, z=0.0)
                ),
            ),
        )

        occupied = np.zeros((rows, cols), dtype=bool)
        # Wall at col=35 (x=1.75) spanning rows 0..79, completely separating left and right
        occupied[:, 35] = True

        preferred_mask = np.ones((rows, cols), dtype=bool)
        preferred_mask[occupied] = False

        # Only cells to the left of the wall (col < 35) are reachable
        steps = np.full((rows, cols), -1, dtype=np.int32)
        steps[:, :35] = 10

        costmap_data = np.zeros((rows, cols), dtype=np.int16)
        costmap_data[occupied] = 254
        costmap_steps = np.full((rows, cols), -1, dtype=np.int32)
        costmap_steps[:, :35] = 10

        validation = (message, costmap_data, costmap_steps)

        req = SimpleNamespace(
            message=message,
            steps=steps,
            preferred_mask=preferred_mask,
            occupied=occupied,
            validation=validation,
            map_to_physical_xy=lambda x, y: (float(x), float(y)),
        )
        ctx = SimpleNamespace(source_place_id=1)

        candidates = fixture.pending_local_work_item_candidates(
            req, ctx, frontier=np.zeros((rows, cols), dtype=bool), approach_cells=1,
        )

        # Because the wall blocks all line-of-sight from the reachable left side to the anchor on the right,
        # no candidate should be chosen through the wall.
        self.assertEqual(len(candidates), 0)

    def test_w18_selected_goal_is_admitted_by_goal_manager(self):
        """Verify the safe viewpoint selected for W18 is fully admitted by GoalManager."""
        fixture = SafeViewpointFixture()

        item = fixture.place_work_items._new_item(
            1,
            frozenset({(29, 30)}),
            1.0,
            anchor_xy=(3.05, 2.90),
            normal_xy=(0.0, -1.0),
        )

        res = 0.05
        rows, cols = 100, 100
        message = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            info=SimpleNamespace(
                width=cols,
                height=rows,
                resolution=res,
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0, z=0.0)
                ),
            ),
        )

        occupied = np.zeros((rows, cols), dtype=bool)
        occupied[70:, :] = True  # Obstacle at y >= 3.50

        preferred_mask = np.zeros((rows, cols), dtype=bool)
        preferred_mask[:60, 20:80] = True  # Safe clearance at y <= 2.95

        steps = np.full((rows, cols), -1, dtype=np.int32)
        steps[:70, :] = 10

        costmap_data = np.zeros((rows, cols), dtype=np.int16)
        costmap_data[70:, :] = 254
        costmap_data[60:70, :] = 16  # Inflation zone
        costmap_steps = np.full((rows, cols), 10, dtype=np.int32)

        validation = (message, costmap_data, costmap_steps)

        req = SimpleNamespace(
            message=message,
            steps=steps,
            preferred_mask=preferred_mask,
            occupied=occupied,
            validation=validation,
            map_to_physical_xy=lambda x, y: (float(x), float(y)),
        )
        ctx = SimpleNamespace(source_place_id=1)

        candidates = fixture.pending_local_work_item_candidates(
            req, ctx, frontier=np.zeros((rows, cols), dtype=bool), approach_cells=1,
        )
        self.assertEqual(len(candidates), 1)
        _row, _col, candidate = candidates[0]

        # Feed the selected goal into GoalManager
        costmap_msg = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            info=SimpleNamespace(
                width=cols,
                height=rows,
                resolution=res,
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0, z=0.0)
                ),
            ),
            data=costmap_data.flatten().tolist(),
        )
        goal_manager = GoalManagerHarness(costmap_msg)

        goal = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            pose=SimpleNamespace(
                position=SimpleNamespace(x=candidate.x, y=candidate.y),
            ),
        )

        # GoalManager must admit this safe goal without rejecting
        admitted = goal_manager.costmap_endpoint_admission(goal)
        self.assertTrue(admitted)
        self.assertEqual(len(goal_manager.events), 0)

    def test_narrow_corridor_tier2_clearance_fallback(self):
        """When preferred_mask is empty (narrow corridor), Tier 2 finds the reachable cell with cost==0."""
        fixture = SafeViewpointFixture()

        item = fixture.place_work_items._new_item(
            1,
            frozenset({(20, 20)}),
            1.0,
            anchor_xy=(1.0, 1.0),
            normal_xy=(1.0, 0.0),
        )

        res = 0.05
        rows, cols = 60, 60
        message = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            info=SimpleNamespace(
                width=cols,
                height=rows,
                resolution=res,
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0, z=0.0)
                ),
            ),
        )

        # Narrow corridor: preferred_mask has NO valid cells (clearance 0.52m impossible)
        preferred_mask = np.zeros((rows, cols), dtype=bool)

        occupied = np.zeros((rows, cols), dtype=bool)
        # Walls along rows < 10 and rows > 30 (y < 0.5 and y > 1.5)
        occupied[:10, :] = True
        occupied[30:, :] = True

        steps = np.full((rows, cols), -1, dtype=np.int32)
        steps[10:30, :] = 5

        costmap_data = np.zeros((rows, cols), dtype=np.int16)
        costmap_data[occupied] = 254
        # Middle of corridor [18:22, :] has cost == 0
        costmap_data[10:18, :] = 20
        costmap_data[22:30, :] = 20
        costmap_steps = np.full((rows, cols), 5, dtype=np.int32)

        validation = (message, costmap_data, costmap_steps)

        req = SimpleNamespace(
            message=message,
            steps=steps,
            preferred_mask=preferred_mask,
            occupied=occupied,
            validation=validation,
            map_to_physical_xy=lambda x, y: (float(x), float(y)),
        )
        ctx = SimpleNamespace(source_place_id=1)

        candidates = fixture.pending_local_work_item_candidates(
            req, ctx, frontier=np.zeros((rows, cols), dtype=bool), approach_cells=1,
        )

        self.assertEqual(len(candidates), 1)
        _row, _col, candidate = candidates[0]
        cand_row = int(round(candidate.y / res))
        cand_col = int(round(candidate.x / res))
        # Middle of corridor must be selected where cost == 0
        self.assertEqual(costmap_data[cand_row, cand_col], 0)
        self.assertTrue(18 <= cand_row <= 22)




if __name__ == "__main__":
    unittest.main()
