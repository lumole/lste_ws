"""GoalManager must reject non-free local endpoints before publication."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import rospy

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from goal_manager_goal_output import GoalManagerGoalOutputMixin  # noqa: E402


def costmap(data, width=21, height=21, resolution=0.10):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="map"),
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0),
            ),
        ),
        data=list(data),
    )


class Harness(GoalManagerGoalOutputMixin):
    goal_costmap_max_age = 3.0
    goal_costmap_clearance = 0.52
    latest_costmap_received_at = 0.0
    goal_source = "ctx_pair_follow"

    def __init__(self, message):
        self.latest_costmap = message
        # Zero means "freshness unavailable" in the pure geometry fixture;
        # the live callback always records a simulated receive timestamp.
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


class GoalManagerCostmapAdmissionTest(unittest.TestCase):
    @staticmethod
    def goal(x, y):
        return SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            pose=SimpleNamespace(
                position=SimpleNamespace(x=float(x), y=float(y)),
            ),
        )

    def test_inflated_endpoint_is_rejected(self):
        data = [0] * (21 * 21)
        data[10 * 21 + 10] = 99
        manager = Harness(costmap(data))
        self.assertFalse(manager.costmap_endpoint_admission(self.goal(1.05, 1.05)))
        self.assertEqual(manager.events[-1][0], "goal_rejected_costmap_endpoint")

    def test_known_zero_cost_endpoint_is_admitted(self):
        manager = Harness(costmap([0] * (21 * 21)))
        self.assertTrue(manager.costmap_endpoint_admission(self.goal(1.05, 1.05)))

    def test_lethal_margin_is_rejected_even_when_endpoint_is_zero(self):
        data = [0] * (21 * 21)
        data[10 * 21 + 5] = 100
        manager = Harness(costmap(data))
        self.assertFalse(manager.costmap_endpoint_admission(self.goal(0.85, 1.05)))


if __name__ == "__main__":
    unittest.main()
