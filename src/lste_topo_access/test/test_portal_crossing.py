"""Unit tests for odometry-based directional portal evidence."""

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from global_frontier_portal_crossing import (
    portal_crossing_depth,
    portal_crossing_evidence,
    portal_execution_goal,
    portal_signed_distance,
    portal_source_side_is_proven,
)
from global_frontier_portal_crossing_lifecycle import (
    GlobalFrontierPortalCrossingLifecycleMixin,
)


class _Ledger:
    def __init__(self, record):
        self._record = record

    def get(self, _portal_id):
        return self._record


class _OdomPose:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class CrossingPlaneHarness(GlobalFrontierPortalCrossingLifecycleMixin):
    def __init__(self, *, terminal_xy, goal_xy, record=None,
                 start_xy=(4.0, 0.0)):
        self.clearance = 0.52
        self.pose_odom = _OdomPose(*terminal_xy)
        self.active_start_odom_xy = start_xy
        self.active_portal_gate_odom_xy = (5.0, 0.0)
        self.active_portal_destination_odom_xy = goal_xy
        self.active_portal_gate_approached_at = None
        self.active_portal_crossing_preobserved = False
        self.active_route_id = 7
        self.portal_transaction = None
        self.last_portal_hypothesis_id = 1
        self.portal_hypothesis_ledger = _Ledger(record)


class PortalCrossingEvidenceTest(unittest.TestCase):
    def test_crossing_depth_uses_door_geometry_not_room_coverage(self):
        # A 0.52 m certified-clearance opening is crossed once the base centre
        # is 0.26 m on its far side. A 1.25 m room-observation radius would
        # incorrectly reject that physical transition.
        self.assertAlmostEqual(portal_crossing_depth(0.52), 0.26)
        evidence = portal_crossing_evidence(
            start_xy=(2.0, 0.0),
            gate_xy=(5.0, 0.0),
            destination_xy=(7.0, 0.0),
            terminal_xy=(5.284, 0.0),
            gate_approached=True,
            minimum_destination_depth=portal_crossing_depth(0.52),
        )
        self.assertTrue(evidence.verified)

    def test_portal_command_uses_its_frozen_odom_destination(self):
        frame, goal = portal_execution_goal(
            "portal_transition", "map", (10.1, 13.2), (10.0, 13.0)
        )
        self.assertEqual(frame, "odom")
        self.assertEqual(goal, (10.0, 13.0))

        frame, goal = portal_execution_goal(
            "frontier_endpoint", "map", (10.1, 13.2), (10.0, 13.0)
        )
        self.assertEqual(frame, "map")
        self.assertEqual(goal, (10.1, 13.2))

    def test_destination_side_depth_proves_a_crossing(self):
        evidence = portal_crossing_evidence(
            start_xy=(2.0, 0.0),
            gate_xy=(5.0, 0.0),
            destination_xy=(7.0, 0.0),
            terminal_xy=(5.6, 0.0),
            gate_approached=True,
            minimum_destination_depth=0.4,
        )

        self.assertTrue(evidence.verified)
        self.assertEqual(evidence.reason, "physical_gate_crossed")
        self.assertAlmostEqual(evidence.terminal_signed_distance, 0.6)

    def test_continuous_directed_plane_crossing_does_not_need_gate_sample(self):
        """A sparse planning tick can miss the gate centre on a valid route."""
        evidence = portal_crossing_evidence(
            start_xy=(2.0, 0.0),
            gate_xy=(5.0, 0.0),
            destination_xy=(7.0, 0.0),
            terminal_xy=(5.6, 0.0),
            gate_approached=False,
            minimum_destination_depth=0.4,
        )

        self.assertTrue(evidence.verified)
        self.assertEqual(evidence.reason, "physical_gate_crossed")

    def test_goal_circle_success_on_source_side_does_not_cross(self):
        evidence = portal_crossing_evidence(
            start_xy=(2.0, 0.0),
            gate_xy=(5.0, 0.0),
            destination_xy=(7.0, 0.0),
            terminal_xy=(4.9, 0.0),
            gate_approached=True,
            minimum_destination_depth=0.4,
        )

        self.assertFalse(evidence.verified)
        self.assertEqual(evidence.reason, "physical_destination_side_not_reached")

    def test_no_start_side_proof_does_not_close_a_source_place(self):
        evidence = portal_crossing_evidence(
            start_xy=(5.2, 0.0),
            gate_xy=(5.0, 0.0),
            destination_xy=(7.0, 0.0),
            terminal_xy=(6.0, 0.0),
            gate_approached=True,
            minimum_destination_depth=0.4,
        )

        self.assertFalse(evidence.verified)
        self.assertEqual(evidence.reason, "physical_source_side_unproven")

    def test_transaction_source_proof_allows_retry_from_the_gate(self):
        """A second route starts at the gate but still belongs to one crossing."""
        evidence = portal_crossing_evidence(
            start_xy=(5.2, 0.0),
            gate_xy=(5.0, 0.0),
            destination_xy=(7.0, 0.0),
            terminal_xy=(5.8, 0.0),
            gate_approached=True,
            minimum_destination_depth=0.4,
            source_side_proven=True,
        )

        self.assertTrue(evidence.verified)
        self.assertEqual(evidence.reason, "physical_gate_crossed")

    def test_destination_view_can_supply_explicit_preobserved_crossing(self):
        evidence = portal_crossing_evidence(
            start_xy=(5.8, 0.0),
            gate_xy=(5.0, 0.0),
            destination_xy=(7.0, 0.0),
            terminal_xy=(5.8, 0.0),
            gate_approached=False,
            minimum_destination_depth=0.4,
            preobserved=True,
        )

        self.assertTrue(evidence.verified)
        self.assertEqual(evidence.reason, "portal_crossing_preobserved")

    def test_crossing_plane_uses_the_doorway_normal_not_the_deep_goal(self):
        """Regression: a route toward a goal beyond the door 'crossed' instantly.

        The certified record's gate pair defines the doorway axis. A terminal
        that only advanced toward a navigation goal ten metres past the gate --
        without passing the doorway's own normal plane -- must not verify.
        """
        record = {
            "physical_gate": [5.0, 0.0],
            "physical_destination": [5.26, 0.0],
        }
        # North of the gate, well short of its east-facing normal plane.
        harness = CrossingPlaneHarness(
            terminal_xy=(5.0, 3.0),
            goal_xy=(5.0, 10.0),
            record=record,
        )

        gate_xy, destination_xy = harness.active_portal_crossing_plane()
        self.assertEqual(gate_xy, (5.0, 0.0))
        self.assertAlmostEqual(destination_xy[0], 5.26)
        self.assertAlmostEqual(destination_xy[1], 0.0)

        evidence = harness.active_portal_crossing_evidence()
        self.assertFalse(evidence.verified)
        self.assertEqual(
            evidence.reason, "physical_destination_side_not_reached",
        )

    def test_crossing_plane_still_verifies_a_real_passage(self):
        record = {
            "physical_gate": [5.0, 0.0],
            "physical_destination": [5.26, 0.0],
        }
        harness = CrossingPlaneHarness(
            terminal_xy=(5.4, 0.1),
            goal_xy=(5.0, 10.0),
            record=record,
        )

        evidence = harness.active_portal_crossing_evidence()

        self.assertTrue(evidence.verified)
        self.assertEqual(evidence.reason, "physical_gate_crossed")

    def test_missing_ledger_record_keeps_the_historical_goal_plane(self):
        """No certified record means no doorway axis; fall back unchanged."""
        harness = CrossingPlaneHarness(
            terminal_xy=(5.0, 3.0),
            goal_xy=(5.0, 10.0),
            record=None,
            # South of the gate: on the goal-directed plane's source side.
            start_xy=(5.0, -1.0),
        )

        evidence = harness.active_portal_crossing_evidence()

        self.assertTrue(evidence.verified)

    def test_directional_portal_requires_a_source_side_start(self):
        self.assertAlmostEqual(
            portal_signed_distance((2.0, 0.0), (5.0, 0.0), (7.0, 0.0)),
            -3.0,
        )
        self.assertTrue(
            portal_source_side_is_proven(
                (2.0, 0.0), (5.0, 0.0), (7.0, 0.0),
            )
        )
        self.assertFalse(
            portal_source_side_is_proven(
                (5.2, 0.0), (5.0, 0.0), (7.0, 0.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
