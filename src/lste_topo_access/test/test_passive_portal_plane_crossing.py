#!/usr/bin/env python3
"""Unit tests for passive portal plane crossing certification and dedicated probe transaction switch."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault("rospy", SimpleNamespace())

from global_frontier_portal_crossing import (
    portal_signed_distance,
    portal_plane_cross_lateral_distance,
    check_portal_plane_crossing,
)
from global_frontier_portal_probe_ledger import (
    PortalProbeLedger,
    PROBE_PENDING,
    PROBE_CERTIFIED,
)
from global_frontier_portal_belief import PortalHypothesisLedger
from global_frontier_place_memory import FrontierRegionMemory
from global_frontier_portal_live_crossing import GlobalFrontierPortalLiveCrossingMixin
from global_frontier_graph_route_planner import (
    GraphRoutePlanner,
    ACTION_PROBE_PORTAL,
    ACTION_OBSERVE_LOCAL_WORK,
    PLAN_READY,
)


class DummyExplorer(GlobalFrontierPortalLiveCrossingMixin):
    """Test harness combining live crossing mixin with ledgers and region memory."""

    def __init__(
        self,
        enable_dedicated_portal_probe_transaction=False,
        passive_plane_crossing_certification=True,
    ):
        self.enable_dedicated_portal_probe_transaction = (
            enable_dedicated_portal_probe_transaction
        )
        self.passive_plane_crossing_certification = (
            passive_plane_crossing_certification
        )
        self.clearance = 0.52
        self.current_physical_place_id = 1
        self.current_task_version = "v1"
        self._portal_plane_prev_signed = {}
        self.published_events = []
        self.pose_odom = SimpleNamespace(x=0.0, y=0.0)

        self.region_memory = FrontierRegionMemory(
            radius=2.5,
            information_delta=8.0,
            stagnation_timeout=12.0,
            failure_limit=2,
            limit=128,
        )
        # Register place 1 as current
        self.region_memory.enter(
            0.0, 0.0, 0.0, force_new=True,
            component={
                "epoch": 1, "label": 1, "cells": 10,
                "center_x": 0.0, "center_y": 0.0,
                "min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0,
            },
        )
        self.portal_hypothesis_ledger = PortalHypothesisLedger(match_radius=0.5)
        self.portal_probe_ledger = PortalProbeLedger(match_radius=0.5)

    def publish_status(self, event, **fields):
        self.published_events.append((event, fields))

    def now_for(self, _obj):
        return 1.0


class PassivePortalPlaneCrossingTest(unittest.TestCase):
    """Verify geometry, lifecycle certification, and planner toggle switch."""

    def test_portal_plane_geometry(self):
        """Verify signed distance, lateral window, and plane crossing detection."""
        gate_xy = (2.0, 0.0)
        dest_xy = (3.0, 0.0)

        # Source side: robot at (1.5, 0.0)
        dist_src = portal_signed_distance((1.5, 0.0), gate_xy, dest_xy)
        self.assertAlmostEqual(dist_src, -0.5)

        # Lateral distance: robot at (1.5, 0.8)
        lat_dist = portal_plane_cross_lateral_distance((1.5, 0.8), gate_xy, dest_xy)
        self.assertAlmostEqual(lat_dist, 0.8)

        # Crossing check: source side (-0.5) to dest side (+0.2)
        crossed, current_signed, _lat = check_portal_plane_crossing(
            point_xy=(2.2, 0.0),
            gate_xy=gate_xy,
            destination_xy=dest_xy,
            previous_signed_distance=-0.5,
            lateral_tolerance=1.2,
        )
        self.assertTrue(crossed)
        self.assertAlmostEqual(current_signed, 0.2)

        # Rejection: lateral distance outside tolerance (e.g. through adjacent wall)
        crossed_wide, _, _ = check_portal_plane_crossing(
            point_xy=(2.2, 2.5),
            gate_xy=gate_xy,
            destination_xy=dest_xy,
            previous_signed_distance=-0.5,
            lateral_tolerance=1.2,
        )
        self.assertFalse(crossed_wide)

        # Rejection: robot moved further into source side, not forward
        crossed_back, current_signed, _ = check_portal_plane_crossing(
            point_xy=(1.0, 0.0),
            gate_xy=gate_xy,
            destination_xy=dest_xy,
            previous_signed_distance=-0.5,
            lateral_tolerance=1.2,
        )
        self.assertFalse(crossed_back)
        self.assertAlmostEqual(current_signed, -1.0)

    def test_passive_crossing_certifies_probe_and_enters_place(self):
        """A physical base transition across doorway plane certifies portal and enters place."""
        explorer = DummyExplorer(
            enable_dedicated_portal_probe_transaction=False,
            passive_plane_crossing_certification=True,
        )
        gate_xy = (2.0, 0.0)
        dest_xy = (3.0, 0.0)

        # Register a probe in pending state
        probe_rec = explorer.portal_probe_ledger.observe(
            source_place_id=1,
            physical_gate_xy=gate_xy,
            normal_xy=(1.0, 0.0),
            now=0.0,
            map_gate_xy=gate_xy,
            observation_source="structural_boundary",
        )
        probe_id = probe_rec["id"]
        self.assertEqual(probe_rec["state"], PROBE_PENDING)

        # Step 1: Robot approaches on source side (x=1.5, y=0.1)
        explorer.pose_odom = SimpleNamespace(x=1.5, y=0.1)
        committed1 = explorer.observe_passive_portal_plane_crossing(now=1.0)
        self.assertIsNone(committed1)
        self.assertEqual(explorer.current_physical_place_id, 1)

        # Step 2: Robot crosses doorway plane to destination side (x=2.3, y=0.1)
        explorer.pose_odom = SimpleNamespace(x=2.3, y=0.1)
        committed2 = explorer.observe_passive_portal_plane_crossing(now=2.0)
        self.assertIsNotNone(committed2)

        # Verify probe promoted to certified
        probe_after = explorer.portal_probe_ledger.get(probe_id)
        self.assertEqual(probe_after["state"], PROBE_CERTIFIED)
        self.assertEqual(probe_after["last_reason"], "passive_plane_crossing_certified")

        # Verify hypothesis created and crossed
        portal_id = probe_after.get("portal_id")
        self.assertIsNotNone(portal_id)
        hypo = explorer.portal_hypothesis_ledger.get(portal_id)
        self.assertEqual(hypo["state"], "crossed")

        # Verify Place transition
        self.assertNotEqual(explorer.current_physical_place_id, 1)
        self.assertEqual(explorer.current_physical_place_id, hypo["destination_place_id"])

        # Verify status publications
        event_names = [e[0] for e in explorer.published_events]
        self.assertIn("portal_hypothesis_certified", event_names)
        self.assertIn("portal_probe_promoted", event_names)
        self.assertIn("portal_place_entered", event_names)

    def test_passive_crossing_disabled_switch(self):
        """When passive_plane_crossing_certification is False, crossing plane does not certify."""
        explorer = DummyExplorer(
            enable_dedicated_portal_probe_transaction=False,
            passive_plane_crossing_certification=False,
        )
        gate_xy = (2.0, 0.0)
        probe_rec = explorer.portal_probe_ledger.observe(
            source_place_id=1,
            physical_gate_xy=gate_xy,
            normal_xy=(1.0, 0.0),
            now=0.0,
            map_gate_xy=gate_xy,
        )
        probe_id = probe_rec["id"]

        # Approach from source side
        explorer.pose_odom = SimpleNamespace(x=1.5, y=0.0)
        explorer.observe_passive_portal_plane_crossing(now=1.0)

        # Cross to destination side
        explorer.pose_odom = SimpleNamespace(x=2.5, y=0.0)
        committed = explorer.observe_passive_portal_plane_crossing(now=2.0)
        self.assertIsNone(committed)

        # Probe remains pending and place does not change
        probe_after = explorer.portal_probe_ledger.get(probe_id)
        self.assertEqual(probe_after["state"], PROBE_PENDING)
        self.assertEqual(explorer.current_physical_place_id, 1)

    def test_enable_dedicated_portal_probe_transaction_switch(self):
        """Verify planner respects enable_dedicated_portal_probe_transaction flag."""
        places = [{"id": 1, "state": "open", "endpoint_observations": 1}]
        portals = [{
            "id": 10,
            "source_place_id": 1,
            "destination_place_id": None,
            "state": "certified",
        }]
        probes = [{
            "id": 20,
            "source_place_id": 1,
            "portal_id": 10,
            "state": "pending",
        }]
        work_items = [{
            "id": 30,
            "place_id": 1,
            "state": "unresolved",
            "active_attempt_id": None,
        }]

        # 1. When dedicated probe transactions are DISABLED (default: False):
        # Planner prioritizes local work item and suppresses probe_portal actions.
        planner_disabled = GraphRoutePlanner(enable_dedicated_portal_probe_transaction=False)
        plan_disabled = planner_disabled.plan(
            current_place_id=1,
            places=places,
            portals=portals,
            probes=probes,
            work_items=work_items,
            branch_first=True,
            structural_boundary_first=True,
        )
        self.assertEqual(plan_disabled.status, PLAN_READY)
        self.assertEqual(plan_disabled.action, ACTION_OBSERVE_LOCAL_WORK)
        self.assertEqual(plan_disabled.obligation_id, 30)

        # 2. When dedicated probe transactions are ENABLED:
        # Planner generates dedicated probe_portal action for the unbound doorway.
        planner_enabled = GraphRoutePlanner(enable_dedicated_portal_probe_transaction=True)
        plan_enabled = planner_enabled.plan(
            current_place_id=1,
            places=places,
            portals=portals,
            probes=probes,
            work_items=work_items,
            branch_first=True,
            structural_boundary_first=True,
        )
        self.assertEqual(plan_enabled.status, PLAN_READY)
        self.assertEqual(plan_enabled.action, ACTION_PROBE_PORTAL)
        self.assertEqual(plan_enabled.obligation_id, 20)


if __name__ == "__main__":
    unittest.main()
