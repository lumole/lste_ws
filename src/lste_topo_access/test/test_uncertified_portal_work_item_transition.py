#!/usr/bin/env python3
"""Tests for Scheme A pre-interception and Scheme B post-hoc transitions across uncertified doorways."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    import rospy
except ImportError:
    rospy = SimpleNamespace()
    sys.modules["rospy"] = rospy
for fn in ("loginfo", "logwarn", "logerr"):
    if not hasattr(rospy, fn):
        setattr(rospy, fn, lambda *args, **kwargs: None)

from global_frontier_candidate_place_policy import (  # noqa: E402
    GlobalFrontierCandidatePlacePolicyMixin,
)
from global_frontier_observation_coverage import (  # noqa: E402
    GlobalFrontierObservationCoverageMixin,
)
from global_frontier_place_memory import FrontierRegionMemory  # noqa: E402
from global_frontier_portal_arrival_commit import (  # noqa: E402
    GlobalFrontierPortalArrivalCommitMixin,
)
from global_frontier_portal_belief import PortalHypothesisLedger  # noqa: E402
from global_frontier_work_items import PlaceWorkItemLedger  # noqa: E402


class PlacePolicyStub(GlobalFrontierCandidatePlacePolicyMixin):
    def __init__(self):
        self.last_place_graph_unresolved_candidates = 0
        self.last_uncertified_place_transition_skips = 0
        self.last_cross_place_endpoint_deferrals = 0
        self.last_place_graph_hop_skips = 0
        self.last_portal_missing_transition_goals = 0
        self.place_graph_hop_limit = 2
        self.clearance = 0.35
        self.region_topology_clearance = 0.50
        self.place_furniture_max_span_m = 2.5
        self._portal_transition_certification_cache = {}
        self.registered_probes = []

    def _portal_wall_support_kwargs(self, _message, components):
        return {
            "structural_occupied": getattr(components, "structural_occupied", None),
            "minimum_wall_span_cells": 5,
            "minimum_opening_cells": 2,
            "wall_search_radius": 4,
        }

    def cell_xy(self, _message, row, col):
        return float(col) * 0.1, float(row) * 0.1

    def sealed_portal_reentry(self, *_args, **_kwargs):
        return None

    def unresolved_portal_probe(self, _request, _row, _col):
        return None

    def _queue_first_portal_action(self, *_args, **_kwargs):
        pass

    def register_portal_probe(self, _request, probe):
        self.registered_probes.append(probe)
        return probe


class ExplorerStub(
    GlobalFrontierObservationCoverageMixin,
    GlobalFrontierPortalArrivalCommitMixin,
):
    def __init__(self):
        self.region_memory = FrontierRegionMemory(
            radius=3.0,
            information_delta=5.0,
            stagnation_timeout=30.0,
            failure_limit=3,
            limit=64,
        )
        self.portal_hypothesis_ledger = PortalHypothesisLedger(match_radius=0.5)
        self.place_work_items = PlaceWorkItemLedger(merge_radius=0.5)
        self.current_physical_place_id = None
        self.active_frontier_region_id = None
        self.active_work_item_place_id = None
        self.active_work_item_id = None
        self.active_portal_probe_id = None
        self.active_route_kind = "frontier_endpoint"
        self.target_region_claim_active = False
        self.completed_frontiers = []
        self.published_events = []
        self.pose_odom = None
        self.enable_post_hoc_place_transition = True

    def publish_status(self, event, **fields):
        self.published_events.append((event, fields))

    def frontier_is_completed(self, _x, _y):
        return False

    def active_work_item_support_is_observed(self):
        return True

    def settle_active_work_item(self, now, state, reason):
        if self.active_work_item_id is not None:
            self.place_work_items.settle(
                self.active_work_item_id, state, now, reason,
            )


class UncertifiedPortalWorkItemTransitionTest(unittest.TestCase):
    def test_place_work_item_ledger_reattribution(self):
        """Test PlaceWorkItemLedger reattributing individual and nearby items."""
        ledger = PlaceWorkItemLedger(merge_radius=0.5)
        item1 = ledger._new_item(1, frozenset(), 10.0, xy=(1.0, 1.0), anchor_xy=(1.0, 1.0))
        item2 = ledger._new_item(1, frozenset(), 10.0, xy=(1.2, 1.1), anchor_xy=(1.2, 1.1))
        item3 = ledger._new_item(1, frozenset(), 10.0, xy=(5.0, 5.0), anchor_xy=(5.0, 5.0))
        item1_id = int(item1["id"])
        item2_id = int(item2["id"])
        item3_id = int(item3["id"])

        # Single item reattribution
        updated = ledger.reattribute_work_item(item1_id, new_place_id=2, now=11.0)
        self.assertIsNotNone(updated)
        self.assertEqual(updated["place_id"], 2)
        self.assertEqual(updated["last_reason"], "place_reattributed")

        # Radius-based reattribution near (1.1, 1.0)
        reattributed = ledger.reattribute_items_near(
            old_place_id=1, new_place_id=2, physical_xy=(1.1, 1.0), radius=0.8, now=12.0,
        )
        self.assertIn(item2_id, reattributed)
        self.assertNotIn(item3_id, reattributed)
        self.assertEqual(ledger._items[item2_id]["place_id"], 2)
        self.assertEqual(ledger._items[item3_id]["place_id"], 1)

    def test_scheme_a_uncertified_portal_pre_interception(self):
        """Scheme A: Route crossing uncertified boundary is intercepted as probe candidate."""
        policy = PlacePolicyStub()

        # Grid setup: 10x10 map with vertical wall dividing col 4 and 5, door throat at row 5
        labels = np.ones((10, 10), dtype=int)
        labels[:, 5:] = 2  # Label 2 on the right side
        labels[:, 4] = 0   # Doorway throat / boundary at column 4

        steps = np.full((10, 10), -1, dtype=int)
        # Simple straight path from (5, 1) to (5, 8)
        for c in range(1, 9):
            steps[5, c] = c - 1

        structural_occupied = np.zeros((10, 10), dtype=bool)
        # Wall jambs above and below door throat (row 5, col 4)
        structural_occupied[0:4, 4] = True
        structural_occupied[7:10, 4] = True

        components = SimpleNamespace(
            labels=labels,
            structural_occupied=structural_occupied,
            epoch=1,
        )
        message = SimpleNamespace(
            info=SimpleNamespace(
                resolution=0.1,
                width=10,
                height=10,
                origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
            ),
        )
        request = SimpleNamespace(
            components=components,
            message=message,
            steps=steps,
            unknown=np.zeros((10, 10), dtype=bool),
            occupied=structural_occupied,
            allow_portal_transitions=True,
            map_to_physical_xy=lambda x, y: (x, y),
        )
        context = SimpleNamespace(
            covered_place_reachable=None,
            place_graph_ready=True,
            source_label=1,
            source_place_id=1,
            source_place_observed=False,
            observed_reentry_labels=set(),
        )

        # Target is at (5, 8) in label 2, starting from source_label 1
        result = policy._classify_candidate_approach(
            request=request,
            target_row=5,
            target_col=8,
            frontier_row=5,
            frontier_col=8,
            context=context,
            portal_sources={},
        )

        self.assertIsNotNone(result)
        tier, hops, gate_xy, probe = result
        self.assertEqual(tier, "probe")
        self.assertEqual(hops, 0)
        self.assertIsNotNone(gate_xy)
        self.assertIsNotNone(probe)
        self.assertEqual(probe.observation_source, "structural_boundary")
        # Opening cell should be at throat/doorway around column 4
        self.assertEqual(probe.opening_cell[1], 4)

    def test_scheme_b_post_hoc_place_and_portal_creation(self):
        """Scheme B: Reaching WorkItem across doorway retroactively builds Place and Portal."""
        explorer = ExplorerStub()

        # Step 1: Establish initial Place 1 at origin
        init_region, _ = explorer.region_memory.enter(
            x=0.0, y=0.0, now=1.0, physical_xy=(0.0, 0.0),
        )
        old_place_id = init_region["id"]
        explorer.current_physical_place_id = old_place_id
        explorer.active_frontier_region_id = old_place_id

        # Step 2: Create a WorkItem in Place 1 targeting (4.0, 0.0) in the next room
        item = explorer.place_work_items._new_item(
            place_id=old_place_id, support_cells=frozenset(), now=2.0, xy=(4.0, 0.0), anchor_xy=(4.0, 0.0),
        )
        item_id = int(item["id"])
        explorer.active_work_item_id = item_id

        # Step 3: Robot physically arrives at (4.0, 0.0)
        # Component indicates a new space label 2
        comp_next_room = {
            "epoch": 1,
            "label": 2,
            "cells": 50,
            "center_x": 4.0,
            "center_y": 0.0,
            "min_x": 3.0,
            "max_x": 5.0,
            "min_y": -1.0,
            "max_y": 1.0,
        }

        explorer.pose_odom = SimpleNamespace(x=4.0, y=0.0)
        observed_region = explorer.mark_frontier_observed(
            x=4.0,
            y=0.0,
            component=comp_next_room,
            region_id=old_place_id,
            physical_xy=(4.0, 0.0),
            now=5.0,
        )

        # Verification 1: New Place established
        self.assertIsNotNone(observed_region)
        new_place_id = observed_region["id"]
        self.assertNotEqual(new_place_id, old_place_id)
        self.assertEqual(explorer.current_physical_place_id, new_place_id)
        self.assertEqual(explorer.active_frontier_region_id, new_place_id)

        # Verification 2: Portal hypothesis was certified, crossed, and destination-bound
        records = list(explorer.portal_hypothesis_ledger._records.values())
        self.assertTrue(len(records) >= 1)
        portal_record = records[0]
        self.assertEqual(portal_record["source_place_id"], old_place_id)
        self.assertEqual(portal_record["destination_place_id"], new_place_id)
        self.assertEqual(portal_record["state"], "crossed")

        # Verification 3: WorkItem was reattributed to new_place_id and resolved
        item = explorer.place_work_items._items[item_id]
        self.assertEqual(item["place_id"], new_place_id)
        self.assertEqual(item["state"], "resolved")
        self.assertEqual(item["last_reason"], "endpoint_coverage_recorded")

        # Verification 4: Status events published
        event_names = [e[0] for e in explorer.published_events]
        self.assertIn("work_item_promoted_to_new_place", event_names)
        self.assertIn("portal_place_entered", event_names)
        self.assertIn("frontier_endpoint_observed", event_names)

    def test_regression_same_place_work_item_no_post_hoc_transition(self):
        """Regression test: Reaching a WorkItem in the same room does NOT create false Place."""
        explorer = ExplorerStub()

        # Establish Place 1
        comp_room1 = {
            "epoch": 1,
            "label": 1,
            "cells": 80,
            "center_x": 0.5,
            "center_y": 0.5,
            "min_x": -1.0,
            "max_x": 2.0,
            "min_y": -1.0,
            "max_y": 2.0,
        }
        init_region, _ = explorer.region_memory.enter(
            x=0.0, y=0.0, now=1.0, component=comp_room1, physical_xy=(0.0, 0.0),
        )
        place_id = init_region["id"]
        explorer.current_physical_place_id = place_id
        explorer.active_frontier_region_id = place_id

        # Create WorkItem in same room at (0.8, 0.8)
        item = explorer.place_work_items._new_item(
            place_id=place_id, support_cells=frozenset(), now=2.0, xy=(0.8, 0.8), anchor_xy=(0.8, 0.8),
        )
        item_id = int(item["id"])
        explorer.active_work_item_id = item_id

        # Arrive at (0.8, 0.8) with same component label 1
        explorer.pose_odom = SimpleNamespace(x=0.8, y=0.8)
        observed_region = explorer.mark_frontier_observed(
            x=0.8,
            y=0.8,
            component=comp_room1,
            region_id=place_id,
            physical_xy=(0.8, 0.8),
            now=3.0,
        )

        self.assertIsNotNone(observed_region)
        self.assertEqual(observed_region["id"], place_id)
        self.assertEqual(explorer.current_physical_place_id, place_id)

        # No portal hypotheses should be created
        self.assertEqual(len(explorer.portal_hypothesis_ledger._records), 0)

        # WorkItem stays in place_id
        item = explorer.place_work_items._items[item_id]
        self.assertEqual(item["place_id"], place_id)

        # Events check: No post-hoc promotion events
        event_names = [e[0] for e in explorer.published_events]
        self.assertNotIn("work_item_promoted_to_new_place", event_names)
        self.assertIn("frontier_endpoint_observed", event_names)


if __name__ == "__main__":
    unittest.main()
