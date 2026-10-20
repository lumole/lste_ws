"""Regression tests for physical-place-owned frontier work."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from global_frontier_observation_coverage import GlobalFrontierObservationCoverageMixin
from global_frontier_place_memory import FrontierRegionMemory
from global_frontier_portal_probe_ledger import PortalProbeLedger
from global_frontier_work_items import (
    ATTEMPT_FAILED,
    PlaceWorkItemLedger,
    WORK_DEFERRED,
    WORK_RESOLVED,
    WORK_UNRESOLVED,
)


def component(label):
    return {
        "epoch": 1,
        "label": label,
        "cells": 24,
        "center_x": float(label),
        "center_y": 0.0,
        "min_x": float(label) - 0.5,
        "max_x": float(label) + 0.5,
        "min_y": -0.5,
        "max_y": 0.5,
    }


class ActivationProbe(GlobalFrontierObservationCoverageMixin):
    def __init__(self):
        self.region_memory = FrontierRegionMemory(
            radius=2.5,
            information_delta=4.0,
            stagnation_timeout=10.0,
            failure_limit=2,
            limit=8,
        )
        self.last_frontier_region_tier = None


class PlaceWorkItemTest(unittest.TestCase):
    def test_local_component_split_reuses_physical_owner(self):
        probe = ActivationProbe()
        owner, _tier = probe.region_memory.activate(
            1.0, 0.0, 8.0, now=1.0, component=component(1),
        )

        with patch("global_frontier_observation_coverage.rospy.loginfo", create=True):
            selected = probe.activate_frontier_region(
                4.0,
                0.0,
                12.0,
                now=2.0,
                transition="test",
                component=component(2),
                physical_place_id=owner["id"],
            )

        self.assertEqual(selected["id"], owner["id"])
        self.assertEqual(len(probe.region_memory.regions), 1)
        self.assertEqual(selected["last_association"], "current_physical_place_owner")

    def test_selecting_local_work_reopens_suspended_place(self):
        probe = ActivationProbe()
        owner, _tier = probe.region_memory.activate(
            1.0, 0.0, 8.0, now=1.0, component=component(1),
        )
        probe.region_memory.endpoint_observed(
            1.0, 0.0, now=1.5, component=component(1),
            region_id=owner["id"],
        )
        probe.region_memory.suspend(owner, now=2.0)

        with patch("global_frontier_observation_coverage.rospy.loginfo", create=True):
            selected = probe.activate_frontier_region(
                1.5,
                0.0,
                12.0,
                now=3.0,
                transition="branch_resume",
                component=component(2),
                physical_place_id=owner["id"],
            )

        self.assertIs(selected, owner)
        self.assertEqual(selected["state"], "open")
        self.assertEqual(selected["last_reason"], "suspended_place_reopened")

    def test_failed_viewpoint_keeps_work_unresolved_for_an_alternative_viewpoint(self):
        ledger = PlaceWorkItemLedger(merge_radius=0.5)
        first = ledger.dispatch(2, (8.0, 3.0), now=1.0)
        ledger.settle(first, WORK_RESOLVED, now=2.0, reason="observed")
        self.assertFalse(ledger.candidate_is_available(2, (8.2, 3.1)))

        second = ledger.dispatch(2, (10.0, 3.0), now=3.0)
        ledger.settle(second, WORK_DEFERRED, now=4.0, reason="blocked")
        item = ledger._items[second]
        self.assertEqual(item["state"], WORK_UNRESOLVED)
        self.assertEqual(item["attempts"][-1]["state"], ATTEMPT_FAILED)
        # A route failure blocks the same physical standoff point, not the
        # unknown-side observation boundary itself.
        self.assertFalse(ledger.candidate_is_available(2, (10.2, 3.1)))
        self.assertTrue(ledger.candidate_is_available(2, (12.0, 3.0)))
        retry = ledger.dispatch_existing(
            second,
            now=5.0,
            viewpoint_xy=(12.0, 3.0),
            map_goal=(12.0, 3.0),
        )
        self.assertEqual(retry, second)
        self.assertNotEqual(
            ledger._items[second]["attempts"][-1]["id"],
            ledger._items[second]["attempts"][-2]["id"],
        )

    def test_route_rejection_blocks_only_pre_dispatch_viewpoint(self):
        ledger = PlaceWorkItemLedger(merge_radius=0.5)
        item = ledger._new_item(
            2,
            frozenset({(4, 4)}),
            now=1.0,
            anchor_xy=(2.0, 2.0),
            normal_xy=(1.0, 0.0),
        )

        _item, rejection = ledger.record_route_rejection(
            item["id"],
            now=2.0,
            viewpoint_xy=(1.0, 2.0),
            map_goal=(1.0, 2.0),
            reason="navfn_endpoint_offset",
            map_epoch=7,
            plan_endpoint=(1.0, 1.8),
        )

        self.assertIsNotNone(rejection)
        self.assertIsNone(ledger.active_attempt_id(item["id"]))
        self.assertFalse(
            ledger.viewpoint_is_available(item["id"], (1.0, 2.0), map_epoch=7)
        )
        self.assertTrue(
            ledger.viewpoint_is_available(item["id"], (1.0, 3.0), map_epoch=7)
        )
        self.assertTrue(ledger.has_failed_viewpoint(item["id"], map_epoch=7))
        # The rejection remains in the audit trail, but a fresh map epoch may
        # reassess the same physical standoff instead of inheriting a stale
        # costmap decision.
        self.assertTrue(
            ledger.viewpoint_is_available(item["id"], (1.0, 2.0), map_epoch=8)
        )
        record = ledger.snapshot()[0]
        self.assertEqual(record["route_rejection_count"], 1)
        self.assertEqual(
            record["route_rejections"][0]["reason"],
            "navfn_endpoint_offset",
        )

    def test_portal_owned_work_is_not_counted_as_local_observation(self):
        work = PlaceWorkItemLedger(merge_radius=0.5)
        item_id = work.dispatch(2, (8.0, 3.0), now=1.0)
        probes = PortalProbeLedger(match_radius=0.5)
        probe = probes.observe(2, (8.0, 3.0), (1.0, 0.0), now=1.0)
        probes.bind_work_item(probe["id"], item_id)

        self.assertEqual(work.unresolved_count(2), 1)
        self.assertEqual(
            work.unresolved_observation_count(2, portal_probe_ledger=probes),
            0,
        )

    def test_dead_corner_abandonment_marks_blacklist_and_resolves(self):
        work = PlaceWorkItemLedger(merge_radius=0.5, density_radius=1.0)
        item = work._new_item(
            place_id=2,
            support_cells=frozenset({(10, 10)}),
            now=1.0,
            anchor_xy=(5.0, 5.0),
            normal_xy=(1.0, 0.0),
        )
        item_id = item["id"]

        # Abandon as dead corner
        resolved = work.resolve(item_id, now=2.0, reason="dead_corner_abandoned_controller_stall")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["state"], WORK_RESOLVED)
        self.assertTrue(work.is_dead_corner((5.0, 5.0)))
        self.assertTrue(work.is_dead_corner((5.2, 5.1), radius=0.6))
        self.assertFalse(work.is_dead_corner((6.0, 6.0), radius=0.6))

        # Dead corner blacklist prevents recreation in reconcile
        from global_frontier_work_item_components import ObservationSupport
        supp = ObservationSupport(
            frontier_cells=frozenset({(1, 1)}),
            support_cells=frozenset({(10, 10)}),
            anchor_xy=(5.1, 5.0),
            normal_xy=(1.0, 0.0),
        )
        cells, details = work.reconcile(2, (supp,), now=3.0)
        self.assertEqual(len(cells), 0)
        self.assertEqual(len(details), 0)

    def test_record_unreachable_viewpoint_abandonment(self):
        work = PlaceWorkItemLedger(merge_radius=0.5)
        item = work._new_item(
            place_id=2,
            support_cells=frozenset({(10, 10)}),
            now=1.0,
            anchor_xy=(2.0, 3.0),
            normal_xy=(1.0, 0.0),
        )
        item_id = item["id"]

        # First failure: not yet abandoned
        abandoned1 = work.record_unreachable_viewpoint(item_id, now=1.0, threshold=2)
        self.assertFalse(abandoned1)
        self.assertEqual(item["state"], WORK_UNRESOLVED)
        self.assertEqual(item["unreachable_viewpoint_count"], 1)

        # Second failure: abandoned as dead corner
        abandoned2 = work.record_unreachable_viewpoint(item_id, now=2.0, threshold=2)
        self.assertTrue(abandoned2)
        self.assertEqual(item["state"], WORK_RESOLVED)
        self.assertEqual(item["last_reason"], "dead_corner_unreachable")
        self.assertTrue(work.is_dead_corner((2.0, 3.0)))

    def test_prune_passed_through_work_items_resolves_when_close_and_aligned(self):
        work = PlaceWorkItemLedger(merge_radius=0.5)
        item = work._new_item(
            place_id=1,
            support_cells=frozenset({(8, 8)}),
            now=1.0,
            anchor_xy=(2.0, 2.0),
            normal_xy=(1.0, 0.0),  # heading = 0.0
        )
        item_id = item["id"]

        # Robot at (2.2, 2.1) -> dist = 0.223m <= 0.4m, heading = 0.1 rad (~5.7 deg <= 30 deg)
        # Not the active target (active_target = 999)
        robot_pose = (2.2, 2.1, 0.1)
        pruned = work.prune_passed_through_work_items(
            robot_pose,
            active_work_item_id=999,
            now=2.0,
            distance_threshold=0.4,
            angle_threshold_deg=30.0,
        )
        self.assertIn(item_id, pruned)
        self.assertEqual(item["state"], WORK_RESOLVED)
        self.assertEqual(item["resolved_reason"], "opportunistic_pass_through")

    def test_prune_passed_through_work_items_ignores_when_angle_mismatched(self):
        work = PlaceWorkItemLedger(merge_radius=0.5)
        item = work._new_item(
            place_id=1,
            support_cells=frozenset({(8, 8)}),
            now=1.0,
            anchor_xy=(2.0, 2.0),
            normal_xy=(1.0, 0.0),  # heading = 0.0
        )
        item_id = item["id"]

        # Robot at (2.2, 2.1) -> dist = 0.223m <= 0.4m, but heading = 1.57 rad (90 deg > 30 deg)
        robot_pose = (2.2, 2.1, 1.57)
        pruned = work.prune_passed_through_work_items(
            robot_pose,
            active_work_item_id=999,
            now=2.0,
            distance_threshold=0.4,
            angle_threshold_deg=30.0,
        )
        self.assertEqual(pruned, [])
        self.assertEqual(item["state"], WORK_UNRESOLVED)

    def test_prune_passed_through_work_items_ignores_when_distance_too_far(self):
        work = PlaceWorkItemLedger(merge_radius=0.5)
        item = work._new_item(
            place_id=1,
            support_cells=frozenset({(8, 8)}),
            now=1.0,
            anchor_xy=(2.0, 2.0),
            normal_xy=(1.0, 0.0),
        )
        item_id = item["id"]

        # Robot at (2.5, 2.0) -> dist = 0.5m > 0.4m, even though angle is 0
        robot_pose = (2.5, 2.0, 0.0)
        pruned = work.prune_passed_through_work_items(
            robot_pose,
            active_work_item_id=999,
            now=2.0,
            distance_threshold=0.4,
            angle_threshold_deg=30.0,
        )
        self.assertEqual(pruned, [])
        self.assertEqual(item["state"], WORK_UNRESOLVED)

    def test_prune_passed_through_work_items_never_resolves_active_target(self):
        work = PlaceWorkItemLedger(merge_radius=0.5)
        item = work._new_item(
            place_id=1,
            support_cells=frozenset({(8, 8)}),
            now=1.0,
            anchor_xy=(2.0, 2.0),
            normal_xy=(1.0, 0.0),
        )
        item_id = item["id"]

        # Robot at (2.1, 2.0, 0.0), BUT this item IS the active target being executed!
        robot_pose = (2.1, 2.0, 0.0)
        pruned = work.prune_passed_through_work_items(
            robot_pose,
            active_work_item_id=item_id,
            now=2.0,
            distance_threshold=0.4,
            angle_threshold_deg=30.0,
        )
        self.assertEqual(pruned, [])
        self.assertEqual(item["state"], WORK_UNRESOLVED)

    def test_unresolved_work_item_keeps_stable_anchor_and_absorbs_support(self):
        from types import SimpleNamespace
        work = PlaceWorkItemLedger(merge_radius=0.5)
        item = work._new_item(
            place_id=1,
            support_cells=frozenset({(8, 8)}),
            now=1.0,
            anchor_xy=(2.0, 2.0),
            normal_xy=(1.0, 0.0),
        )
        support = SimpleNamespace(
            support_cells=frozenset({(8, 8), (9, 8)}),
            frontier_cells=frozenset({(8, 8)}),
            anchor_xy=(2.4, 2.1),
            normal_xy=(1.0, 0.0),
        )
        _cells, details = work.reconcile(1, [support], now=2.0)
        self.assertEqual(item["anchor_xy"], (2.0, 2.0))
        self.assertIn((9, 8), item["support_cells"])


if __name__ == "__main__":
    unittest.main()

