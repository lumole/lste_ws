"""Regression tests for durable place identity after a portal crossing."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

rospy = sys.modules.setdefault(
    "rospy",
    SimpleNamespace(loginfo=lambda *_args, **_kwargs: None, logwarn=lambda *_args, **_kwargs: None),
)
if not hasattr(rospy, "loginfo"):
    rospy.loginfo = lambda *_args, **_kwargs: None
if not hasattr(rospy, "logwarn"):
    rospy.logwarn = lambda *_args, **_kwargs: None

from global_frontier_place_memory import FrontierRegionMemory
from global_frontier_models import PendingPortalArrival
from global_frontier_portal_transaction import PortalTransaction
from global_frontier_portal_belief import PortalHypothesisLedger
from global_frontier_portal_lifecycle import GlobalFrontierPortalLifecycleMixin


class PortalExplorer(GlobalFrontierPortalLifecycleMixin):
    def __init__(self):
        self.region_memory = FrontierRegionMemory(
            radius=2.5,
            information_delta=8,
            stagnation_timeout=12,
            failure_limit=2,
            limit=16,
        )
        self.active_route_id = 19
        self.active_route_kind = "portal_transition"
        self.active_frontier = (10, 20, 8.0, 4.0)
        self.active_portal_retry = False
        self.active_portal_gate_xy = (7.2, 4.0)
        self.active_start_odom_xy = (5.0, 4.0)
        self.active_portal_gate_odom_xy = (7.2, 4.0)
        self.active_portal_destination_odom_xy = (8.5, 4.0)
        self.active_portal_gate_approached_at = 2.0
        self.pose_odom = SimpleNamespace(x=8.0, y=4.0)
        self.waypoint_release_radius = 0.4
        self.endpoint_terminal_wait_radius = 0.4
        self.clearance = 0.52
        self.active_frontier_component = {
            "epoch": 12,
            "label": 4,
            "cells": 80,
            "center_x": 8.0,
            "center_y": 4.0,
            "min_x": 7.0,
            "max_x": 9.0,
            "min_y": 3.0,
            "max_y": 5.0,
        }
        self.pending_portal_arrivals = []
        self.pending_portal_retry = None
        self.place_departure = SimpleNamespace(region_id=None)
        self.events = []

    def publish_status(self, event, **fields):
        self.events.append((event, fields))


class PortalPlaceArrivalTest(unittest.TestCase):
    def test_terminal_binds_next_selection_to_the_arrived_place(self):
        explorer = PortalExplorer()

        region = explorer.record_portal_place_arrival(
            (10, 20, 8.0, 4.0), now=3.0
        )

        self.assertIsNotNone(region)
        self.assertEqual(region["entered_at"], 3.0)
        self.assertEqual(region["observation_started_at"], 3.0)
        self.assertEqual(region["endpoint_observations"], 1)
        self.assertEqual(region["portal_arrivals"], 1)
        self.assertEqual(
            region["entry_portals"],
            [{
                "gate": [7.2, 4.0],
                "inside": [8.0, 4.0],
                "physical_gate": [7.2, 4.0],
                "physical_inside": [8.0, 4.0],
                "normal": [1.0, 0.0],
            }],
        )
        self.assertEqual(explorer.events[-1][0], "portal_place_entered")
        self.assertTrue(any(
            event == "portal_place_entry_observed"
            for event, _fields in explorer.events
        ))
        tier, matched = explorer.region_memory.candidate_tier(
            8.1, 4.0, 20.0, component=explorer.active_frontier_component
        )
        self.assertEqual(tier, "revisit")
        self.assertEqual(matched["id"], region["id"])

    def test_missing_terminal_component_is_committed_from_a_fresh_snapshot(self):
        explorer = PortalExplorer()
        explorer.active_frontier_component = None
        component = {
            "epoch": 13,
            "label": 7,
            "cells": 80,
            "center_x": 8.0,
            "center_y": 4.0,
            "min_x": 7.0,
            "max_x": 9.0,
            "min_y": 3.0,
            "max_y": 5.0,
        }
        explorer.component_at_map_position = lambda *_args: component

        self.assertIsNone(
            explorer.record_portal_place_arrival((10, 20, 8.0, 4.0), now=3.0)
        )
        self.assertEqual(explorer.events[-1][0], "portal_place_arrival_pending")
        self.assertEqual(len(explorer.pending_portal_arrivals), 1)

        entered = explorer.retry_pending_portal_arrivals(
            None, None, None, now=4.0
        )

        self.assertEqual(len(entered), 1)
        self.assertEqual(len(explorer.pending_portal_arrivals), 0)
        self.assertEqual(entered[0]["portal_arrivals"], 1)
        self.assertEqual(explorer.events[-1][0], "portal_place_entered")
        self.assertEqual(
            explorer.events[-1][1]["evidence_source"], "fresh_map_snapshot"
        )

    def test_source_component_does_not_reject_a_pending_destination_arrival(self):
        explorer = PortalExplorer()
        source, _tier = explorer.region_memory.enter(
            5.0,
            4.0,
            now=1.0,
            component=explorer.active_frontier_component,
        )
        explorer.region_memory.endpoint_observed(
            5.0,
            4.0,
            now=2.0,
            component=explorer.active_frontier_component,
            region_id=source["id"],
        )
        explorer.place_departure.region_id = source["id"]
        explorer.active_frontier_component = None
        explorer.component_at_map_position = lambda *_args: {
            **source["component"],
        }

        explorer.record_portal_place_arrival((10, 20, 8.0, 4.0), now=3.0)
        entered = explorer.retry_pending_portal_arrivals(
            None, None, None, now=4.0
        )

        self.assertEqual(entered, [])
        self.assertEqual(len(explorer.pending_portal_arrivals), 1)
        self.assertFalse(any(event == "portal_place_arrival_rejected"
                             for event, _fields in explorer.events))

    def test_verified_crossing_creates_destination_when_core_stays_connected(self):
        explorer = PortalExplorer()
        source, _tier = explorer.region_memory.enter(
            5.0,
            4.0,
            now=1.0,
            component=explorer.active_frontier_component,
        )
        explorer.region_memory.endpoint_observed(
            5.0,
            4.0,
            now=2.0,
            component=explorer.active_frontier_component,
            region_id=source["id"],
        )
        explorer.place_departure.region_id = source["id"]
        explorer.portal_hypothesis_ledger = PortalHypothesisLedger()
        edge, _created = explorer.portal_hypothesis_ledger.certify(
            source["id"],
            explorer.active_portal_gate_odom_xy,
            explorer.active_portal_destination_odom_xy,
            now=2.0,
        )
        explorer.last_portal_hypothesis_id = edge["id"]
        transaction = PortalTransaction()
        transaction.start(
            explorer.active_route_id,
            portal_id=edge["id"],
            source_place_id=source["id"],
            gate_xy=explorer.active_portal_gate_odom_xy,
            destination_xy=explorer.active_portal_destination_odom_xy,
            source_side_proven=True,
        )
        transaction.crossing_verified(explorer.active_route_id)
        explorer.portal_transaction = transaction

        entered = explorer.record_portal_place_arrival(
            (10, 20, 8.0, 4.0), now=3.0,
        )

        self.assertIsNotNone(entered)
        self.assertNotEqual(entered["id"], source["id"])
        self.assertEqual(entered["portal_arrivals"], 1)
        self.assertEqual(explorer.current_physical_place_id, entered["id"])
        self.assertEqual(edge["destination_place_id"], entered["id"])
        self.assertEqual(
            explorer.events[-1][0], "portal_place_entered",
        )

    def test_verified_crossing_materializes_destination_without_component(self):
        """A physical edge must not wait for an unavailable SLAM label."""
        explorer = PortalExplorer()
        source, _tier = explorer.region_memory.enter(
            5.0,
            4.0,
            now=1.0,
            component=explorer.active_frontier_component,
        )
        explorer.region_memory.endpoint_observed(
            5.0,
            4.0,
            now=2.0,
            component=explorer.active_frontier_component,
            region_id=source["id"],
        )
        explorer.place_departure.region_id = source["id"]
        explorer.active_frontier_component = None
        explorer.portal_hypothesis_ledger = PortalHypothesisLedger()
        edge, _created = explorer.portal_hypothesis_ledger.certify(
            source["id"],
            explorer.active_portal_gate_odom_xy,
            explorer.active_portal_destination_odom_xy,
            now=2.0,
        )
        explorer.last_portal_hypothesis_id = edge["id"]
        transaction = PortalTransaction()
        transaction.start(
            explorer.active_route_id,
            portal_id=edge["id"],
            source_place_id=source["id"],
            gate_xy=explorer.active_portal_gate_odom_xy,
            destination_xy=explorer.active_portal_destination_odom_xy,
            source_side_proven=True,
        )
        transaction.crossing_verified(explorer.active_route_id)
        explorer.portal_transaction = transaction

        entered = explorer.record_portal_place_arrival(
            (10, 20, 8.0, 4.0), now=3.0,
        )

        self.assertIsNotNone(entered)
        self.assertNotEqual(entered["id"], source["id"])
        self.assertEqual(explorer.current_physical_place_id, entered["id"])
        self.assertEqual(edge["destination_place_id"], entered["id"])
        self.assertEqual(
            explorer.events[-1][1]["evidence_source"],
            "portal_identity_without_component",
        )

    def test_verified_arrival_in_covered_place_is_recorded_without_reopening_it(self):
        explorer = PortalExplorer()
        destination = explorer.active_frontier_component
        region, _tier = explorer.region_memory.enter(
            8.0, 4.0, now=1.0, component=destination,
        )
        explorer.region_memory.observe(
            8.0,
            4.0,
            10.0,
            robot_x=8.0,
            robot_y=4.0,
            now=2.0,
            component=destination,
            region_id=region["id"],
            observation_ready=True,
        )
        explorer.region_memory.dormant(
            region, now=3.0, reason="physical_departure",
        )

        arrived = explorer.record_portal_place_arrival(
            (10, 20, 8.0, 4.0), now=4.0,
        )

        self.assertIs(arrived, region)
        self.assertEqual(region["state"], "dormant")
        self.assertEqual(region["covered_arrivals"], 1)
        self.assertEqual(region["last_reason"], "portal_arrival_already_covered")
        self.assertEqual(explorer.events[-1][0], "portal_place_covered_arrival")

    def test_reverse_portal_arrival_reuses_the_edge_source_place(self):
        explorer = PortalExplorer()
        source_component = {
            "epoch": 20,
            "label": 1,
            "cells": 80,
            "center_x": 5.0,
            "center_y": 4.0,
            "min_x": 4.0,
            "max_x": 6.0,
            "min_y": 3.0,
            "max_y": 5.0,
        }
        destination_component = {
            **source_component,
            "label": 2,
            "center_x": 8.0,
        }
        source, _ = explorer.region_memory.activate(
            5.0, 4.0, 10.0, now=1.0, component=source_component,
        )
        destination, _ = explorer.region_memory.activate(
            8.0, 4.0, 10.0, now=2.0, component=destination_component,
        )
        explorer.region_memory.endpoint_observed(
            5.0, 4.0, now=1.5, component=source_component,
            region_id=source["id"],
        )
        explorer.region_memory.endpoint_observed(
            8.0, 4.0, now=2.5, component=destination_component,
            region_id=destination["id"],
        )
        explorer.place_departure.region_id = destination["id"]
        explorer.active_frontier_component = source_component
        explorer.last_portal_hypothesis_id = 1
        explorer.portal_hypothesis_ledger = PortalHypothesisLedger()
        edge, _ = explorer.portal_hypothesis_ledger.certify(
            source["id"], (7.2, 4.0), (8.5, 4.0), now=2.0,
        )
        explorer.portal_hypothesis_ledger.crossed(edge["id"], now=2.5)
        explorer.portal_hypothesis_ledger.bind_destination(
            edge["id"], destination["id"], now=2.5,
        )

        arrived = explorer.record_portal_place_arrival(
            (10, 20, 8.0, 4.0), now=3.0,
        )

        self.assertIs(arrived, source)
        self.assertEqual(arrived["id"], source["id"])
        self.assertEqual(len(explorer.region_memory.regions), 2)
        # Reverse transit reuses the already-bound directed edge; it must not
        # overwrite the original destination with the Place being returned to.
        self.assertEqual(edge["destination_place_id"], destination["id"])
        self.assertFalse(any(
            event == "portal_hypothesis_destination_bound"
            for event, _fields in explorer.events
        ))

    def test_goal_circle_success_at_the_gate_never_enters_the_destination(self):
        explorer = PortalExplorer()
        # The action endpoint is beyond the door, but TEB may report success
        # anywhere in its XY tolerance circle. This pose is still at the gate.
        explorer.pose_odom = SimpleNamespace(x=7.3, y=4.0)

        entered = explorer.record_portal_place_arrival(
            (10, 20, 8.0, 4.0), now=3.0,
        )

        self.assertIsNone(entered)
        self.assertEqual(explorer.pending_portal_arrivals, [])
        self.assertIsNotNone(explorer.pending_portal_retry)
        self.assertEqual(explorer.events[-1][0], "portal_crossing_unconfirmed")
        self.assertEqual(
            explorer.events[-1][1]["reason"],
            "physical_destination_side_not_reached",
        )

    def test_stale_arrival_is_rejected_before_place_memory_mutation(self):
        explorer = PortalExplorer()
        transaction = PortalTransaction()
        transaction.start(
            20,
            portal_id=9,
            source_place_id=3,
            gate_xy=(7.2, 4.0),
            destination_xy=(8.5, 4.0),
        )
        transaction.crossing_verified(20)
        explorer.portal_transaction = transaction
        explorer.last_portal_hypothesis_id = 9

        arrival = PendingPortalArrival(
            route_id=19,
            goal_xy=(8.0, 4.0),
            terminal_time=4.0,
            source_region_id=3,
            portal_gate_xy=(7.2, 4.0),
            physical_goal_xy=(8.0, 4.0),
            physical_portal_gate_xy=(7.2, 4.0),
        )
        before = len(explorer.region_memory.regions)

        self.assertIsNone(
            explorer.commit_portal_arrival(
                arrival,
                explorer.active_frontier_component,
                now=4.0,
                evidence_source="stale_terminal",
            )
        )
        self.assertEqual(len(explorer.region_memory.regions), before)
        self.assertEqual(
            explorer.events[-1][1]["reason"],
            "stale_or_mismatched_portal_transaction",
        )

    def test_live_odom_crossing_commits_before_the_controller_terminal(self):
        """A completed graph edge cannot be retried after a late TEB stall."""
        explorer = PortalExplorer()

        region = explorer.observe_active_portal_crossing(now=3.0)

        self.assertIsNotNone(region)
        self.assertTrue(explorer.active_portal_crossing_observed)
        self.assertEqual(region["portal_arrivals"], 1)
        self.assertEqual(explorer.events[-1][0], "portal_place_entered")
        self.assertEqual(
            explorer.events[-1][1]["evidence_source"], "continuous_odom"
        )
        self.assertIsNone(explorer.observe_active_portal_crossing(now=4.0))
        self.assertEqual(region["portal_arrivals"], 1)

    def test_live_odom_check_does_not_prepare_a_retry_before_crossing(self):
        """Approaching a gate is not a failed edge and should remain quiet."""
        explorer = PortalExplorer()
        explorer.pose_odom = SimpleNamespace(x=7.3, y=4.0)

        self.assertIsNone(explorer.observe_active_portal_crossing(now=3.0))
        self.assertIsNone(explorer.pending_portal_retry)
        self.assertEqual(explorer.events, [])

    def test_self_loop_rejection_closes_transaction_and_departure(self):
        """A same-Place portal result must not strand successor planning."""
        explorer = PortalExplorer()
        source, _tier = explorer.region_memory.enter(
            5.0,
            4.0,
            now=1.0,
            component=explorer.active_frontier_component,
        )
        explorer.region_memory.endpoint_observed(
            5.0,
            4.0,
            now=2.0,
            component=explorer.active_frontier_component,
            region_id=source["id"],
        )
        class Departure:
            active = True
            region_id = source["id"]

            def clear(self):
                self.active = False

        explorer.place_departure = Departure()
        explorer._clear_active_place_departure = explorer.place_departure.clear
        explorer.portal_hypothesis_ledger = PortalHypothesisLedger()
        edge, _created = explorer.portal_hypothesis_ledger.certify(
            source["id"],
            explorer.active_portal_gate_odom_xy,
            explorer.active_portal_destination_odom_xy,
            now=2.0,
        )
        explorer.last_portal_hypothesis_id = edge["id"]
        transaction = PortalTransaction()
        transaction.start(
            explorer.active_route_id,
            portal_id=edge["id"],
            source_place_id=source["id"],
            gate_xy=explorer.active_portal_gate_odom_xy,
            destination_xy=explorer.active_portal_destination_odom_xy,
            source_side_proven=True,
        )
        transaction.crossing_verified(explorer.active_route_id)
        explorer.portal_transaction = transaction

        # Force the destination resolver to return the source Place, matching
        # the false self-loop seen in the Level 4 run when no new component was
        # available at the portal endpoint.
        explorer.region_memory.enter_new_portal_destination = (
            lambda *_args, **_kwargs: (source, "covered_arrival")
        )
        arrival = PendingPortalArrival(
            route_id=explorer.active_route_id,
            goal_xy=(8.0, 4.0),
            terminal_time=3.0,
            source_region_id=source["id"],
            portal_gate_xy=explorer.active_portal_gate_xy,
            physical_goal_xy=(8.0, 4.0),
            physical_portal_gate_xy=explorer.active_portal_gate_odom_xy,
        )

        self.assertIsNone(
            explorer.commit_portal_arrival(
                arrival,
                None,
                now=3.0,
                evidence_source="portal_identity_without_component",
            )
        )
        self.assertFalse(explorer.portal_transaction.active)
        self.assertTrue(explorer.active_portal_crossing_rejected)
        self.assertFalse(explorer.place_departure.active)
        self.assertEqual(
            explorer.portal_hypothesis_ledger.get(edge["id"])["state"],
            "failed",
        )
        self.assertTrue(any(
            event == "portal_transaction_aborted" for event, _fields in explorer.events
        ))
        # The rejected route cannot reopen the same arrival on a later pose or
        # terminal callback.
        event_count = len(explorer.events)
        self.assertIsNone(
            explorer.record_portal_place_arrival(
                (10, 20, 8.0, 4.0), now=4.0,
            )
        )
        self.assertEqual(len(explorer.events), event_count)

    def test_mirrored_crossing_reuses_the_room_instead_of_minting_a_third_place(
        self,
    ):
        """One doorway, two directions, one Place per room.

        A revisit from the opposite side certifies a mirrored identity at the
        same physical gate (test_opposite_direction_is_a_distinct_portal_
        identity in test_portal_belief), and that identity is born with no
        destination.  Committing its crossing through the ordinary path minted
        a *third* Place for a room the doorway already reached.
        """
        explorer = PortalExplorer()
        # The crossing that reaches a room the map has not separated yet
        # commits without a SLAM component (evidence source
        # portal_identity_without_component), which is the condition under
        # which the destination has to come from the edge identity.
        explorer.active_frontier_component = None
        source, _tier = explorer.region_memory.enter(
            5.0, 4.0, now=1.0, component=None,
        )
        explorer.region_memory.endpoint_observed(
            5.0,
            4.0,
            now=2.0,
            component=None,
            region_id=source["id"],
        )
        explorer.place_departure.region_id = source["id"]
        ledger = PortalHypothesisLedger()
        explorer.portal_hypothesis_ledger = ledger

        def cross(edge, terminal_x):
            explorer.active_portal_crossing_observed = False
            explorer.place_departure.region_id = source["id"]
            explorer.last_portal_hypothesis_id = edge["id"]
            # The base really passes through the doorway on this crossing:
            # east of the gate for the forward edge, west of it for the mirror.
            explorer.pose_odom = SimpleNamespace(x=float(terminal_x), y=4.0)
            transaction = PortalTransaction()
            transaction.start(
                explorer.active_route_id,
                portal_id=edge["id"],
                source_place_id=source["id"],
                gate_xy=tuple(edge["physical_gate"]),
                destination_xy=tuple(edge["physical_destination"]),
                source_side_proven=True,
            )
            transaction.crossing_verified(explorer.active_route_id)
            explorer.portal_transaction = transaction
            return explorer.record_portal_place_arrival(
                (10, 20, float(terminal_x), 4.0), now=4.0,
            )

        # First crossing: the machinery mints the room beyond the doorway.
        forward, _created = ledger.certify(
            source["id"],
            explorer.active_portal_gate_odom_xy,
            explorer.active_portal_destination_odom_xy,
            now=2.0,
        )
        ledger.crossed(forward["id"], now=2.5)
        destination = cross(forward, 8.0)
        self.assertIsNotNone(destination)
        self.assertNotEqual(destination["id"], source["id"])
        self.assertEqual(forward["destination_place_id"], destination["id"])
        places_before = len(explorer.region_memory.regions)

        # Second crossing, same doorway, opposite direction.
        mirrored, created = ledger.certify(
            source["id"], (7.0, 4.0), (6.3, 4.0), now=3.0,
        )
        self.assertTrue(created)
        self.assertIsNone(mirrored["destination_place_id"])
        entered = cross(mirrored, 6.5)

        self.assertIsNotNone(entered)
        self.assertEqual(entered["id"], destination["id"])
        self.assertEqual(len(explorer.region_memory.regions), places_before)
        # The mirrored identity adopts the doorway's established endpoint, so
        # later traversals through it reuse the same room.
        self.assertEqual(mirrored["destination_place_id"], destination["id"])

    def test_unrelated_crossing_still_creates_a_destination_place(self):
        """The gate fallback must not turn an unseen doorway into a reuse."""
        explorer = PortalExplorer()
        source, _tier = explorer.region_memory.enter(
            5.0, 4.0, now=1.0, component=explorer.active_frontier_component,
        )
        explorer.region_memory.endpoint_observed(
            5.0,
            4.0,
            now=2.0,
            component=explorer.active_frontier_component,
            region_id=source["id"],
        )
        explorer.place_departure.region_id = source["id"]
        ledger = PortalHypothesisLedger()
        explorer.portal_hypothesis_ledger = ledger
        far, _created = ledger.certify(
            source["id"], (1.0, 1.0), (1.0, 1.8), now=2.0,
        )
        ledger.crossed(far["id"], now=2.0)
        ledger.bind_destination(far["id"], 9, now=2.0)
        edge, _created = ledger.certify(
            source["id"],
            explorer.active_portal_gate_odom_xy,
            explorer.active_portal_destination_odom_xy,
            now=3.0,
        )
        explorer.last_portal_hypothesis_id = edge["id"]
        transaction = PortalTransaction()
        transaction.start(
            explorer.active_route_id,
            portal_id=edge["id"],
            source_place_id=source["id"],
            gate_xy=explorer.active_portal_gate_odom_xy,
            destination_xy=explorer.active_portal_destination_odom_xy,
            source_side_proven=True,
        )
        transaction.crossing_verified(explorer.active_route_id)
        explorer.portal_transaction = transaction

        entered = explorer.record_portal_place_arrival(
            (10, 20, 8.0, 4.0), now=4.0,
        )

        self.assertIsNotNone(entered)
        self.assertNotEqual(entered["id"], source["id"])
        self.assertEqual(len(explorer.region_memory.regions), 2)


if __name__ == "__main__":
    unittest.main()
