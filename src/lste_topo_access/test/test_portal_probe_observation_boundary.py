"""Regression tests for the Portal-probe observation transaction boundary."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

sys.modules.setdefault(
    "rospy",
    SimpleNamespace(
        loginfo=lambda *args, **kwargs: None,
        logwarn=lambda *args, **kwargs: None,
    ),
)

from global_frontier_observation_coverage import (  # noqa: E402
    GlobalFrontierObservationCoverageMixin,
)
from global_frontier_candidate_lifecycle import (  # noqa: E402
    GlobalFrontierCandidateLifecycleMixin,
)
from global_frontier_work_items import (  # noqa: E402
    GlobalFrontierWorkItemLifecycleMixin,
)


# The coverage gate is part of the settle contract; bind the real methods so
# these tests exercise the production evidence rule rather than a stub.
COVERAGE_GATE = GlobalFrontierWorkItemLifecycleMixin
COVERAGE_METHODS = (
    "active_work_item_support_is_observed",
    "active_work_item_unknown_support_checker",
    "_active_support_quantum",
)


class CoverageStub(GlobalFrontierObservationCoverageMixin):
    # Bind the production settle gate; with no active WorkItem it fails open
    # (observed), preserving the historical resolve behavior of these tests.
    for _method in COVERAGE_METHODS:
        locals()[_method] = getattr(COVERAGE_GATE, _method)
    del _method

    def __init__(self, active_probe, active_phase="source"):
        self.target_region_claim_active = False
        self.pose_odom = None
        self.active_portal_probe_id = active_probe
        self.portal_work_phase = active_phase
        self.portal_probe_ledger = SimpleNamespace(
            get=lambda _probe_id: (
                None
                if active_probe is None
                else {"active_phase": self.portal_work_phase}
            )
        )
        self.completed_frontiers = []
        self.events = []
        self.region_memory = SimpleNamespace(
            endpoint_observed=lambda *args, **kwargs: {
                "id": 4,
                "visits": 1,
                "endpoint_observations": 1,
                "state": "open",
                "last_reason": "endpoint_observed",
            },
        )

    def settle_active_portal_probe(self, result, now, reason):
        self.events.append(("probe", result, now, reason))
        self.active_portal_probe_id = None

    def settle_active_work_item(self, now, state, reason):
        self.events.append(("work", now, state, reason))

    def frontier_is_completed(self, _x, _y):
        return False

    def publish_status(self, event, **fields):
        self.events.append((event, fields))


class CandidateLifecycleStub(GlobalFrontierCandidateLifecycleMixin):
    """Expose the generic-route boundary for a probe-owned WorkItem."""

    def __init__(self, probe):
        self.portal_probe_ledger = SimpleNamespace(
            probe_for_work_item=lambda item_id: (
                probe if int(item_id) == 21 else None
            )
        )

    @staticmethod
    def portal_observation_probe_for_frontier(*_args):
        return None


class TargetObservationLifecycleStub(CandidateLifecycleStub):
    """Expose a probe candidate that target ownership must suppress locally."""

    @staticmethod
    def portal_observation_probe_for_frontier(*_args):
        return {"id": 31, "state": "pending"}


def _coverage_explorer(support_cell, unknown_value, wall_cell=None):
    """Build a coverage stub with one active item on a small known map.

    The map is 6x6 at 0.25 m from origin (0,0); the support lattice quantum
    matches the resolution, so lattice cell (4,0) projects to map cell
    (row=0, col=4) at physical (1.0, 0.0).  The pose sits 0.5 m away, well
    inside a 10 m scan horizon, so the gate evaluates that one cell.
    ``wall_cell`` marks one occupied map cell, which occludes whatever lies
    behind it from the pose.
    """
    explorer = CoverageStub(active_probe=None)
    data = [0] * 36
    data[0 * 6 + 4] = unknown_value
    if wall_cell is not None:
        data[wall_cell[0] * 6 + wall_cell[1]] = 100
    explorer.place_work_items = SimpleNamespace(
        support_for=lambda _item_id: frozenset({(4, 0)}),
    )
    explorer.active_work_item_id = 7
    explorer.scan_range_max = 10.0
    explorer.pose_odom = SimpleNamespace(x=1.0, y=0.5)
    explorer.map_msg = SimpleNamespace(
        header=SimpleNamespace(frame_id="map"),
        info=SimpleNamespace(
            resolution=0.25,
            width=6,
            height=6,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0),
            ),
        ),
        data=data,
    )
    # Identity projector: the physical (odom) frame coincides with the map.
    explorer.planar_xy_projector = lambda _target, _source: (lambda x, y: (x, y))
    return explorer


class PortalProbeObservationBoundaryTest(unittest.TestCase):
    def test_source_arrival_does_not_resolve_observation_work(self):
        explorer = CoverageStub(active_probe=9)

        explorer.mark_frontier_observed(1.0, 2.0, now=3.0)

        self.assertEqual(explorer.events[0][:2], ("probe", "source_arrived"))
        self.assertFalse(any(event[0] == "work" for event in explorer.events))

    def test_ordinary_endpoint_still_resolves_observation_work(self):
        explorer = CoverageStub(active_probe=None)

        explorer.mark_frontier_observed(1.0, 2.0, now=3.0)

        self.assertTrue(any(event[0] == "work" for event in explorer.events))

    def test_endpoint_with_unseen_support_defers_instead_of_resolving(self):
        # The dispatch goal was reached, but the item's support cell is still
        # unknown on the map within scan range: arrival is not coverage.
        explorer = _coverage_explorer((4, 0), unknown_value=-1)

        explorer.mark_frontier_observed(1.0, 2.0, now=3.0)

        work_events = [e for e in explorer.events if e[0] == "work"]
        self.assertEqual(len(work_events), 1)
        self.assertEqual(work_events[0][2], "deferred")
        self.assertEqual(work_events[0][3], "endpoint_reached_support_unobserved")

    def test_endpoint_with_observed_support_resolves_observation_work(self):
        # Same geometry, but the support cell is known free on the map: the
        # coverage gate passes and the item resolves with its evidence.
        explorer = _coverage_explorer((4, 0), unknown_value=0)

        explorer.mark_frontier_observed(1.0, 2.0, now=3.0)

        work_events = [e for e in explorer.events if e[0] == "work"]
        self.assertEqual(len(work_events), 1)
        self.assertEqual(work_events[0][2], "resolved")
        self.assertEqual(work_events[0][3], "endpoint_coverage_recorded")

    def test_support_cell_hidden_behind_a_wall_does_not_block_observation(self):
        # The live two-room failure: the support patch is the whole unknown
        # flood out to the sensor horizon, and part of it lies behind a wall
        # no standoff can see past.  Charging that occluded cell to this
        # observation would defer every ordinary boundary forever.
        explorer = _coverage_explorer(
            (4, 0), unknown_value=-1, wall_cell=(1, 4),
        )

        explorer.mark_frontier_observed(1.0, 2.0, now=3.0)

        work_events = [e for e in explorer.events if e[0] == "work"]
        self.assertEqual(len(work_events), 1)
        self.assertEqual(work_events[0][2], "resolved")

    def test_support_cell_beyond_the_sensor_horizon_does_not_block(self):
        # A patch cell farther than the sensor can reach cannot be observed
        # from this pose either, so it is not this observation's evidence.
        explorer = _coverage_explorer((4, 0), unknown_value=-1)
        explorer.scan_range_max = 0.4
        explorer.pose_odom = SimpleNamespace(x=1.0, y=0.5)

        explorer.mark_frontier_observed(1.0, 2.0, now=3.0)

        work_events = [e for e in explorer.events if e[0] == "work"]
        self.assertEqual(len(work_events), 1)
        self.assertEqual(work_events[0][2], "resolved")

    def test_destination_arrival_uses_observed_terminal(self):
        explorer = CoverageStub(active_probe=9, active_phase="destination")

        explorer.mark_frontier_observed(1.0, 2.0, now=3.0)

        self.assertEqual(explorer.events[0][:2], ("probe", "observed"))
        self.assertEqual(explorer.events[0][3], "destination_viewpoint_observed")
        self.assertFalse(any(event[0] == "work" for event in explorer.events))

    def test_probe_owned_work_item_cannot_become_generic_local_route(self):
        explorer = CandidateLifecycleStub({"id": 9, "state": "source_arrived"})
        context = SimpleNamespace(
            source_place_id=7,
            source_place_observed=True,
            work_item_cells={(3, 4): 21},
            work_item_details={21: {"match": "inherited"}},
        )

        result = explorer.local_observation_work_item(
            SimpleNamespace(), context, 3, 4, 0,
        )

        self.assertEqual(result, (None, None, 0, None))

    def test_target_observation_keeps_frontier_as_local_viewpoint(self):
        explorer = TargetObservationLifecycleStub(None)
        context = SimpleNamespace(
            source_place_id=7,
            source_place_observed=True,
            target_observation_work_pending=True,
            work_item_cells={(3, 4): 21},
            work_item_details={21: {"match": "inherited"}},
        )

        result = explorer.local_observation_work_item(
            SimpleNamespace(), context, 3, 4, 0,
        )

        self.assertEqual(result, (21, "inherited", 0, None))


if __name__ == "__main__":
    unittest.main()
