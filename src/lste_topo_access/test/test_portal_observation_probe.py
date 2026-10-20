"""Pure contracts for source-side doorway observation probes."""

from pathlib import Path
import sys
import unittest

import numpy as np
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from global_frontier_portal_probes import portal_observation_probe
from global_frontier_candidate_portals import GlobalFrontierCandidatePortalMixin


class ProbeHarness(GlobalFrontierCandidatePortalMixin):
    scan_observation_horizon = 4.0
    region_topology_clearance = 0.77
    place_furniture_max_span_m = 2.5

    @staticmethod
    def request(unknown, structural_occupied):
        return SimpleNamespace(
            unknown=unknown,
            components=SimpleNamespace(structural_occupied=structural_occupied),
            message=SimpleNamespace(
                info=SimpleNamespace(resolution=0.1)
            ),
        )


class PortalObservationProbeTest(unittest.TestCase):
    def doorway_walls(self):
        walls = np.zeros((9, 9), dtype=bool)
        # A vertical passage through a horizontal wall: the source is left of
        # the opening, so the probe normal points right. Wall runs extend away
        # from the opening in both tangent directions.
        walls[1:4, 4] = True
        walls[5:8, 4] = True
        return walls

    def test_unknown_wall_bounded_opening_creates_source_side_probe(self):
        unknown = np.zeros((9, 9), dtype=bool)
        unknown[4, 5] = True

        probe = portal_observation_probe(
            unknown,
            self.doorway_walls(),
            (4, 4),
            support_radius=1,
            minimum_wall_span_cells=3,
        )

        self.assertIsNotNone(probe)
        self.assertEqual(probe.opening_cell, (4, 4))
        self.assertEqual(probe.normal, (0, 1))

    def test_unknown_without_opposing_structural_walls_is_not_a_probe(self):
        unknown = np.zeros((9, 9), dtype=bool)
        unknown[4, 5] = True

        self.assertIsNone(portal_observation_probe(
            unknown,
            np.zeros_like(unknown),
            (4, 4),
            support_radius=1,
            minimum_wall_span_cells=3,
        ))

    def test_known_far_side_cannot_become_a_probe(self):
        unknown = np.zeros((9, 9), dtype=bool)

        self.assertIsNone(portal_observation_probe(
            unknown,
            self.doorway_walls(),
            (4, 4),
            support_radius=1,
            minimum_wall_span_cells=3,
        ))

    def test_partially_observed_wall_uses_scan_horizon_for_probe_evidence(self):
        walls = np.zeros((25, 15), dtype=bool)
        walls[2:8, 7] = True
        walls[17:23, 7] = True
        unknown = np.zeros_like(walls)
        # The far-side cell of the doorway is still unknown: a partially
        # observed wall. The gap itself spans nine cells (0.9 m), wide enough
        # for the clearance disk the router drives, so it may become a probe.
        unknown[8, 8] = True
        probe = ProbeHarness().portal_observation_probe_for_frontier(
            ProbeHarness.request(unknown, walls), 8, 7,
        )
        self.assertIsNotNone(probe)
        self.assertEqual(probe.opening_cell, (8, 7))

    def test_probe_adapter_rejects_gap_narrower_than_the_vehicle(self):
        """The same adapter must not probe a slit the body cannot pass."""
        walls = np.zeros((25, 15), dtype=bool)
        walls[2:8, 7] = True
        walls[17:23, 7] = True
        # Three free rows in the wall: a rasterisation slit, not a doorway.
        walls[11:17, 7] = True
        unknown = np.zeros_like(walls)
        unknown[8, 8] = True

        # The width bound is what rejects it: without the footprint check the
        # same slit yields a probe.
        self.assertIsNotNone(
            portal_observation_probe(
                unknown, walls, (8, 7),
                support_radius=40, minimum_wall_span_cells=2,
            )
        )
        self.assertIsNone(
            ProbeHarness().portal_observation_probe_for_frontier(
                ProbeHarness.request(unknown, walls), 8, 7,
            )
        )


if __name__ == "__main__":
    unittest.main()
