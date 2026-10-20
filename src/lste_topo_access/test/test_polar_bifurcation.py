"""Unit and regression tests for polar continuous field multimodal bifurcation detector."""

from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from global_frontier_polar_bifurcation import (
    analyze_polar_bifurcation,
    sample_polar_clearance,
    smooth_polar_field,
    snap_to_cardinal_if_close,
)


class PolarBifurcationTest(unittest.TestCase):
    def test_snap_to_cardinal(self):
        # Near (0, 1) within 15 deg
        vec = (0.05, 0.99)
        unit, grid = snap_to_cardinal_if_close(vec, tolerance_deg=15.0)
        self.assertEqual(unit, (0.0, 1.0))
        self.assertEqual(grid, (0, 1))

        # Near (1, 0) within 15 deg
        vec = (0.98, -0.06)
        unit, grid = snap_to_cardinal_if_close(vec, tolerance_deg=15.0)
        self.assertEqual(unit, (1.0, 0.0))
        self.assertEqual(grid, (1, 0))

        # Truly 45 deg diagonal (should NOT snap to cardinal)
        vec = (0.7071, 0.7071)
        unit, grid = snap_to_cardinal_if_close(vec, tolerance_deg=15.0)
        self.assertAlmostEqual(unit[0], 0.7071, places=3)
        self.assertAlmostEqual(unit[1], 0.7071, places=3)
        self.assertEqual(grid, (1, 1))

    def test_standard_vertical_wall_doorway(self):
        """0.9 m doorway in a vertical wall (wall along rows, passage along cols)."""
        grid = np.zeros((50, 50), dtype=bool)
        grid[10:40, 25] = True
        grid[21:30, 25] = False  # 9 cells = 0.9m at 0.1m/cell

        result = analyze_polar_bifurcation(
            grid,
            (25, 25),
            resolution=0.1,
            max_opening_m=1.8,
            min_opening_m=0.6,
        )

        self.assertTrue(result.is_bottleneck)
        self.assertAlmostEqual(result.opening_width, 0.9, delta=0.15)
        # Normal must be strictly horizontal (col axis: 0, 1 or 0, -1)
        self.assertEqual(result.normal_grid[0], 0)
        self.assertIn(result.normal_grid[1], (1, -1))
        self.assertEqual(result.normal_vector[0], 0.0)
        self.assertTrue(result.constriction_ratio >= 1.2)

    def test_standard_horizontal_wall_doorway(self):
        """1.0 m doorway in a horizontal wall (wall along cols, passage along rows)."""
        grid = np.zeros((50, 50), dtype=bool)
        grid[25, 10:40] = True
        grid[25, 20:30] = False  # 10 cells = 1.0m at 0.1m/cell

        result = analyze_polar_bifurcation(
            grid,
            (25, 25),
            resolution=0.1,
            max_opening_m=1.8,
            min_opening_m=0.6,
        )

        self.assertTrue(result.is_bottleneck)
        self.assertAlmostEqual(result.opening_width, 1.0, delta=0.15)
        # Normal must be strictly vertical (row axis: 1, 0 or -1, 0)
        self.assertIn(result.normal_grid[0], (1, -1))
        self.assertEqual(result.normal_grid[1], 0)
        self.assertEqual(result.normal_vector[1], 0.0)

    def test_tilted_45deg_doorway_rotation_invariance(self):
        """1.0 m doorway on a wall tilted at 45 degrees."""
        grid = np.zeros((51, 51), dtype=bool)
        # Wall equation: (c - 25) + (r - 25) = 0
        # Tangent: (-0.707, 0.707), Normal: (0.707, 0.707)
        for r in range(51):
            for c in range(51):
                dn = ((c - 25) + (r - 25)) * np.sqrt(0.5) * 0.1
                dt = (-(c - 25) + (r - 25)) * np.sqrt(0.5) * 0.1
                if abs(dn) < 0.08 and abs(dt) >= 0.5:
                    grid[r, c] = True

        result = analyze_polar_bifurcation(
            grid,
            (25, 25),
            resolution=0.1,
            max_opening_m=1.8,
            min_opening_m=0.6,
        )

        self.assertTrue(result.is_bottleneck)
        self.assertAlmostEqual(result.opening_width, 1.0, delta=0.15)
        # Normal must be aligned with 45 deg or 225 deg (not cardinal)
        dot_45 = abs(result.normal_vector[0] * np.sqrt(0.5) + result.normal_vector[1] * np.sqrt(0.5))
        self.assertGreater(dot_45, 0.90)

    def test_env04_extra_wide_3_2m_partition_rejected(self):
        """3.2 m open partition between rooms (env04 root cause) must be rejected."""
        grid = np.zeros((60, 120), dtype=bool)
        grid[30, 5:51] = True
        grid[30, 83:115] = True  # Opening from col 51 to 83 = 32 cells = 3.2m
        center = (30, 67)

        result = analyze_polar_bifurcation(
            grid,
            center,
            resolution=0.1,
            max_opening_m=1.8,
            min_opening_m=0.6,
        )

        self.assertFalse(result.is_bottleneck)
        self.assertIn("opening_too_wide", result.reason)
        self.assertGreater(result.opening_width, 3.0)

    def test_narrow_slit_rejected(self):
        """0.3 m slit narrower than vehicle footprint must be rejected."""
        grid = np.zeros((50, 50), dtype=bool)
        grid[10:40, 25] = True
        grid[24:27, 25] = False  # 3 cells = 0.3m

        result = analyze_polar_bifurcation(
            grid,
            (25, 25),
            resolution=0.1,
            max_opening_m=1.8,
            min_opening_m=0.6,
        )

        self.assertFalse(result.is_bottleneck)
        self.assertIn("opening_too_narrow", result.reason)

    def test_open_space_rejected(self):
        """Center of open room without opposing jambs must be rejected."""
        grid = np.zeros((50, 50), dtype=bool)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True

        result = analyze_polar_bifurcation(
            grid,
            (25, 25),
            resolution=0.1,
            max_opening_m=1.8,
            min_opening_m=0.6,
        )

        self.assertFalse(result.is_bottleneck)

    def test_uniform_corridor_rejected(self):
        """Point inside a uniform corridor has no outward wall runs at jambs."""
        grid = np.zeros((50, 50), dtype=bool)
        # Corridor runs along cols (rows 20 to 30 is corridor, width 1.0m)
        grid[19, :] = True
        grid[31, :] = True

        result = analyze_polar_bifurcation(
            grid,
            (25, 25),
            resolution=0.1,
            max_opening_m=1.8,
            min_opening_m=0.6,
            min_wall_span_cells=2,
        )

        self.assertFalse(result.is_bottleneck)

    def test_env04_3_2m_doorway_accepted_with_3_5m_limit(self):
        """3.2 m physical doorway in env04 near (6.5, 2.2) is accepted and points vertical (+Y)."""
        grid = np.zeros((80, 140), dtype=bool)
        # Horizontal walls along cols at row 40
        grid[40, 10:51] = True
        grid[40, 83:130] = True  # Opening from col 51 to 83 = 32 cells = 3.2m at 0.1m/cell
        center = (40, 67)

        result = analyze_polar_bifurcation(
            grid,
            center,
            resolution=0.1,
            max_opening_m=3.5,
            min_opening_m=0.6,
        )

        self.assertTrue(result.is_bottleneck)
        self.assertAlmostEqual(result.opening_width, 3.2, delta=0.25)
        # Normal must point vertically (row axis: +1 or -1, col axis: 0)
        self.assertIn(result.normal_grid[0], (1, -1))
        self.assertEqual(result.normal_grid[1], 0)
        self.assertEqual(result.normal_vector[1], 0.0)


if __name__ == "__main__":
    unittest.main()
