"""Lightweight polar Gaussian continuous field and multimodal bifurcation detector.

Inspired by GP Frontier (Gaussian continuous field modeling, polar coordinate
angular frontier multimodal clustering to detect bifurcation points):
This module provides rotation-invariant continuous polar clearance analysis:
1. Samples 360° continuous polar clearance R(theta) around a candidate opening.
2. Applies 1D circular Gaussian smoothing to filter discrete rasterization noise.
3. Detects multimodal passage peaks (openings) and opposing jamb valleys (saddles).
4. Enforces physical doorway scale constraints (e.g. W_min <= W <= W_max, where
   W_max is typically ~1.8m, rejecting wide open room partitions like 3.2m).
5. Enforces constriction / narrowing ratio between passage clearance and throat width.
6. Computes strictly orthogonal passage normal vectors (n perp t_door) with
   cardinal snapping for rectilinear architectural walls (snaps to +/-X or +/-Y
   when within tolerance, e.g. 15 deg).
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_diff(a: float, b: float) -> float:
    """Signed angular difference a - b wrapped to [-pi, pi]."""
    return wrap_angle(a - b)


def snap_to_cardinal_if_close(
    vector_row_col: Tuple[float, float],
    tolerance_deg: float = 15.0,
) -> Tuple[Tuple[float, float], Tuple[int, int]]:
    """Snap a 2D direction vector to cardinal (axis-aligned) if within tolerance.

    Parameters
    ----------
    vector_row_col: (dr, dc) direction vector in grid coordinates.
    tolerance_deg: maximum angle deviation to snap to cardinal axes.

    Returns
    -------
    ((norm_dr, norm_dc), (int_dr, int_dc))
    Exact float unit vector and snapped integer direction.
    """
    dr, dc = float(vector_row_col[0]), float(vector_row_col[1])
    length = math.hypot(dr, dc)
    if length <= 1e-9:
        return (0.0, 0.0), (0, 0)
    dr, dc = dr / length, dc / length

    cardinals = (
        (1.0, 0.0, 1, 0),
        (-1.0, 0.0, -1, 0),
        (0.0, 1.0, 0, 1),
        (0.0, -1.0, 0, -1),
    )
    cos_tol = math.cos(math.radians(max(0.0, float(tolerance_deg))))
    for cr, cc, ir, ic in cardinals:
        dot = dr * cr + dc * cc
        if dot >= cos_tol:
            return (cr, cc), (ir, ic)

    # Diagonal / angled door: return normalized float and nearest rounded grid step
    int_dr = int(round(dr))
    int_dc = int(round(dc))
    if int_dr == 0 and int_dc == 0:
        if abs(dr) >= abs(dc):
            int_dr = 1 if dr > 0 else -1
        else:
            int_dc = 1 if dc > 0 else -1
    return (dr, dc), (int_dr, int_dc)


@dataclass(frozen=True)
class PolarBifurcationResult:
    """Topological bifurcation / bottleneck verdict from polar continuous field."""

    is_bottleneck: bool
    opening_width: float = 0.0
    normal_rad: float = 0.0
    normal_vector: Tuple[float, float] = (0.0, 0.0)
    normal_grid: Tuple[int, int] = (0, 0)
    tangent_rad: float = 0.0
    tangent_vector: Tuple[float, float] = (0.0, 0.0)
    constriction_ratio: float = 0.0
    passage_clearance: float = 0.0
    jamb_clearance: float = 0.0
    jamb_contacts: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0))
    midpoint_cell: Tuple[float, float] = (0.0, 0.0)
    peaks_deg: Tuple[float, ...] = ()
    valleys_deg: Tuple[float, ...] = ()
    reason: str = "uninitialized"


def sample_polar_clearance(
    structural_occupied: np.ndarray,
    center_cell: Tuple[int, int],
    resolution: float = 0.1,
    num_angles: int = 72,
    max_range_m: float = 4.0,
    step_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample continuous polar clearance R(theta) from center_cell.

    Returns
    -------
    thetas: 1D array of angles in radians [0, 2*pi).
    R_raw: 1D array of clearance distances in meters.
    contacts: 2D array of contact (row, col) coordinates shape (num_angles, 2).
    """
    rows_total, cols_total = structural_occupied.shape
    center_r, center_c = int(center_cell[0]), int(center_cell[1])
    res = max(1e-6, float(resolution))
    num_angles = max(16, int(num_angles))
    max_range_m = max(0.5, float(max_range_m))
    step = 0.5 * res if step_m is None else max(1e-3, float(step_m))

    thetas = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
    steps = np.arange(step, max_range_m + 0.5 * step, step)
    if steps.size == 0:
        steps = np.array([step])

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # cols = x, rows = y
    cols = center_c + np.outer(cos_t, steps) / res
    rows = center_r + np.outer(sin_t, steps) / res
    cols_int = np.round(cols).astype(np.int32)
    rows_int = np.round(rows).astype(np.int32)

    in_bounds = (
        (rows_int >= 0) & (rows_int < rows_total) &
        (cols_int >= 0) & (cols_int < cols_total)
    )

    R_raw = np.full(num_angles, max_range_m, dtype=np.float64)
    contacts = np.zeros((num_angles, 2), dtype=np.int32)

    for k in range(num_angles):
        hit = False
        for m in range(len(steps)):
            if not in_bounds[k, m]:
                R_raw[k] = steps[m]
                contacts[k] = (
                    max(0, min(rows_total - 1, rows_int[k, m])),
                    max(0, min(cols_total - 1, cols_int[k, m])),
                )
                hit = True
                break
            r = rows_int[k, m]
            c = cols_int[k, m]
            if structural_occupied[r, c]:
                R_raw[k] = steps[m]
                contacts[k] = (r, c)
                hit = True
                break
        if not hit:
            # Reached max range
            contacts[k] = (
                max(0, min(rows_total - 1, rows_int[k, -1])),
                max(0, min(cols_total - 1, cols_int[k, -1])),
            )

    return thetas, R_raw, contacts


def smooth_polar_field(
    R: np.ndarray,
    sigma_deg: float = 10.0,
    num_angles: int = 72,
) -> np.ndarray:
    """Apply circular 1D Gaussian kernel smoothing to the polar field R."""
    sigma_bins = max(0.8, (float(sigma_deg) / 360.0) * float(num_angles))
    if _HAS_SCIPY:
        return gaussian_filter1d(R, sigma=sigma_bins, mode="wrap")

    # Pure numpy circular Gaussian convolution fallback
    radius = int(math.ceil(3.0 * sigma_bins))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([R[-radius:], R, R[:radius]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[radius:-radius]


def find_polar_extrema(
    R_smooth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find indices of local minima (valleys) and maxima (peaks) on 1D circle."""
    K = len(R_smooth)
    if K < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    prev_vals = np.roll(R_smooth, 1)
    next_vals = np.roll(R_smooth, -1)

    valleys = []
    peaks = []
    for i in range(K):
        v = R_smooth[i]
        pv = prev_vals[i]
        nv = next_vals[i]
        if v <= pv and v <= nv and (v < pv or v < nv):
            valleys.append(i)
        elif v >= pv and v >= nv and (v > pv or v > nv):
            peaks.append(i)

    return np.array(valleys, dtype=int), np.array(peaks, dtype=int)


def _check_outward_wall_run(
    structural_occupied: np.ndarray,
    contact_cell: Tuple[int, int],
    direction_step: Tuple[float, float],
    span_cells: int,
) -> bool:
    """Verify that a wall continues away from the jamb."""
    if span_cells <= 1:
        return True
    rows, cols = structural_occupied.shape
    cr, cc = float(contact_cell[0]), float(contact_cell[1])
    dr, dc = float(direction_step[0]), float(direction_step[1])
    step_len = math.hypot(dr, dc)
    if step_len <= 1e-6:
        return False
    dr, dc = dr / step_len, dc / step_len

    count = 0
    for s in range(1, int(span_cells) + 1):
        r = int(round(cr + s * dr))
        c = int(round(cc + s * dc))
        if 0 <= r < rows and 0 <= c < cols and structural_occupied[r, c]:
            count += 1
        else:
            # Allow 1 cell raster gap / noise
            if s > 1 and count >= int(span_cells) - 1:
                break
    return count >= max(1, int(span_cells) - 1)


def analyze_polar_bifurcation(
    structural_occupied: np.ndarray,
    center_cell: Tuple[int, int],
    resolution: float = 0.1,
    num_angles: int = 72,
    max_range_m: float = 4.5,
    min_opening_m: float = 0.60,
    max_opening_m: float = 3.50,
    min_constriction: float = 1.20,
    min_wall_span_cells: int = 2,
    cardinal_snap_deg: float = 15.0,
    known_free: Optional[np.ndarray] = None,
) -> PolarBifurcationResult:
    """Analyze polar clearance continuous field around candidate doorway center.

    Identifies topological bottlenecks (doorways) vs wide open partitions:
    - 360° raymarching & Gaussian kernel smoothing.
    - Identification of opposing jamb valleys & passage peaks.
    - Strict physical scale constraint W_min <= W <= W_max (default 0.60m ~ 1.80m).
    - Geometric constriction ratio test.
    - Strict orthogonal passage normal (n perp t_door) with cardinal snapping.
    """
    if not isinstance(structural_occupied, np.ndarray) or structural_occupied.ndim != 2:
        return PolarBifurcationResult(False, reason="invalid_occupancy_array")

    rows, cols = structural_occupied.shape
    r0, c0 = int(center_cell[0]), int(center_cell[1])
    if not (0 <= r0 < rows and 0 <= c0 < cols):
        return PolarBifurcationResult(False, reason="out_of_bounds")
    if structural_occupied[r0, c0]:
        return PolarBifurcationResult(False, reason="occupied_center")

    thetas, R_raw, contacts = sample_polar_clearance(
        structural_occupied,
        (r0, c0),
        resolution=resolution,
        num_angles=num_angles,
        max_range_m=max_range_m,
    )
    R_smooth = smooth_polar_field(R_raw, sigma_deg=10.0, num_angles=num_angles)

    valleys, peaks = find_polar_extrema(R_smooth)
    if len(valleys) < 2:
        return PolarBifurcationResult(False, reason="fewer_than_two_valleys")

    # Find the best opposing valley pair (jambs)
    # Opposing: separation |Delta theta - pi| <= 55 deg (between 125 deg and 235 deg)
    opposing_tol = math.radians(55.0)
    best_valley_pair = None
    best_opening_width = None

    for i in range(len(valleys)):
        v1 = valleys[i]
        for j in range(i + 1, len(valleys)):
            v2 = valleys[j]
            sep = abs(angle_diff(thetas[v1], thetas[v2]))
            if abs(sep - math.pi) <= opposing_tol:
                # Euclidean distance between jamb contact points
                p1 = contacts[v1]
                p2 = contacts[v2]
                dist_m = math.hypot(
                    (p2[0] - p1[0]) * resolution,
                    (p2[1] - p1[1]) * resolution,
                )
                # Fallback to ray distance sum if contact distance is zero
                if dist_m < 1e-3:
                    dist_m = float(R_raw[v1] + R_raw[v2])
                if best_opening_width is None or dist_m < best_opening_width:
                    best_opening_width = dist_m
                    best_valley_pair = (v1, v2)

    if best_valley_pair is None:
        return PolarBifurcationResult(False, reason="no_opposing_jamb_valleys")

    v1, v2 = best_valley_pair
    width_m = float(best_opening_width)

    # 1. Scale constraint: Reject openings narrower than robot or wider than max doorway
    if width_m < float(min_opening_m):
        return PolarBifurcationResult(
            False,
            opening_width=width_m,
            reason=f"opening_too_narrow ({width_m:.2f}m < {min_opening_m:.2f}m)",
        )
    if width_m > float(max_opening_m):
        return PolarBifurcationResult(
            False,
            opening_width=width_m,
            reason=f"opening_too_wide ({width_m:.2f}m > {max_opening_m:.2f}m)",
        )

    # 2. Passage clearance along orthogonal normal directions:
    # In row, col coords: normal perp tangent is (-t_col, t_row)
    contact1 = (int(contacts[v1][0]), int(contacts[v1][1]))
    contact2 = (int(contacts[v2][0]), int(contacts[v2][1]))
    tangent_row = float(contact2[0] - contact1[0])
    tangent_col = float(contact2[1] - contact1[1])
    t_len = math.hypot(tangent_row, tangent_col)
    if t_len <= 1e-6:
        tangent_row = math.sin(thetas[v2]) - math.sin(thetas[v1])
        tangent_col = math.cos(thetas[v2]) - math.cos(thetas[v1])
        t_len = max(1e-6, math.hypot(tangent_row, tangent_col))

    unit_t_row = tangent_row / t_len
    unit_t_col = tangent_col / t_len

    # Tangent angle in standard Cartesian (x = col, y = row)
    tangent_angle = math.atan2(unit_t_row, unit_t_col)

    # The two normal directions strictly perpendicular to tangent
    norm1_angle = wrap_angle(tangent_angle + math.pi / 2.0)
    norm2_angle = wrap_angle(tangent_angle - math.pi / 2.0)

    # Sample maximum clearance in a +/- 25 deg sector around each normal direction
    sector_rad = math.radians(25.0)
    pass1_mask = [
        abs(angle_diff(th, norm1_angle)) <= sector_rad for th in thetas
    ]
    pass2_mask = [
        abs(angle_diff(th, norm2_angle)) <= sector_rad for th in thetas
    ]
    cl_pass1 = float(np.max(R_smooth[pass1_mask])) if any(pass1_mask) else 0.0
    cl_pass2 = float(np.max(R_smooth[pass2_mask])) if any(pass2_mask) else 0.0
    passage_cl = 0.5 * (cl_pass1 + cl_pass2)

    min_passage_clearance = max(0.35, 0.5 * min_opening_m)
    if cl_pass1 < min_passage_clearance or cl_pass2 < min_passage_clearance:
        return PolarBifurcationResult(
            False,
            opening_width=width_m,
            reason=f"passage_blocked (cl1={cl_pass1:.2f}m, cl2={cl_pass2:.2f}m)",
        )

    # 3. Constriction ratio constraint: Space must open up relative to throat width
    constriction = (cl_pass1 + cl_pass2) / max(width_m, 0.1)
    if constriction < float(min_constriction):
        return PolarBifurcationResult(
            False,
            opening_width=width_m,
            constriction_ratio=constriction,
            passage_clearance=passage_cl,
            reason=f"insufficient_constriction ({constriction:.2f} < {min_constriction:.2f})",
        )

    # 4. Outward structural wall support check
    if min_wall_span_cells > 1:
        outward_1 = _check_outward_wall_run(
            structural_occupied,
            contact1,
            (-unit_t_row, -unit_t_col),
            min_wall_span_cells,
        )
        outward_2 = _check_outward_wall_run(
            structural_occupied,
            contact2,
            (unit_t_row, unit_t_col),
            min_wall_span_cells,
        )
        if not (outward_1 and outward_2):
            return PolarBifurcationResult(
                False,
                opening_width=width_m,
                constriction_ratio=constriction,
                reason="insufficient_outward_wall_span",
            )

    # 5. Strict orthogonal normal: n perp t_door
    norm_row = -unit_t_col
    norm_col = unit_t_row

    # 6. Architectural cardinal snapping: snap to pure horizontal/vertical if within tolerance
    (snapped_r, snapped_c), grid_step = snap_to_cardinal_if_close(
        (norm_row, norm_col),
        tolerance_deg=cardinal_snap_deg,
    )

    normal_angle = math.atan2(snapped_r, snapped_c)

    midpoint = (
        0.5 * (contact1[0] + contact2[0]),
        0.5 * (contact1[1] + contact2[1]),
    )

    return PolarBifurcationResult(
        is_bottleneck=True,
        opening_width=width_m,
        normal_rad=normal_angle,
        normal_vector=(snapped_r, snapped_c),
        normal_grid=grid_step,
        tangent_rad=tangent_angle,
        tangent_vector=(unit_t_row, unit_t_col),
        constriction_ratio=constriction,
        passage_clearance=passage_cl,
        jamb_clearance=width_m / 2.0,
        jamb_contacts=(contact1, contact2),
        midpoint_cell=midpoint,
        peaks_deg=(
            float(np.degrees(norm1_angle)),
            float(np.degrees(norm2_angle)),
        ),
        valleys_deg=(
            float(np.degrees(thetas[v1])),
            float(np.degrees(thetas[v2])),
        ),
        reason="valid_bottleneck",
    )


__all__ = [
    "PolarBifurcationResult",
    "wrap_angle",
    "angle_diff",
    "snap_to_cardinal_if_close",
    "sample_polar_clearance",
    "smooth_polar_field",
    "find_polar_extrema",
    "analyze_polar_bifurcation",
]
