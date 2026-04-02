"""
Post-processing: greedy endpoint selection and BPD measurement.

Implements Algorithm 1 from Collins et al. 2026:
  Given candidate sets {L}, {R}, {C} extracted by NMS from the three
  U-Net heatmap channels, select the (left, right) pair whose geometric
  midpoint is closest to the predicted center, eliminating false positives.
"""

import math
from typing import Optional


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _midpoint(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


# ── Greedy endpoint selection (Algorithm 1) ───────────────────────────────────

def greedy_select_endpoints(
    left_candidates:   list[tuple[float, float]],
    right_candidates:  list[tuple[float, float]],
    center_candidates: list[tuple[float, float]],
) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    """
    Select the best (left, right) BPD endpoint pair using the center heatmap
    to disambiguate false positives (Algorithm 1, Collins et al. 2026).

    Rules:
        |C| == 0 : return the highest-confidence L and R directly.
        |C| == 1 : find (li, ri) minimising dist(midpoint(li,ri), c).
        |C|  > 1 : pick center with smallest y (nearest US probe, top of image),
                   then apply the same minimisation.

    Args:
        left_candidates   : [(x, y), ...] sorted by descending heatmap value.
        right_candidates  : [(x, y), ...] sorted by descending heatmap value.
        center_candidates : [(x, y), ...] sorted by descending heatmap value.

    Returns:
        (left_pt, right_pt) or (None, None) if either set is empty.
    """
    if not left_candidates or not right_candidates:
        return None, None

    # No center detected — use best-scoring candidates
    if not center_candidates:
        return left_candidates[0], right_candidates[0]

    # Multiple centers: pick the one nearest the probe (smallest y = top)
    chosen_center = min(center_candidates, key=lambda p: p[1])

    # Find pair minimising distance from geometric midpoint to chosen center
    best_left: Optional[tuple] = None
    best_right: Optional[tuple] = None
    best_dist = math.inf

    for li in left_candidates:
        for ri in right_candidates:
            midpt = _midpoint(li, ri)
            d = _dist(chosen_center, midpt)
            if d < best_dist:
                best_dist = d
                best_left = li
                best_right = ri

    return best_left, best_right  # type: ignore[return-value]


# ── BPD measurement ───────────────────────────────────────────────────────────

def measure_bpd(
    left_pt:   tuple[float, float],
    right_pt:  tuple[float, float],
    px_to_mm:  Optional[float] = None,
) -> dict:
    """
    Compute BPD from two endpoints.

    Args:
        left_pt, right_pt : (x, y) coordinates in pixel space.
        px_to_mm          : Conversion factor (mm per pixel). Pass None or NaN
                            when unavailable.

    Returns:
        Dict with keys:
            bpd_px  – Euclidean distance in pixels.
            bpd_mm  – Distance in millimetres (only when px_to_mm is valid).
    """
    bpd_px = _dist(left_pt, right_pt)
    result: dict = {"bpd_px": bpd_px}

    if px_to_mm is not None and not math.isnan(px_to_mm) and px_to_mm > 0:
        result["bpd_mm"] = bpd_px * px_to_mm

    return result
