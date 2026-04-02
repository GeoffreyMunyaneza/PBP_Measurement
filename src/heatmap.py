"""
Gaussian heatmap generation and non-maximal suppression (NMS) peak extraction.

Follows Collins et al. 2026:
  - Each landmark is encoded as a filled Gaussian circle (sigma = 5 px at 256x256).
  - Peaks are extracted with NMS using a sliding maximum filter.
"""

import numpy as np
from scipy.ndimage import maximum_filter


# ── Heatmap generation ────────────────────────────────────────────────────────

def make_gaussian_heatmap(
    height: int,
    width: int,
    cx: float,
    cy: float,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Create a 2D Gaussian heatmap centered at pixel (cx, cy).

    Args:
        height, width : Spatial dimensions of the output array.
        cx, cy        : Center coordinates in pixel space (can be fractional).
        sigma         : Standard deviation in pixels.

    Returns:
        Float32 array of shape (height, width) with values in [0, 1].
    """
    Y, X = np.ogrid[:height, :width]
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    heatmap = np.exp(-dist_sq / (2.0 * sigma ** 2))
    return heatmap.astype(np.float32)


def make_target_heatmaps(
    height: int,
    width: int,
    x_left: float,
    y_left: float,
    x_right: float,
    y_right: float,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Build the 3-channel ground-truth heatmap tensor for one image.

    Channel layout (matches model output):
        0 – left  BPD endpoint
        1 – right BPD endpoint
        2 – center point  (midpoint of left & right, free annotation)

    Args:
        height, width         : Spatial dimensions (heatmap resolution).
        x_left/y_left         : Left endpoint in heatmap-space pixels.
        x_right/y_right       : Right endpoint in heatmap-space pixels.
        sigma                 : Gaussian sigma in heatmap-space pixels.

    Returns:
        Float32 array of shape (3, height, width).
    """
    x_center = (x_left + x_right) / 2.0
    y_center = (y_left + y_right) / 2.0

    heatmaps = np.stack([
        make_gaussian_heatmap(height, width, x_left,   y_left,   sigma),
        make_gaussian_heatmap(height, width, x_right,  y_right,  sigma),
        make_gaussian_heatmap(height, width, x_center, y_center, sigma),
    ], axis=0)

    return heatmaps  # (3, H, W)


# ── Peak extraction (NMS) ─────────────────────────────────────────────────────

def extract_peaks(
    heatmap: np.ndarray,
    nms_size: int = 21,
    threshold: float = 0.1,
) -> list[tuple[float, float]]:
    """
    Extract local-maximum locations from a single 2D heatmap.

    A pixel is a peak if it equals the maximum within a (nms_size × nms_size)
    neighbourhood AND its value is above `threshold * global_max`.

    Args:
        heatmap   : 2D float array (H, W).
        nms_size  : Side length of the NMS window (must be odd, ≥ 1).
        threshold : Fraction of the global maximum used as a floor.

    Returns:
        List of (x, y) float coordinates, sorted by descending heatmap value.
        Empty list when no peak is found.
    """
    if heatmap.max() < 1e-6:
        return []

    nms_size = nms_size if nms_size % 2 == 1 else nms_size + 1
    max_filtered = maximum_filter(heatmap, size=nms_size)
    min_val = threshold * float(heatmap.max())

    peak_mask = (heatmap == max_filtered) & (heatmap >= min_val)
    peak_ys, peak_xs = np.where(peak_mask)

    if len(peak_ys) == 0:
        return []

    values = heatmap[peak_ys, peak_xs]
    order = np.argsort(values)[::-1]

    return [(float(peak_xs[i]), float(peak_ys[i])) for i in order]
