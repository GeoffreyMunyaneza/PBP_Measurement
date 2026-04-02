"""
Evaluation metrics for BPD landmark detection and measurement.

Mirrors the metrics reported in Collins et al. 2026:
  - Localization error (px and mm)
  - Measurement error  (px and mm)
  - Success rate at multiple thresholds
  - Bland-Altman statistics
  - Precision-Recall / Average Precision
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional


# ── Point-level helpers ───────────────────────────────────────────────────────

def _euclidean(p1: tuple, p2: tuple) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


# ── Localization error ────────────────────────────────────────────────────────

def localization_errors(
    pred_left:   list[Optional[tuple]],
    pred_right:  list[Optional[tuple]],
    gt_left:     list[tuple],
    gt_right:    list[tuple],
    px_to_mm:    list[Optional[float]],
) -> dict:
    """
    Compute per-image localization errors for the left and right endpoints.

    A prediction of None counts as a "no detection" and is excluded from
    the error statistics but tallied separately.

    Args:
        pred_left/right : Predicted (x, y) or None per image.
        gt_left/right   : Ground-truth (x, y) per image.
        px_to_mm        : Scale factor per image (mm/px); None if unknown.

    Returns:
        Dict with:
            errors_px      – per-image mean localization error in pixels (2-pt avg)
            errors_mm      – per-image mean localization error in mm (NaN when no scale)
            n_no_detection – count of images with at least one missing prediction
            summary        – aggregate mean ± std and median
    """
    errors_px: list[float] = []
    errors_mm: list[float] = []
    n_no_detection = 0

    for pl, pr, gl, gr, scale in zip(pred_left, pred_right, gt_left, gt_right, px_to_mm):
        if pl is None or pr is None:
            n_no_detection += 1
            continue
        err_l = _euclidean(pl, gl)
        err_r = _euclidean(pr, gr)
        mean_err_px = (err_l + err_r) / 2.0
        errors_px.append(mean_err_px)

        if scale is not None and not math.isnan(scale) and scale > 0:
            errors_mm.append(mean_err_px * scale)
        else:
            errors_mm.append(float("nan"))

    arr_px = np.array(errors_px, dtype=float)
    arr_mm = np.array(errors_mm, dtype=float)
    valid_mm = arr_mm[~np.isnan(arr_mm)]

    return {
        "errors_px": errors_px,
        "errors_mm": errors_mm,
        "n_no_detection": n_no_detection,
        "summary": {
            "mean_px":   float(arr_px.mean()) if len(arr_px) else float("nan"),
            "std_px":    float(arr_px.std())  if len(arr_px) else float("nan"),
            "median_px": float(np.median(arr_px)) if len(arr_px) else float("nan"),
            "mean_mm":   float(valid_mm.mean())   if len(valid_mm) else float("nan"),
            "std_mm":    float(valid_mm.std())    if len(valid_mm) else float("nan"),
            "median_mm": float(np.median(valid_mm)) if len(valid_mm) else float("nan"),
        },
    }


# ── Measurement error ─────────────────────────────────────────────────────────

def measurement_errors(
    pred_bpd_px: list[Optional[float]],
    gt_bpd_px:   list[float],
    px_to_mm:    list[Optional[float]],
) -> dict:
    """
    Compute per-image BPD measurement errors.

    Returns:
        Dict with:
            errors_px, errors_mm  – per-image absolute errors
            n_no_detection        – images with no prediction
            summary               – aggregate stats
    """
    errors_px: list[float] = []
    errors_mm: list[float] = []
    n_no_detection = 0

    for pred, gt, scale in zip(pred_bpd_px, gt_bpd_px, px_to_mm):
        if pred is None:
            n_no_detection += 1
            continue
        err_px = abs(pred - gt)
        errors_px.append(err_px)
        if scale is not None and not math.isnan(scale) and scale > 0:
            errors_mm.append(err_px * scale)
        else:
            errors_mm.append(float("nan"))

    arr_px = np.array(errors_px, dtype=float)
    arr_mm = np.array(errors_mm, dtype=float)
    valid_mm = arr_mm[~np.isnan(arr_mm)]

    return {
        "errors_px": errors_px,
        "errors_mm": errors_mm,
        "n_no_detection": n_no_detection,
        "summary": {
            "mean_px":   float(arr_px.mean())     if len(arr_px) else float("nan"),
            "std_px":    float(arr_px.std())      if len(arr_px) else float("nan"),
            "median_px": float(np.median(arr_px)) if len(arr_px) else float("nan"),
            "mean_mm":   float(valid_mm.mean())   if len(valid_mm) else float("nan"),
            "std_mm":    float(valid_mm.std())    if len(valid_mm) else float("nan"),
            "median_mm": float(np.median(valid_mm)) if len(valid_mm) else float("nan"),
        },
    }


# ── Success rate ──────────────────────────────────────────────────────────────

def success_rate(
    errors_mm:  list[float],
    thresholds: list[float] = (0.5, 1.0, 2.0, 3.0),
    n_total:    Optional[int] = None,
) -> dict[float, dict]:
    """
    Compute the percentage of images within each error threshold (in mm).

    Args:
        errors_mm  : Per-image measurement errors in mm (NaN excluded).
        thresholds : Error ceilings in mm.
        n_total    : Total images including no-detections for denominator.
                     If None, uses len(errors_mm).

    Returns:
        Dict mapping threshold → {count, pct} for images ≤ threshold,
        plus a "> max" bucket for the remainder.
    """
    arr = np.array([e for e in errors_mm if not math.isnan(e)], dtype=float)
    n_denom = n_total if n_total is not None else len(arr)
    result: dict = {}

    for t in sorted(thresholds):
        count = int((arr <= t).sum())
        result[t] = {"count": count, "pct": 100.0 * count / n_denom if n_denom else 0.0}

    max_t = max(thresholds)
    over = int((arr > max_t).sum())
    result[f">{max_t}"] = {"count": over, "pct": 100.0 * over / n_denom if n_denom else 0.0}

    return result


# ── Bland-Altman ──────────────────────────────────────────────────────────────

def bland_altman_stats(
    pred_mm: list[float],
    gt_mm:   list[float],
) -> dict:
    """
    Compute Bland-Altman agreement statistics.

    Returns:
        Dict with mean_diff, std_diff, upper_loa, lower_loa, mean_means.
    """
    pred = np.array(pred_mm, dtype=float)
    gt   = np.array(gt_mm,   dtype=float)
    diff = pred - gt
    mean_means = (pred + gt) / 2.0

    mean_diff = float(diff.mean())
    std_diff  = float(diff.std())

    return {
        "mean_diff":  mean_diff,
        "std_diff":   std_diff,
        "upper_loa":  mean_diff + 1.96 * std_diff,
        "lower_loa":  mean_diff - 1.96 * std_diff,
        "mean_means": mean_means.tolist(),
        "diffs":      diff.tolist(),
    }


# ── Precision-Recall / Average Precision ──────────────────────────────────────

def precision_recall_ap(
    pred_left:  list[Optional[tuple]],
    pred_right: list[Optional[tuple]],
    gt_left:    list[tuple],
    gt_right:   list[tuple],
    px_to_mm:   list[Optional[float]],
    thresholds_mm: list[float] = (1.0, 2.0, 3.0, 4.0),
) -> dict[float, dict]:
    """
    Compute precision-recall curves and AP at each distance threshold.

    A predicted endpoint is a True Positive if it falls within `threshold` mm
    of the corresponding ground-truth point.

    Returns:
        Dict mapping threshold (mm) → {precision, recall, AP}.
    """
    results: dict = {}

    for thr in thresholds_mm:
        tp_l = tp_r = fp_l = fp_r = fn_l = fn_r = 0

        for pl, pr, gl, gr, scale in zip(pred_left, pred_right, gt_left, gt_right, px_to_mm):
            if scale is None or math.isnan(scale) or scale <= 0:
                continue
            thr_px = thr / scale

            if pl is not None:
                if _euclidean(pl, gl) <= thr_px:
                    tp_l += 1
                else:
                    fp_l += 1
            else:
                fn_l += 1

            if pr is not None:
                if _euclidean(pr, gr) <= thr_px:
                    tp_r += 1
                else:
                    fp_r += 1
            else:
                fn_r += 1

        tp = tp_l + tp_r
        fp = fp_l + fp_r
        fn = fn_l + fn_r

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ap        = precision * recall   # single operating point — area approximation

        results[thr] = {"precision": precision, "recall": recall, "ap": ap}

    return results
