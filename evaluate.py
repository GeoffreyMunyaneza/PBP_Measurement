"""
Phase 3 - Evaluation script for trained BPD heatmap regression model.

Runs inference on the test split, applies NMS + greedy endpoint selection,
and reports all metrics from Collins et al. 2026:
    - Localization error (px and mm)
    - Measurement error  (px and mm)
    - Success rate at 0.5 / 1 / 2 / 3 mm
    - Precision-Recall / Average Precision at 1–4 mm
    - Bland-Altman statistics and plot

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
                       [--split test] [--output-dir results/]
"""

import argparse
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import BPDDataset, IMG_SIZE
from src.heatmap import extract_peaks
from src.metrics import (
    bland_altman_stats,
    localization_errors,
    measurement_errors,
    precision_recall_ap,
    success_rate,
)
from src.model import load_checkpoint
from src.postprocess import greedy_select_endpoints, measure_bpd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BPD heatmap regression model.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint (best.pt / last.pt)")
    p.add_argument("--annotations", default="data/annotations.csv")
    p.add_argument("--root-dir", default=".")
    p.add_argument("--split", default="test",
                   help="Which split to evaluate: train / val / test")
    p.add_argument("--output-dir", default="results",
                   help="Directory for plots and results CSV")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--nms-size", type=int, default=21,
                   help="NMS window size in heatmap pixels (default: 21)")
    p.add_argument("--nms-threshold", type=float, default=0.1,
                   help="NMS peak threshold relative to heatmap max (default: 0.1)")
    return p.parse_args()


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    nms_size: int,
    nms_threshold: float,
) -> list[dict]:
    """
    Run the model on all batches, extract landmarks, and return per-image results.

    For each image the function:
        1. Runs the U-Net to get 3 raw heatmap channels.
        2. Applies sigmoid to get values in [0, 1].
        3. Runs NMS on each channel to find candidate peaks.
        4. Runs the greedy selection algorithm (Algorithm 1) to pick the final
           (left, right) pair, with center heatmap as a false-positive filter.
        5. Scales predicted keypoints back to original image pixel space.
        6. Computes predicted BPD in pixels (and mm when scale is available).

    Returns:
        List of dicts with keys:
            image_path, gt_x_left, gt_y_left, gt_x_right, gt_y_right,
            gt_bpd_px, px_to_mm,
            pred_x_left, pred_y_left, pred_x_right, pred_y_right,
            pred_bpd_px, pred_bpd_mm, no_detection (bool)
    """
    model.eval()
    results = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        metas  = batch["meta"]

        # Forward pass
        raw_preds = model(images)                  # (B, 3, H, W)
        preds = torch.sigmoid(raw_preds).cpu().numpy()   # values in [0, 1]

        batch_size = images.shape[0]
        for i in range(batch_size):
            orig_w   = int(metas["orig_w"][i])
            orig_h   = int(metas["orig_h"][i])
            scale_x  = float(metas["scale_x"][i])   # IMG_SIZE / orig_w
            scale_y  = float(metas["scale_y"][i])   # IMG_SIZE / orig_h
            px_to_mm_val = float(metas["px_to_mm"][i])
            px_to_mm = None if math.isnan(px_to_mm_val) else px_to_mm_val

            hm_left   = preds[i, 0]   # (H, W)
            hm_right  = preds[i, 1]
            hm_center = preds[i, 2]

            # Extract NMS peaks in heatmap space
            L = extract_peaks(hm_left,   nms_size=nms_size, threshold=nms_threshold)
            R = extract_peaks(hm_right,  nms_size=nms_size, threshold=nms_threshold)
            C = extract_peaks(hm_center, nms_size=nms_size, threshold=nms_threshold)

            # Greedy selection
            pred_l_hm, pred_r_hm = greedy_select_endpoints(L, R, C)

            no_detection = (pred_l_hm is None or pred_r_hm is None)

            if no_detection:
                rec = {
                    "image_path":  metas["image_path"][i],
                    "gt_x_left":   float(metas["gt_x_left"][i]),
                    "gt_y_left":   float(metas["gt_y_left"][i]),
                    "gt_x_right":  float(metas["gt_x_right"][i]),
                    "gt_y_right":  float(metas["gt_y_right"][i]),
                    "gt_bpd_px":   float(metas["gt_bpd_px"][i]),
                    "px_to_mm":    px_to_mm,
                    "pred_x_left":  None,
                    "pred_y_left":  None,
                    "pred_x_right": None,
                    "pred_y_right": None,
                    "pred_bpd_px":  None,
                    "pred_bpd_mm":  None,
                    "no_detection": True,
                }
            else:
                # Scale back to original image space
                pred_xl = pred_l_hm[0] / scale_x
                pred_yl = pred_l_hm[1] / scale_y
                pred_xr = pred_r_hm[0] / scale_x
                pred_yr = pred_r_hm[1] / scale_y

                measurement = measure_bpd(
                    (pred_xl, pred_yl),
                    (pred_xr, pred_yr),
                    px_to_mm=px_to_mm,
                )

                rec = {
                    "image_path":  metas["image_path"][i],
                    "gt_x_left":   float(metas["gt_x_left"][i]),
                    "gt_y_left":   float(metas["gt_y_left"][i]),
                    "gt_x_right":  float(metas["gt_x_right"][i]),
                    "gt_y_right":  float(metas["gt_y_right"][i]),
                    "gt_bpd_px":   float(metas["gt_bpd_px"][i]),
                    "px_to_mm":    px_to_mm,
                    "pred_x_left":  pred_xl,
                    "pred_y_left":  pred_yl,
                    "pred_x_right": pred_xr,
                    "pred_y_right": pred_yr,
                    "pred_bpd_px":  measurement["bpd_px"],
                    "pred_bpd_mm":  measurement.get("bpd_mm"),
                    "no_detection": False,
                }

            results.append(rec)

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_bland_altman(
    ba_stats: dict,
    output_path: Path,
    title: str = "Bland-Altman Plot",
) -> None:
    """Save a Bland-Altman plot to disk."""
    means = np.array(ba_stats["mean_means"])
    diffs = np.array(ba_stats["diffs"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(means, diffs, alpha=0.4, s=12, color="steelblue", edgecolors="none")
    ax.axhline(ba_stats["mean_diff"], color="red",    linestyle="--", label=f"Mean diff: {ba_stats['mean_diff']:.2f}")
    ax.axhline(ba_stats["upper_loa"], color="orange", linestyle=":",  label=f"+1.96 SD: {ba_stats['upper_loa']:.2f}")
    ax.axhline(ba_stats["lower_loa"], color="orange", linestyle=":",  label=f"-1.96 SD: {ba_stats['lower_loa']:.2f}")
    ax.set_xlabel("Mean of BPD measurements (mm)")
    ax.set_ylabel("Difference (predicted − ground truth) (mm)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_success_rate(
    sr: dict,
    output_path: Path,
    title: str = "BPD Measurement Success Rate",
) -> None:
    """Bar chart of success rates at each threshold."""
    thresholds = [k for k in sr if isinstance(k, (int, float))]
    pcts = [sr[t]["pct"] for t in thresholds]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(t) for t in thresholds], pcts, color="steelblue", alpha=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Error threshold (mm)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def report_results(results: list[dict], output_dir: Path) -> None:
    """Compute all metrics, print a summary table, and save plots."""
    n_total = len(results)
    detected = [r for r in results if not r["no_detection"]]
    n_no_det = n_total - len(detected)

    print_section(f"Evaluation — {n_total} images  |  {n_no_det} no-detections")

    # ── Unpack lists ───────────────────────────────────────────────────────
    gt_left   = [(r["gt_x_left"],  r["gt_y_left"])  for r in results]
    gt_right  = [(r["gt_x_right"], r["gt_y_right"]) for r in results]
    pred_left  = [(r["pred_x_left"],  r["pred_y_left"])  if not r["no_detection"] else None for r in results]
    pred_right = [(r["pred_x_right"], r["pred_y_right"]) if not r["no_detection"] else None for r in results]
    gt_bpd_px  = [r["gt_bpd_px"]  for r in results]
    pred_bpd_px= [r["pred_bpd_px"] for r in results]
    px_to_mm   = [r["px_to_mm"]    for r in results]

    # ── Localization error ─────────────────────────────────────────────────
    loc = localization_errors(pred_left, pred_right, gt_left, gt_right, px_to_mm)
    s = loc["summary"]
    print("\n  Localization Error")
    print(f"    Pixels : {s['mean_px']:.4f} ± {s['std_px']:.4f}  (median {s['median_px']:.4f})")
    if not math.isnan(s["mean_mm"]):
        print(f"    mm     : {s['mean_mm']:.4f} ± {s['std_mm']:.4f}  (median {s['median_mm']:.4f})")
    print(f"    No detections : {loc['n_no_detection']}")

    # ── Measurement error ──────────────────────────────────────────────────
    meas = measurement_errors(pred_bpd_px, gt_bpd_px, px_to_mm)
    sm = meas["summary"]
    print("\n  BPD Measurement Error")
    print(f"    Pixels : {sm['mean_px']:.4f} ± {sm['std_px']:.4f}  (median {sm['median_px']:.4f})")
    if not math.isnan(sm["mean_mm"]):
        print(f"    mm     : {sm['mean_mm']:.4f} ± {sm['std_mm']:.4f}  (median {sm['median_mm']:.4f})")
    print(f"    No detections : {meas['n_no_detection']}")

    # ── Success rate ───────────────────────────────────────────────────────
    valid_errors_mm = [e for e in meas["errors_mm"] if not math.isnan(e)]
    sr = success_rate(valid_errors_mm, thresholds=[0.5, 1.0, 2.0, 3.0], n_total=n_total)
    print("\n  Success Rate (BPD measurement error in mm)")
    for t in [0.5, 1.0, 2.0, 3.0]:
        info = sr[t]
        print(f"    <= {t:3.1f} mm : {info['count']:4d} / {n_total}  ({info['pct']:.1f}%)")
    over = sr[">3.0"]
    print(f"    > 3.0 mm : {over['count']:4d} / {n_total}  ({over['pct']:.1f}%)")

    # ── Precision-Recall / AP ──────────────────────────────────────────────
    pr = precision_recall_ap(
        pred_left, pred_right, gt_left, gt_right, px_to_mm,
        thresholds_mm=[1.0, 2.0, 3.0, 4.0],
    )
    print("\n  Precision / Recall / AP (localization threshold)")
    for thr, info in pr.items():
        print(f"    {thr:.0f} mm : P={info['precision']:.3f}  R={info['recall']:.3f}  AP={info['ap']:.3f}")

    # ── Bland-Altman ───────────────────────────────────────────────────────
    pred_bpd_mm_valid = [
        r["pred_bpd_mm"] for r in results
        if r["pred_bpd_mm"] is not None and not math.isnan(r["pred_bpd_mm"])
    ]
    gt_bpd_mm_valid = [
        r["gt_bpd_px"] * r["px_to_mm"]
        for r in results
        if r["pred_bpd_mm"] is not None
        and not math.isnan(r["pred_bpd_mm"])
        and r["px_to_mm"] is not None
    ]

    if pred_bpd_mm_valid and len(pred_bpd_mm_valid) == len(gt_bpd_mm_valid):
        ba = bland_altman_stats(pred_bpd_mm_valid, gt_bpd_mm_valid)
        print("\n  Bland-Altman (mm)")
        print(f"    Mean diff   : {ba['mean_diff']:.3f} mm")
        print(f"    SD diff     : {ba['std_diff']:.3f} mm")
        print(f"    +1.96 SD    : {ba['upper_loa']:.3f} mm")
        print(f"    -1.96 SD    : {ba['lower_loa']:.3f} mm")

        plot_bland_altman(ba, output_dir / "bland_altman.png", title="BPD Bland-Altman Plot")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_success_rate(sr, output_dir / "success_rate.png")

    # ── Save CSV ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "test_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results CSV saved → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"  Trained to epoch {ckpt.get('epoch', '?')}, val loss {ckpt.get('val_loss', '?'):.6f}")

    # ── Build dataset ──────────────────────────────────────────────────────
    ann = pd.read_csv(args.annotations)
    split_ann = ann[ann["split"] == args.split].copy()
    print(f"\nEvaluating {args.split} split: {len(split_ann)} images")

    dataset = BPDDataset(split_ann, root_dir=args.root_dir, augment=False)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Inference ──────────────────────────────────────────────────────────
    print("Running inference ...")
    results = run_inference(
        model, loader, device,
        nms_size=args.nms_size,
        nms_threshold=args.nms_threshold,
    )

    # ── Metrics & plots ────────────────────────────────────────────────────
    report_results(results, output_dir)


if __name__ == "__main__":
    main()
