"""
Phase 3 — Single-image inference with BPD measurement visualisation.

Given a trained checkpoint and an image path, predicts the BPD endpoints,
measures the diameter, and saves an annotated overlay image.

Usage:
    python predict.py --checkpoint checkpoints/best.pt
                      --image data/images/trans_thalamic/Patient00168_Plane3_1_of_3.png
                      [--px-to-mm 0.14] [--output predicted.png]
"""

import argparse
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.dataset import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.heatmap import extract_peaks
from src.model import load_checkpoint
from src.postprocess import greedy_select_endpoints, measure_bpd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict BPD on a single ultrasound image.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained model checkpoint")
    p.add_argument("--image", required=True,
                   help="Path to the input ultrasound image")
    p.add_argument("--px-to-mm", type=float, default=None,
                   help="Pixel-to-mm conversion factor (mm per pixel). "
                        "If omitted, only pixel measurement is reported.")
    p.add_argument("--output", default=None,
                   help="Output image path (default: <image_stem>_predicted.png)")
    p.add_argument("--nms-size", type=int, default=21)
    p.add_argument("--nms-threshold", type=float, default=0.1)
    return p.parse_args()


# ── Pre-processing ────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> tuple[torch.Tensor, np.ndarray, int, int]:
    """
    Load and pre-process an image for model inference.

    Returns:
        image_tensor : (1, 3, IMG_SIZE, IMG_SIZE) float32 tensor, normalised.
        image_rgb    : (H, W, 3) uint8 array (original, for overlay).
        orig_w, orig_h : original image dimensions.
    """
    img = Image.open(image_path).convert("L")   # grayscale
    orig_w, orig_h = img.size

    # Resize to model input resolution
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0

    # Normalise using ImageNet statistics (applied to each replicated channel)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr3 = np.stack([arr, arr, arr], axis=-1)  # (H, W, 3)
    arr3 = (arr3 - mean) / std                 # normalise

    image_tensor = torch.from_numpy(arr3.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)

    # Original image (grayscale → RGB for coloured overlay)
    image_rgb = np.array(img.resize((orig_w, orig_h)).convert("RGB"), dtype=np.uint8)

    return image_tensor, image_rgb, orig_w, orig_h


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    nms_size: int,
    nms_threshold: float,
) -> tuple:
    """
    Run model inference on a single image tensor.

    Returns:
        heatmaps    : (3, H, W) float32 numpy array (after sigmoid).
        left_cands  : NMS candidates for left endpoint.
        right_cands : NMS candidates for right endpoint.
        center_cands: NMS candidates for center point.
        pred_left   : Selected left endpoint (heatmap space) or None.
        pred_right  : Selected right endpoint (heatmap space) or None.
    """
    image_tensor = image_tensor.to(device)
    raw = model(image_tensor)                     # (1, 3, H, W)
    heatmaps = torch.sigmoid(raw).squeeze(0).cpu().numpy()  # (3, H, W)

    L = extract_peaks(heatmaps[0], nms_size=nms_size, threshold=nms_threshold)
    R = extract_peaks(heatmaps[1], nms_size=nms_size, threshold=nms_threshold)
    C = extract_peaks(heatmaps[2], nms_size=nms_size, threshold=nms_threshold)

    pred_left, pred_right = greedy_select_endpoints(L, R, C)

    return heatmaps, L, R, C, pred_left, pred_right


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_and_save(
    image_rgb:   np.ndarray,
    heatmaps:    np.ndarray,
    pred_left:   tuple | None,
    pred_right:  tuple | None,
    orig_w: int,
    orig_h: int,
    measurement: dict | None,
    output_path: Path,
) -> None:
    """
    Save a 3-panel figure:
        Left  – original US image with BPD line and endpoints overlaid.
        Centre – predicted heatmap composite (max across 3 channels).
        Right  – heatmap composite overlaid on the image.
    """
    # Scale heatmap-space predictions → original image space for overlay
    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: original image + overlay ─────────────────────────────────
    ax = axes[0]
    ax.imshow(image_rgb)
    ax.set_title("BPD Prediction")

    if pred_left is not None and pred_right is not None:
        xl = pred_left[0]  * scale_x
        yl = pred_left[1]  * scale_y
        xr = pred_right[0] * scale_x
        yr = pred_right[1] * scale_y

        ax.plot([xl, xr], [yl, yr], color="red", linewidth=2.5)
        ax.scatter([xl, xr], [yl, yr], color="lime", s=60, zorder=5)

        label = f"BPD = {measurement['bpd_px']:.1f} px"
        if "bpd_mm" in measurement:
            label += f"  ({measurement['bpd_mm']:.1f} mm)"
        ax.set_xlabel(label, fontsize=10)
    else:
        ax.set_xlabel("No detection", color="red", fontsize=10)

    # ── Panel 2: heatmap composite ─────────────────────────────────────────
    ax = axes[1]
    composite = heatmaps.max(axis=0)
    ax.imshow(composite, cmap="hot", vmin=0, vmax=1)
    ax.set_title("Predicted Heatmaps (max composite)")

    colours = {"Left": "cyan", "Right": "magenta", "Center": "lime"}
    for label, pts, col in [
        ("Left",   [pred_left]  if pred_left  else [], "cyan"),
        ("Right",  [pred_right] if pred_right else [], "magenta"),
    ]:
        for pt in pts:
            ax.scatter(pt[0], pt[1], color=col, s=60, zorder=5)

    patches = [mpatches.Patch(color=c, label=l) for l, c in colours.items()]
    ax.legend(handles=patches[:2], fontsize=7, loc="upper right")

    # ── Panel 3: overlay ──────────────────────────────────────────────────
    ax = axes[2]
    overlay_img = Image.fromarray(image_rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    ax.imshow(np.array(overlay_img))
    heat_rgb = plt.cm.hot(composite)[:, :, :3]   # (H, W, 3)
    ax.imshow(heat_rgb, alpha=0.4)
    ax.set_title("Heatmap Overlay")

    for ax_ in axes:
        ax_.axis("off")

    fig.suptitle(output_path.stem, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Load model
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Checkpoint : epoch {ckpt.get('epoch', '?')}")

    # Pre-process
    image_tensor, image_rgb, orig_w, orig_h = preprocess(args.image)

    # Inference
    heatmaps, L, R, C, pred_left, pred_right = predict(
        model, image_tensor, device,
        nms_size=args.nms_size,
        nms_threshold=args.nms_threshold,
    )

    # Scale to original space
    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE
    measurement = None

    if pred_left is not None and pred_right is not None:
        left_orig  = (pred_left[0]  * scale_x, pred_left[1]  * scale_y)
        right_orig = (pred_right[0] * scale_x, pred_right[1] * scale_y)
        measurement = measure_bpd(left_orig, right_orig, px_to_mm=args.px_to_mm)

        print(f"\nLeft  endpoint : ({left_orig[0]:.1f}, {left_orig[1]:.1f}) px")
        print(f"Right endpoint : ({right_orig[0]:.1f}, {right_orig[1]:.1f}) px")
        print(f"BPD            : {measurement['bpd_px']:.2f} px", end="")
        if "bpd_mm" in measurement:
            print(f"  =  {measurement['bpd_mm']:.2f} mm")
        else:
            print()
    else:
        print("\nNo endpoints detected.")

    # Determine output path
    image_stem = Path(args.image).stem
    out_path = Path(args.output) if args.output else Path(f"{image_stem}_predicted.png")

    # Visualise
    visualise_and_save(
        image_rgb, heatmaps, pred_left, pred_right,
        orig_w, orig_h, measurement, out_path,
    )


if __name__ == "__main__":
    main()
