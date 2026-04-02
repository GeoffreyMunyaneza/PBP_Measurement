"""
Phase 2: Fetch BPD/head annotations for FETAL_PLANES trans-thalamic images.

Source: Multi-Centre Fetal Biometry dataset (surgical-vision/Multicentre-Fetal-Biometry)
        FP/Head.csv — contains BPD + OFD landmarks for FETAL_PLANES images.

BPD (Biparietal Diameter): bpd_1/bpd_2 — the two outer parietal bone endpoints.
OFD (Occipito-Frontal Diameter): ofd_1/ofd_2 — perpendicular diameter.

Together, BPD and OFD define the head ellipse, which we render as a binary
segmentation mask (255 = head, 0 = background).

Outputs:
  data/annotations.csv   — per-image keypoints + ellipse parameters
  data/masks/*.png       — binary head segmentation masks
"""

import io
import math
import sys
sys.stdout.reconfigure(encoding="utf-8")

import requests
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images" / "trans_thalamic"
MASKS_DIR  = DATA_DIR / "masks"
MANIFEST   = DATA_DIR / "trans_thalamic_manifest.csv"
ANN_OUT    = DATA_DIR / "annotations.csv"

MASKS_DIR.mkdir(parents=True, exist_ok=True)

FP_HEAD_URL = (
    "https://raw.githubusercontent.com/surgical-vision/"
    "Multicentre-Fetal-Biometry/main/data/annotations/FP/Head.csv"
)


# ── 1. Download FP annotation CSV ─────────────────────────────────────────
def fetch_fp_annotations(url: str) -> pd.DataFrame:
    print(f"[fetch] Downloading FP/Head.csv ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    print(f"[fetch] {len(df)} rows, columns: {list(df.columns)}")
    return df


# ── 2. Match to manifest ───────────────────────────────────────────────────
def match_to_manifest(fp_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Join FP annotations to the user's manifest on image filename."""
    # manifest image_path: "data/images/trans_thalamic/Patient00683_Plane3_1_of_2.png"
    manifest = manifest.copy()
    manifest["image_name"] = manifest["image_path"].apply(
        lambda p: Path(p).name
    )

    # FP image_name column — ensure .png extension
    fp_df = fp_df.copy()
    fp_df["image_name"] = fp_df["image_name"].apply(
        lambda n: n if str(n).lower().endswith(".png") else str(n) + ".png"
    )

    merged = manifest.merge(fp_df, on="image_name", how="left")

    n_total  = len(manifest)
    n_matched = merged["bpd_1_x"].notna().sum()
    print(f"[match] {n_matched}/{n_total} images have FP annotations ({100*n_matched/n_total:.1f}%)")

    missing = merged[merged["bpd_1_x"].isna()]["image_name"].tolist()
    if missing:
        print(f"[match] {len(missing)} images without annotations (first 5): {missing[:5]}")

    return merged


# ── 3. Ellipse geometry helpers ────────────────────────────────────────────
def dist(x1, y1, x2, y2) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def ellipse_params(row) -> dict | None:
    """
    Compute ellipse center, semi-axes, and rotation angle from BPD + OFD landmarks.

    The head circumference ellipse has:
      - semi-major axis = OFD / 2
      - semi-minor axis = BPD / 2
      - center = midpoint of OFD (≈ midpoint of BPD; we average both)
      - angle  = orientation of the OFD axis
    """
    try:
        bpd_1x, bpd_1y = float(row["bpd_1_x"]), float(row["bpd_1_y"])
        bpd_2x, bpd_2y = float(row["bpd_2_x"]), float(row["bpd_2_y"])
        ofd_1x, ofd_1y = float(row["ofd_1_x"]), float(row["ofd_1_y"])
        ofd_2x, ofd_2y = float(row["ofd_2_x"]), float(row["ofd_2_y"])
    except (ValueError, TypeError):
        return None

    bpd_cx = (bpd_1x + bpd_2x) / 2
    bpd_cy = (bpd_1y + bpd_2y) / 2
    ofd_cx = (ofd_1x + ofd_2x) / 2
    ofd_cy = (ofd_1y + ofd_2y) / 2

    cx = (bpd_cx + ofd_cx) / 2
    cy = (bpd_cy + ofd_cy) / 2

    semi_ofd = dist(ofd_1x, ofd_1y, ofd_2x, ofd_2y) / 2  # semi-major
    semi_bpd = dist(bpd_1x, bpd_1y, bpd_2x, bpd_2y) / 2  # semi-minor

    # Rotation: angle of the OFD axis (major axis)
    angle_rad = math.atan2(ofd_2y - ofd_1y, ofd_2x - ofd_1x)

    return {
        "ell_cx": cx,
        "ell_cy": cy,
        "ell_semi_ofd": semi_ofd,
        "ell_semi_bpd": semi_bpd,
        "ell_angle_deg": math.degrees(angle_rad),
        "_angle_rad": angle_rad,
    }


# ── 4. Generate segmentation mask ─────────────────────────────────────────
def make_ellipse_mask(img_hw: tuple[int, int], cx: float, cy: float,
                      semi_a: float, semi_b: float, angle_rad: float) -> np.ndarray:
    """
    Return a binary uint8 mask (255 inside ellipse, 0 outside).

    semi_a: semi-major axis (OFD/2), aligned with angle_rad.
    semi_b: semi-minor axis (BPD/2), perpendicular to angle_rad.
    """
    h, w = img_hw
    Y, X = np.ogrid[:h, :w]
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = X - cx
    dy = Y - cy
    xr = cos_a * dx + sin_a * dy   # coordinate along major axis
    yr = -sin_a * dx + cos_a * dy  # coordinate along minor axis
    inside = (xr / semi_a) ** 2 + (yr / semi_b) ** 2 <= 1
    return (inside * 255).astype(np.uint8)


def get_image_size(image_path: str) -> tuple[int, int] | None:
    """Return (height, width) of the image, or None if unreadable."""
    full = BASE_DIR / image_path
    try:
        with Image.open(full) as img:
            w, h = img.size
        return (h, w)
    except Exception:
        return None


# ── 5. Build output annotation rows ───────────────────────────────────────
def build_annotations(merged: pd.DataFrame) -> pd.DataFrame:
    """
    For each annotated image: normalise BPD left/right by x-coordinate,
    compute BPD center, compute ellipse params, generate mask.
    """
    rows = []
    annotated = merged[merged["bpd_1_x"].notna()].copy()

    for _, row in tqdm(annotated.iterrows(), total=len(annotated),
                       desc="Building annotations & masks"):
        b1x, b1y = float(row["bpd_1_x"]), float(row["bpd_1_y"])
        b2x, b2y = float(row["bpd_2_x"]), float(row["bpd_2_y"])

        # Assign left/right by x-coordinate for consistency
        if b1x <= b2x:
            x_left, y_left, x_right, y_right = b1x, b1y, b2x, b2y
        else:
            x_left, y_left, x_right, y_right = b2x, b2y, b1x, b1y

        x_center = (x_left + x_right) / 2
        y_center = (y_left + y_right) / 2

        bpd_px = dist(x_left, y_left, x_right, y_right)
        ofd_px = dist(float(row["ofd_1_x"]), float(row["ofd_1_y"]),
                      float(row["ofd_2_x"]), float(row["ofd_2_y"]))

        ell = ellipse_params(row)

        # Generate mask
        mask_fname = Path(row["image_path"]).name
        mask_path  = f"data/masks/{mask_fname}"
        mask_saved = False

        if ell is not None:
            img_hw = get_image_size(row["image_path"])
            if img_hw is not None:
                mask = make_ellipse_mask(
                    img_hw,
                    ell["ell_cx"], ell["ell_cy"],
                    ell["ell_semi_ofd"], ell["ell_semi_bpd"],
                    ell["_angle_rad"],
                )
                mask_out = MASKS_DIR / mask_fname
                Image.fromarray(mask, mode="L").save(mask_out)
                mask_saved = True

        rec = {
            "image_path":    row["image_path"],
            "split":         row.get("split", ""),
            "patient_id":    row.get("patient_id", ""),
            # BPD keypoints
            "x_left":        x_left,
            "y_left":        y_left,
            "x_right":       x_right,
            "y_right":       y_right,
            "x_center":      x_center,
            "y_center":      y_center,
            # OFD keypoints
            "ofd_1_x":       row["ofd_1_x"],
            "ofd_1_y":       row["ofd_1_y"],
            "ofd_2_x":       row["ofd_2_x"],
            "ofd_2_y":       row["ofd_2_y"],
            # Distances
            "bpd_px":        round(bpd_px, 2),
            "ofd_px":        round(ofd_px, 2),
            "px_to_mm_rate": row.get("px_to_mm_rate", float("nan")),
            # Ellipse (for segmentation mask reconstruction)
            "ell_cx":        round(ell["ell_cx"], 2)        if ell else float("nan"),
            "ell_cy":        round(ell["ell_cy"], 2)        if ell else float("nan"),
            "ell_semi_ofd":  round(ell["ell_semi_ofd"], 2)  if ell else float("nan"),
            "ell_semi_bpd":  round(ell["ell_semi_bpd"], 2)  if ell else float("nan"),
            "ell_angle_deg": round(ell["ell_angle_deg"], 3)  if ell else float("nan"),
            "mask_path":     mask_path if mask_saved else "",
        }
        rows.append(rec)

    return pd.DataFrame(rows)


# ── 6. Summary ────────────────────────────────────────────────────────────
def print_summary(ann: pd.DataFrame, total_images: int) -> None:
    print("\n" + "=" * 60)
    print("Phase 2 — Annotation fetch summary")
    print("=" * 60)
    print(f"  Total trans-thalamic images : {total_images}")
    print(f"  Annotated                   : {len(ann)}")
    print(f"  Coverage                    : {100*len(ann)/total_images:.1f}%")
    print(f"  Masks generated             : {ann['mask_path'].astype(bool).sum()}")
    print(f"  BPD (px)  mean ± std        : {ann['bpd_px'].mean():.1f} ± {ann['bpd_px'].std():.1f}")
    print(f"  OFD (px)  mean ± std        : {ann['ofd_px'].mean():.1f} ± {ann['ofd_px'].std():.1f}")
    split_counts = ann["split"].value_counts().to_dict()
    print(f"  Split counts                : {split_counts}")
    print(f"\n  Saved → {ANN_OUT}")
    print(f"  Masks → {MASKS_DIR}/")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 2: Fetch BPD/Head Annotations — BPD Measurement")
    print("=" * 60)

    # Load user's manifest
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}\nRun prepare_data.py first.")
    manifest = pd.read_csv(MANIFEST)
    print(f"[manifest] {len(manifest)} trans-thalamic images loaded.")

    # Download FP annotations
    fp_df = fetch_fp_annotations(FP_HEAD_URL)

    # Match
    merged = match_to_manifest(fp_df, manifest)

    # Build annotation rows + generate masks
    ann = build_annotations(merged)

    if ann.empty:
        print("[WARNING] No annotations matched. Check image filenames and FP CSV.")
        return

    # Save
    ann.to_csv(ANN_OUT, index=False)
    print_summary(ann, total_images=len(manifest))


if __name__ == "__main__":
    main()
