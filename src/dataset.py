"""
PyTorch Dataset for BPD heatmap regression.

Each sample returns:
    image   – (3, IMG_SIZE, IMG_SIZE) float32 tensor, ImageNet-normalised.
              Grayscale US image replicated to 3 channels.
    heatmap – (3, IMG_SIZE, IMG_SIZE) float32 tensor.
              Channel 0 = left BPD endpoint, 1 = right, 2 = center.
    meta    – dict with original coordinates, scale, split, image path.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.heatmap import make_target_heatmaps

# ── Constants ─────────────────────────────────────────────────────────────────

IMG_SIZE = 256          # spatial resolution fed to U-Net (pixels)
HEATMAP_SIGMA = 5.0     # Gaussian sigma in heatmap-space pixels
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Augmentation pipelines ────────────────────────────────────────────────────

def _train_transforms() -> A.Compose:
    """
    Augmentation pipeline for training images with keypoint tracking.
    Mirrors Collins et al. 2026: random rotation, brightness/contrast, flips.
    """
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.0),          # not standard for head scans
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=True,
            angle_in_degrees=True,
        ),
    )


def _val_transforms() -> A.Compose:
    """Validation / test pipeline — resize and normalise only."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=True,
            angle_in_degrees=True,
        ),
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class BPDDataset(Dataset):
    """
    Dataset that loads trans-thalamic fetal US images and produces
    3-channel Gaussian heatmap targets for BPD landmark regression.

    Args:
        annotations : DataFrame with columns image_path, x_left, y_left,
                      x_right, y_right, px_to_mm_rate (and optionally split).
        root_dir    : Project root; image_path is relative to this.
        augment     : If True, apply training augmentations.
    """

    def __init__(
        self,
        annotations: pd.DataFrame,
        root_dir: str | Path,
        augment: bool = False,
    ) -> None:
        self.annotations = annotations.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transforms = _train_transforms() if augment else _val_transforms()

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        row = self.annotations.iloc[idx]

        # ── Load image ─────────────────────────────────────────────────────
        img_path = self.root_dir / row["image_path"]
        image = np.array(Image.open(img_path).convert("L"))  # grayscale → (H, W)
        orig_h, orig_w = image.shape

        # Replicate single channel → 3 channels for ImageNet-pretrained encoder
        image_rgb = np.stack([image, image, image], axis=-1)  # (H, W, 3)

        # ── Keypoints in original image space ──────────────────────────────
        x_left, y_left   = float(row["x_left"]),  float(row["y_left"])
        x_right, y_right = float(row["x_right"]), float(row["y_right"])
        x_center = (x_left + x_right) / 2.0
        y_center = (y_left + y_right) / 2.0

        keypoints = [
            (x_left,   y_left),    # index 0 → left
            (x_right,  y_right),   # index 1 → right
            (x_center, y_center),  # index 2 → center
        ]

        # ── Apply transforms ───────────────────────────────────────────────
        transformed = self.transforms(image=image_rgb, keypoints=keypoints)
        image_tensor: torch.Tensor = transformed["image"]  # (3, H, W)
        aug_kps: list[tuple] = transformed["keypoints"]

        # If any keypoint was lost (went out of bounds after augmentation),
        # retry with the original image without augmentation.
        # In practice this rarely happens with the chosen augmentation range.
        if len(aug_kps) < 3:
            val_tfm = _val_transforms()
            transformed = val_tfm(image=image_rgb, keypoints=keypoints)
            image_tensor = transformed["image"]
            aug_kps = transformed["keypoints"]

        aug_xl, aug_yl = aug_kps[0][0], aug_kps[0][1]
        aug_xr, aug_yr = aug_kps[1][0], aug_kps[1][1]

        # ── Build heatmap target ───────────────────────────────────────────
        heatmap = make_target_heatmaps(
            height=IMG_SIZE,
            width=IMG_SIZE,
            x_left=aug_xl,
            y_left=aug_yl,
            x_right=aug_xr,
            y_right=aug_yr,
            sigma=HEATMAP_SIGMA,
        )
        heatmap_tensor = torch.from_numpy(heatmap)  # (3, H, W)

        # ── Scale factors (original → heatmap space) ───────────────────────
        scale_x = IMG_SIZE / orig_w
        scale_y = IMG_SIZE / orig_h

        # Ground-truth BPD in original pixel space
        gt_bpd_px = math.hypot(x_right - x_left, y_right - y_left)

        px_to_mm = float(row.get("px_to_mm_rate", float("nan")))
        if math.isnan(px_to_mm):
            px_to_mm = None

        meta = {
            "image_path": str(row["image_path"]),
            "orig_w": orig_w,
            "orig_h": orig_h,
            "scale_x": scale_x,
            "scale_y": scale_y,
            # Ground-truth keypoints in original image space
            "gt_x_left":   x_left,
            "gt_y_left":   y_left,
            "gt_x_right":  x_right,
            "gt_y_right":  y_right,
            "gt_bpd_px":   gt_bpd_px,
            "px_to_mm":    px_to_mm,
            "split":       str(row.get("split", "")),
        }

        return {
            "image":   image_tensor,    # (3, IMG_SIZE, IMG_SIZE) float32
            "heatmap": heatmap_tensor,  # (3, IMG_SIZE, IMG_SIZE) float32
            "meta":    meta,
        }
