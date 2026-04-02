# Automatic Fetal BPD Measurement from Ultrasound

> Adapts the landmark heatmap regression method of **Collins et al. (2026)** — originally developed for fetal femur length — to automatically measure **Biparietal Diameter (BPD)** from standard trans-thalamic fetal ultrasound planes.

---

## Overview

BPD (Biparietal Diameter) is one of the four standard fetal biometry measurements used to estimate gestational age and detect abnormal development. Existing automated approaches rely on full head segmentation, which requires ~20 hours of expert annotation. This project reframes BPD measurement as a **2-point landmark detection problem**, requiring only ~4 hours of annotation.

The pipeline detects two outer parietal bone endpoints and measures the Euclidean distance between them. A third "center point" (the midpoint of the two endpoints, added for free) is used by a greedy algorithm to suppress false-positive detections.

**Architecture:** U-Net with a ResNeXt101-32x8d encoder (95.8M parameters), trained to output 3 Gaussian heatmap channels: left endpoint, right endpoint, and center point.

---

## Results

| Method                        | Loc. Error (mm) | Meas. Error (mm)  |
|-------------------------------|:---------------:|:-----------------:|
| SegNet baseline               | 1.19 ± 2.15     | 1.04 ± 2.88       |
| **U-Net, 3 keypoints (ours)** | **0.70 ± 0.43** | **0.56 ± 0.51**   |

*(Numbers will be updated after training on this dataset.)*

---

## Project Structure

```
pbp_measurement/
├── prepare_data.py        # Phase 1: download & filter trans-thalamic images
├── fetch_annotations.py   # Phase 2: fetch BPD/OFD keypoints + generate masks
├── train.py               # Phase 3: train U-Net heatmap regression model
├── evaluate.py            # Phase 3: run all evaluation metrics + plots
├── predict.py             # Phase 3: single-image inference + visualisation
│
├── src/
│   ├── dataset.py         # BPDDataset — image loading, augmentation, heatmap targets
│   ├── model.py           # U-Net factory (segmentation-models-pytorch)
│   ├── heatmap.py         # Gaussian heatmap generation + NMS peak extraction
│   ├── postprocess.py     # Greedy endpoint selection (Algorithm 1, Collins 2026)
│   └── metrics.py         # Localization error, success rate, Bland-Altman, PR/AP
│
├── data/
│   ├── images/
│   │   └── trans_thalamic/    # 1,638 PNG ultrasound images (not tracked in git)
│   ├── masks/                 # Binary head ellipse masks (not tracked in git)
│   ├── trans_thalamic_manifest.csv
│   └── annotations.csv        # BPD/OFD keypoints for 1,637 images
│
├── checkpoints/               # Saved model checkpoints (not tracked in git)
├── results/                   # Evaluation plots and results CSV (not tracked in git)
│
├── environment.yml            # Conda environment (CUDA 11.8)
├── requirements.txt           # pip requirements
└── prd_bpd_measurement.docx   # Project requirements document
```

---

## Dataset

| Split | Images |
|-------|-------:|
| Train | 1,005  |
| Val   |   243  |
| Test  |   389  |
| **Total** | **1,637** |

Source: [FETAL_PLANES_DB](https://zenodo.org/records/3904280) (Burgos-Artizzu et al., 2020). Annotations from [Multicentre-Fetal-Biometry](https://github.com/surgical-vision/Multicentre-Fetal-Biometry). Splits are patient-level (no data leakage).

---

## Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate pbp_measurement
```

### Option B — pip

```bash
# Install PyTorch with CUDA 11.8 first:
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Then install remaining dependencies:
pip install -r requirements.txt
```

---

## Reproducing the Pipeline

### Phase 1 — Prepare data

Downloads the FETAL_PLANES_DB dataset (~2 GB), filters trans-thalamic plane images, and creates the train/val/test split manifest.

```bash
python prepare_data.py
```

Output: `data/images/trans_thalamic/` (1,638 PNGs) and `data/trans_thalamic_manifest.csv`.

---

### Phase 2 — Fetch annotations

Downloads BPD/OFD keypoint annotations from the Multicentre-Fetal-Biometry dataset and generates elliptical head segmentation masks.

```bash
python fetch_annotations.py
```

Output: `data/annotations.csv` and `data/masks/`.

---

### Phase 3 — Train

```bash
python train.py
```

Key hyperparameters (Collins et al. 2026):

| Parameter        | Value                      |
|------------------|----------------------------|
| Encoder          | ResNeXt101-32x8d (ImageNet)|
| Input resolution | 256 × 256                  |
| Batch size       | 8                          |
| Optimizer        | RAdam, lr = 1e-3           |
| LR scheduler     | ReduceLROnPlateau (×0.25, patience 5) |
| Loss             | MSE                        |
| Epochs           | 100                        |
| Augmentation     | Flip, Rotate ±20°, Brightness/Contrast ±20% |

Checkpoints are saved to `checkpoints/best.pt` (best val loss) and `checkpoints/last.pt`.

All options:

```bash
python train.py --help
```

---

### Phase 3 — Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

Outputs (saved to `results/`):
- `test_results.csv` — per-image predictions and errors
- `bland_altman.png` — Bland-Altman agreement plot
- `success_rate.png` — success rate bar chart

Metrics reported:
- Localization error (px and mm, mean ± std, median)
- BPD measurement error (px and mm)
- Success rate at 0.5 / 1 / 2 / 3 mm
- Precision, Recall, Average Precision at 1–4 mm
- Bland-Altman statistics (limits of agreement)

---

### Phase 3 — Predict on a single image

```bash
python predict.py \
    --checkpoint checkpoints/best.pt \
    --image data/images/trans_thalamic/Patient00168_Plane3_1_of_3.png \
    --px-to-mm 0.14
```

Saves a 3-panel figure: US image with BPD line overlaid, predicted heatmaps, and heatmap overlay.

---

## Method

### Heatmap generation

Each landmark is encoded as a 2D Gaussian circle (σ = 5 px at 256×256) over the annotated pixel location. Three heatmaps are produced per image:

| Channel | Landmark   | Source          |
|---------|------------|-----------------|
| 0       | Left endpoint  | Manual annotation |
| 1       | Right endpoint | Manual annotation |
| 2       | Center point   | Automatic (midpoint) — no extra annotation needed |

### Greedy false-positive filter (Algorithm 1, Collins et al. 2026)

After NMS extracts candidate peaks from each channel, the greedy algorithm selects the (left, right) pair whose geometric midpoint is closest to the predicted center point. This eliminates false positives from confounding bone structures.

### Pixel-to-mm conversion

Each image in the dataset includes a `px_to_mm_rate` value (mm per pixel) derived from the ultrasound scale bar. BPD in mm = BPD in pixels × px_to_mm_rate.

---

## Reference

```
Collins, T., Munyaneza, G., et al. (2026).
End to end automatic measurement of fetal femur length in ultrasound images.
Surgical Data Science Department, IRCAD Institute.
```

---

## License

For research use only. Dataset subject to original source licenses:
- FETAL_PLANES_DB: [Zenodo CC BY 4.0](https://zenodo.org/records/3904280)
- Multicentre-Fetal-Biometry: [GitHub](https://github.com/surgical-vision/Multicentre-Fetal-Biometry)
