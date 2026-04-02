"""
Phase 1: Data Preparation for BPD Measurement Project
Downloads FETAL_PLANES_DB, extracts trans-thalamic images,
and produces a train/val/test split CSV.
"""

import os
import sys
import zipfile
sys.stdout.reconfigure(encoding='utf-8')
import shutil
import hashlib
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images" / "trans_thalamic"
ZIP_PATH   = RAW_DIR / "FETAL_PLANES_ZENODO.zip"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_URL = (
    "https://zenodo.org/records/3904280/files/"
    "FETAL_PLANES_ZENODO.zip?download=1"
)

# ── 1. Download ────────────────────────────────────────────────────────────
def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[download] Already exists: {dest.name} — skipping.")
        return
    print(f"[download] Fetching {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[download] Saved to {dest}")


# ── 2. Extract ─────────────────────────────────────────────────────────────
def extract(zip_path: Path, dest: Path) -> Path:
    """Extract zip and return the directory that contains the CSV."""
    marker = dest / ".extracted"
    if not marker.exists():
        print(f"[extract] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        marker.touch()
        print(f"[extract] Done -> {dest}")
    else:
        print("[extract] Already extracted -- skipping.")
    # Return the folder that actually contains the CSV (may be dest itself)
    csvs = list(dest.rglob("*.csv"))
    if csvs:
        return csvs[0].parent
    return dest


# ── 3. Explore structure ───────────────────────────────────────────────────
def find_csv(root: Path) -> Path:
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {root}")
    # prefer the main metadata file
    csvs.sort(key=lambda p: len(p.parts))
    print(f"[explore] CSV files found: {[str(c.relative_to(root)) for c in csvs]}")
    return csvs[0]


def find_image_root(root: Path) -> Path:
    """Return the directory that contains the PNG images."""
    # look for any .png
    sample = next(root.rglob("*.png"), None)
    if sample is None:
        raise FileNotFoundError("No PNG images found in extracted archive.")
    return sample.parent


# ── 4. Parse metadata & filter trans-thalamic ─────────────────────────────
def load_metadata(csv_path: Path, img_root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    print(f"[metadata] Columns: {list(df.columns)}")
    print(f"[metadata] Shape: {df.shape}")
    print(df.head(3).to_string())
    return df


def filter_trans_thalamic(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only trans-thalamic (TT) brain images."""
    # column names may vary — find them case-insensitively
    col_map = {c.lower(): c for c in df.columns}

    # plane column
    plane_col = next(
        (col_map[k] for k in col_map if "plane" in k or "class" in k), None
    )
    brain_plane_col = next(
        (col_map[k] for k in col_map if "brain" in k), None
    )

    print(f"[filter] Plane col: {plane_col}, Brain sub-plane col: {brain_plane_col}")

    if plane_col is None:
        raise ValueError("Cannot identify the plane/class column in metadata CSV.")

    # keep brain images
    brain_mask = df[plane_col].str.lower().str.contains("brain", na=False)
    df_brain = df[brain_mask].copy()
    print(f"[filter] Brain images: {len(df_brain)}")

    if brain_plane_col is not None:
        tt_mask = df_brain[brain_plane_col].str.lower().str.contains(
            "trans.thalamic|thalamic|tt", na=False, regex=True
        )
        df_tt = df_brain[tt_mask].copy()
    else:
        # fall back: look in filename
        df_tt = df_brain[
            df_brain.apply(
                lambda r: "thalamic" in str(r).lower() or "_tt" in str(r).lower(),
                axis=1,
            )
        ].copy()

    print(f"[filter] Trans-thalamic images: {len(df_tt)}")
    return df_tt


# ── 5. Copy images to working directory ───────────────────────────────────
def copy_images(df: pd.DataFrame, img_root: Path, dest: Path) -> pd.DataFrame:
    """Copy TT images into data/images/trans_thalamic/ and record actual paths."""
    filename_col = next(
        (c for c in df.columns if "filename" in c.lower() or "image" in c.lower()),
        df.columns[0],
    )
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        fname = str(row[filename_col]).strip()
        # add .png if missing
        if not fname.lower().endswith(".png"):
            fname += ".png"
        src = img_root / fname
        if not src.exists():
            # try recursive search
            candidates = list(img_root.parent.rglob(fname))
            src = candidates[0] if candidates else src
        if src.exists():
            dst = dest / fname
            if not dst.exists():
                shutil.copy2(src, dst)
            paths.append(str(Path("data/images/trans_thalamic") / fname))
        else:
            paths.append(None)
            missing += 1
    df = df.copy()
    df["image_path"] = paths
    if missing:
        print(f"[copy] WARNING: {missing} images not found.")
    df = df[df["image_path"].notna()].reset_index(drop=True)
    print(f"[copy] {len(df)} images copied.")
    return df


# ── 6. Train / val / test split ───────────────────────────────────────────
def split_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """60/15/25 patient-level split."""
    # identify patient column
    patient_col = next(
        (c for c in df.columns if "patient" in c.lower()), None
    )

    rng = np.random.default_rng(seed)

    if patient_col:
        patients = df[patient_col].unique()
        rng.shuffle(patients)
        n = len(patients)
        n_train = int(n * 0.60)
        n_val   = int(n * 0.15)
        train_pts = set(patients[:n_train])
        val_pts   = set(patients[n_train : n_train + n_val])
        def assign(pid):
            if pid in train_pts:
                return "train"
            if pid in val_pts:
                return "val"
            return "test"
        df["split"] = df[patient_col].map(assign)
    else:
        # fall back to random row-level split (no patient info)
        print("[split] WARNING: no patient column found — using random row split.")
        n = len(df)
        idx = rng.permutation(n)
        n_train = int(n * 0.60)
        n_val   = int(n * 0.15)
        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        split_arr = [""] * n
        for i, s in zip(idx, splits):
            split_arr[i] = s
        df["split"] = split_arr

    counts = df["split"].value_counts()
    print(f"[split] train={counts.get('train',0)}  val={counts.get('val',0)}  test={counts.get('test',0)}")
    return df


# ── 7. Pixel-size metadata ────────────────────────────────────────────────
def add_pixel_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    FETAL_PLANES_DB does not include pixel-size metadata.
    We mark it as NaN for now; it will be filled during annotation
    from DICOM headers or scale-bar extraction.
    """
    if "pixel_size_mm" not in df.columns:
        df["pixel_size_mm"] = float("nan")
        print(
            "[pixel_size] No pixel-size info in this dataset. "
            "Column 'pixel_size_mm' added as NaN — to be filled from DICOM/scale bars."
        )
    return df


# ── 8. Save deliverable CSV ───────────────────────────────────────────────
def save_manifest(df: pd.DataFrame) -> Path:
    # select and rename key columns
    col_map = {c.lower(): c for c in df.columns}

    keep = {"image_path": "image_path", "split": "split", "pixel_size_mm": "pixel_size_mm"}

    # patient id
    for key in col_map:
        if "patient" in key:
            keep[col_map[key]] = "patient_id"
            break

    # machine / operator if present
    for key in col_map:
        if "machine" in key or "us_machine" in key:
            keep[col_map[key]] = "us_machine"
        if "operator" in key:
            keep[col_map[key]] = "operator"

    rename = {v: keep[v] for v in keep if v in df.columns}
    out = df[[c for c in rename]].rename(columns=rename)

    out_path = DATA_DIR / "trans_thalamic_manifest.csv"
    out.to_csv(out_path, index=False)
    print(f"[manifest] Saved → {out_path}  ({len(out)} rows)")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 1: Data Preparation — BPD Measurement")
    print("=" * 60)

    download(ZENODO_URL, ZIP_PATH)

    extract_root = RAW_DIR / "extracted"
    extract_root.mkdir(exist_ok=True)
    dataset_root = extract(ZIP_PATH, extract_root)

    # find metadata CSV and image directory
    csv_path = find_csv(dataset_root)
    img_root = find_image_root(dataset_root)
    print(f"[explore] Image root: {img_root.relative_to(dataset_root)}")

    df = load_metadata(csv_path, img_root)
    df_tt = filter_trans_thalamic(df)

    if len(df_tt) < 100:
        print(
            f"[WARNING] Only {len(df_tt)} trans-thalamic images found. "
            "Check the CSV structure above."
        )

    df_tt = copy_images(df_tt, img_root, IMAGES_DIR)
    df_tt = split_dataset(df_tt)
    df_tt = add_pixel_size(df_tt)
    manifest = save_manifest(df_tt)

    print("\n" + "=" * 60)
    print("Phase 1 complete.")
    print(f"  Images  : {IMAGES_DIR}")
    print(f"  Manifest: {manifest}")
    print("=" * 60)


if __name__ == "__main__":
    main()
