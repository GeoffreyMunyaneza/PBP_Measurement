"""
Phase 3 — Training script for BPD heatmap regression.

Implements the best-performing experiment from Collins et al. 2026:
    Single U-Net (ResNeXt101-32x8d encoder) with 3 output channels predicting
    heatmaps for left BPD endpoint, right BPD endpoint, and center point.

Usage:
    python train.py [--epochs 100] [--batch-size 8] [--lr 1e-3]
                    [--output-dir checkpoints/] [--resume checkpoints/last.pt]
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.dataset import BPDDataset
from src.model import build_model, count_parameters


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train U-Net heatmap regression for fetal BPD measurement."
    )
    p.add_argument("--annotations", default="data/annotations.csv",
                   help="Path to annotations CSV (default: data/annotations.csv)")
    p.add_argument("--root-dir", default=".",
                   help="Project root for resolving image paths")
    p.add_argument("--output-dir", default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs (default: 100)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size (default: 8)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Initial learning rate (default: 1e-3)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker count (default: 4)")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--encoder", default="resnext101_32x8d",
                   help="SMP encoder name (default: resnext101_32x8d)")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def build_dataloaders(
    annotations_path: str,
    root_dir: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from the annotations CSV."""
    ann = pd.read_csv(annotations_path)

    train_ann = ann[ann["split"] == "train"].copy()
    val_ann   = ann[ann["split"] == "val"].copy()

    print(f"  Train: {len(train_ann)} images")
    print(f"  Val  : {len(val_ann)} images")

    train_ds = BPDDataset(train_ann, root_dir=root_dir, augment=True)
    val_ds   = BPDDataset(val_ann,   root_dir=root_dir, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    """Run one full epoch. Returns mean loss."""
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            images   = batch["image"].to(device, non_blocking=True)
            heatmaps = batch["heatmap"].to(device, non_blocking=True)

            preds = model(images)
            # Apply sigmoid before MSE so predicted values are in [0,1],
            # matching the Gaussian heatmap targets.
            loss  = criterion(torch.sigmoid(preds), heatmaps)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches if n_batches else float("nan")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint and return (start_epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"  Resumed from epoch {ckpt['epoch']}, best val loss {ckpt['best_val_loss']:.6f}")
    return ckpt["epoch"] + 1, ckpt["best_val_loss"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    train_loader, val_loader = build_dataloaders(
        args.annotations, args.root_dir, args.batch_size, args.num_workers
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nBuilding model ...")
    model = build_model(encoder_name=args.encoder).to(device)
    print(f"  Parameters : {count_parameters(model) / 1e6:.2f} M")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = RAdam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.25,
        patience=5,
    )
    criterion = nn.MSELoss()

    # ── Resume (optional) ─────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = math.inf

    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from {args.resume} ...")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs (batch={args.batch_size}, lr={args.lr})\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>8}  {'Time':>6}")
    print("-" * 50)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch + 1:5d}  {train_loss:10.6f}  {val_loss:10.6f}  "
            f"{current_lr:8.2e}  {elapsed:5.1f}s"
        )

        # Save last checkpoint (always)
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, scheduler,
            epoch, val_loss, best_val_loss
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, scheduler,
                epoch, val_loss, best_val_loss
            )
            print(f"           >> New best val loss: {best_val_loss:.6f}  (saved best.pt)")

    print("\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.6f}")
    print(f"Checkpoints   : {output_dir}/best.pt  |  {output_dir}/last.pt")


if __name__ == "__main__":
    main()
