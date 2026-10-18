#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from vesuvius_code.checkpoints import load_checkpoint, read_checkpoint_metadata
from vesuvius_code.data_contract import open_raster
from vesuvius_code.dataset import open_volume_array
from vesuvius_code.inference import predict_region
from vesuvius_code.metrics import average_precision, threshold_sweep_metrics
from vesuvius_code.model_factory import build_model_from_config
from vesuvius_code.splits import SpatialSplitManifest


def load_strict_model(ckpt_path: Path, device: torch.device):
    meta = read_checkpoint_metadata(ckpt_path)
    model = build_model_from_config(meta["run_config"]["model"]).to(device)
    load_checkpoint(ckpt_path, model=model, strict=True, map_location=device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Render side-by-side macroscopic Greek letter predictions on Fragment 3 validation regions"
    )
    parser.add_argument("--zarr", default="/workspace/data/public/kaggle2023_fragment3/volume.zarr")
    parser.add_argument("--mask", default="/workspace/data/public/kaggle2023_fragment3/mask.tif")
    parser.add_argument("--ink", default="/workspace/data/public/kaggle2023_fragment3/inklabels.tif")
    parser.add_argument("--split", default="/workspace/data/public/kaggle2023_fragment3/split.json")
    parser.add_argument("--residual-ckpt", required=True)
    parser.add_argument("--axial-ckpt", required=True)
    parser.add_argument("--physical-ckpt", required=True)
    parser.add_argument("--output-png", default="fragment3_gate_a_comparison.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    vol = open_volume_array(args.zarr)
    ink_raster = open_raster(args.ink)
    mask_raster = open_raster(args.mask)
    split = SpatialSplitManifest.load(args.split)
    val_regions = split.regions("validation")
    if not val_regions:
        raise ValueError("No validation regions found in split manifest")

    # Pick validation region with highest positive pixel count (clearest Greek letters)
    best_region = max(
        val_regions,
        key=lambda r: float((np.asarray(ink_raster[r.y0 : r.y1, r.x0 : r.x1]) > 0).sum()),
    )
    region = (best_region.y0, best_region.x0, best_region.y1, best_region.x1)
    y0, x0, y1, x1 = region

    gt = (np.asarray(ink_raster[y0:y1, x0:x1], dtype=np.float32) > 0).astype(np.float32)
    valid = np.asarray(mask_raster[y0:y1, x0:x1]) > 0
    gt_t = torch.from_numpy(gt)
    valid_t = torch.from_numpy(valid)

    models = [
        ("1. Residual U-Net (Sin Física)", Path(args.residual_ckpt), "#EF4444"),
        ("2. Kintsugi Axial (Sin Física)", Path(args.axial_ckpt), "#F59E0B"),
        ("3. Kintsugi Physical (Con Física)", Path(args.physical_ckpt), "#22C55E"),
    ]

    preds = []
    for label, ckpt_path, color in models:
        model = load_strict_model(ckpt_path, device)
        prob = predict_region(
            model,
            vol,
            region=region,
            patch_size=(16, 128, 128),
            stride=(64, 64),
            device=device,
            batch_size=4,
        )
        prob_t = torch.from_numpy(prob)
        ap = average_precision(prob_t, gt_t, valid_mask=valid_t)["average_precision"]
        sweep = threshold_sweep_metrics(prob_t, gt_t, valid_mask=valid_t)
        preds.append((label, prob * valid, ap, sweep["f1"], sweep["fp_per_mpix"], color))

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=140, facecolor="#0D1117")
    axes[0].imshow(gt * valid, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title(
        f"Ground Truth (Fragmento 3)\ny={y0}..{y1}, x={x0}..{x1}\nPrevalencia={gt[valid].mean():.4f}",
        color="#38BDF8",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].axis("off")

    for ax, (label, prob_img, ap, f1, fp_mpix, color) in zip(axes[1:], preds):
        ax.imshow(prob_img, cmap="inferno", vmin=0.0, vmax=1.0)
        ax.contour(gt, levels=[0.5], colors=["#38BDF8"], linewidths=0.7, alpha=0.65)
        ax.set_title(
            f"{label}\nAP={ap:.4f} | F1={f1:.4f} | FP/Mpix={fp_mpix:,.0f}",
            color=color,
            fontsize=11,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(args.output_png, dpi=140, facecolor="#0D1117")
    plt.close(fig)
    print(f"Saved macroscopic comparison to {args.output_png}")


if __name__ == "__main__":
    main()
