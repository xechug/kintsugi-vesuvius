#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import os
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
from vesuvius_code.metrics import (
    average_precision,
    binary_segmentation_metrics,
    threshold_sweep_metrics,
)
from vesuvius_code.model_factory import build_model_from_config
from vesuvius_code.splits import SpatialSplitManifest


def load_strict_model(ckpt_path: Path, device: torch.device):
    meta = read_checkpoint_metadata(ckpt_path)
    model = build_model_from_config(meta["run_config"]["model"]).to(device)
    load_checkpoint(ckpt_path, model=model, strict=True, map_location=device)
    model.eval()
    return model


def main():
    default_base = (
        Path("/workspace/data/public/kaggle2023_fragment3")
        if Path("/workspace/data/public/kaggle2023_fragment3").exists()
        else REPO_DIR / "data/public/kaggle2023_fragment3"
    )
    parser = argparse.ArgumentParser(
        description="Evaluate Gate A models on Fragment 3 validation+test splits and render macroscopic diagnostic figure"
    )
    parser.add_argument("--zarr", default=str(default_base / "volume.zarr"))
    parser.add_argument("--mask", default=str(default_base / "mask.tif"))
    parser.add_argument("--ink", default=str(default_base / "inklabels.tif"))
    parser.add_argument("--split", default=str(default_base / "split.json"))
    parser.add_argument("--residual-ckpt", required=True)
    parser.add_argument("--axial-ckpt", required=True)
    parser.add_argument("--physical-ckpt", required=True)
    parser.add_argument("--output-png", default="fragment3_gate_a_comparison.png")
    parser.add_argument("--metrics-json", default="gate_a_fragment3_metrics.json")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--tta", action="store_true", help="Apply 8-fold D4 dihedral Test-Time Augmentation")
    args = parser.parse_args()

    device = torch.device(args.device)
    vol = open_volume_array(args.zarr)
    ink_raster = open_raster(args.ink)
    mask_raster = open_raster(args.mask)
    split = SpatialSplitManifest.load(args.split)
    val_regions = split.regions("validation")
    if not val_regions:
        raise ValueError("No validation regions found in split manifest")

    best_region = max(
        val_regions,
        key=lambda r: float((np.asarray(ink_raster[r.y0 : r.y1, r.x0 : r.x1]) > 0).sum()),
    )
    target_coords = (best_region.y0, best_region.x0, best_region.y1, best_region.x1)

    models_spec = [
        ("residual_unet", "1. Residual U-Net (No Physics)", Path(args.residual_ckpt), "#EF4444"),
        ("kintsugi_axial", "2. Kintsugi Axial (No Physics)", Path(args.axial_ckpt), "#F59E0B"),
        ("kintsugi_physical", "3. Kintsugi Physical (Full Physics 1+2+3)", Path(args.physical_ckpt), "#22C55E"),
    ]

    summary = {}
    if Path(args.metrics_json).exists():
        try:
            summary = json.loads(Path(args.metrics_json).read_text())
        except Exception:
            summary = {}

    panel_data = []

    for key, title, ckpt_path, color in models_spec:
        model = load_strict_model(ckpt_path, device)
        print(f"Loaded {key} on {device}")
        
        # Check if we already have precomputed metrics for this model
        has_metrics = key in summary and "validation" in summary[key] and "test" in summary[key]
        
        if has_metrics:
            print(f"Reusing precomputed split metrics for {key}")
            prob = predict_region(
                model,
                vol,
                region=target_coords,
                patch_size=(16, 128, 128),
                stride=(64, 64),
                device=device,
                batch_size=16,
                tta=args.tta,
            )
            gt = (
                np.asarray(ink_raster[target_coords[0] : target_coords[2], target_coords[1] : target_coords[3]], dtype=np.float32) > 0
            ).astype(np.float32)
            valid = np.asarray(mask_raster[target_coords[0] : target_coords[2], target_coords[1] : target_coords[3]]) > 0
            panel_data.append((title, prob, gt, valid, color, summary[key]["validation"], summary[key]["test"]))
        else:
            summary[key] = {}
            panel_item = None
            for split_name in ["validation", "test"]:
                all_probs, all_gts, all_valids = [], [], []
                for reg in split.regions(split_name):
                    r = (reg.y0, reg.x0, reg.y1, reg.x1)
                    prob = predict_region(
                        model,
                        vol,
                        region=r,
                        patch_size=(16, 128, 128),
                        stride=(64, 64),
                        device=device,
                        batch_size=16,
                        tta=args.tta,
                    )
                    gt = (
                        np.asarray(ink_raster[reg.y0 : reg.y1, reg.x0 : reg.x1], dtype=np.float32) > 0
                    ).astype(np.float32)
                    valid = np.asarray(mask_raster[reg.y0 : reg.y1, reg.x0 : reg.x1]) > 0
                    all_probs.append(torch.from_numpy(prob.reshape(-1)))
                    all_gts.append(torch.from_numpy(gt.reshape(-1)))
                    all_valids.append(torch.from_numpy(valid.reshape(-1)))
                    if split_name == "validation" and r == target_coords:
                        panel_item = (title, prob, gt, valid, color)
                p_cat = torch.cat(all_probs)
                g_cat = torch.cat(all_gts)
                v_cat = torch.cat(all_valids)
                ap = average_precision(p_cat, g_cat, valid_mask=v_cat)
                sweep = threshold_sweep_metrics(p_cat, g_cat, valid_mask=v_cat)
                m05 = binary_segmentation_metrics(p_cat, g_cat, threshold=0.5, valid_mask=v_cat)
                prev = ap["positive_pixels"] / max(1, ap["valid_pixels"])
                summary[key][split_name] = {
                    "ap": round(ap["average_precision"], 5),
                    "prevalence": round(prev, 5),
                    "lift_over_chance": round(ap["average_precision"] - prev, 5),
                    "best_f1": round(sweep["f1"], 5),
                    "best_threshold": round(sweep["threshold"], 2),
                    "fp_per_mpix_at_05": round(m05["fp_per_mpix"], 1),
                }
            panel_data.append((*panel_item, summary[key]["validation"], summary[key]["test"]))

    Path(args.metrics_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    y0, x0, y1, x1 = target_coords
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5), dpi=140, facecolor="#0D1117")
    gt0 = panel_data[0][2]
    val0 = panel_data[0][3]
    axes[0, 0].imshow(gt0 * val0, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(
        f"Ground Truth Infrared Ink (Fragment 3)\ny={y0}..{y1}, x={x0}..{x1}\nPrevalence = {gt0[val0].mean():.3%}",
        color="#38BDF8",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0, 0].axis("off")

    raw_crop = np.asarray(vol[:, y0:y1, x0:x1], dtype=np.float32).mean(axis=0)
    p5, p99 = np.percentile(raw_crop[val0], [5, 99])
    axes[1, 0].imshow(np.clip((raw_crop - p5) / max(1e-5, p99 - p5), 0, 1) * val0, cmap="gray")
    axes[1, 0].contour(gt0, levels=[0.5], colors=["#38BDF8"], linewidths=0.9)
    axes[1, 0].set_title(
        f"Raw Synchrotron X-Ray (Mean Z)\nCyan Contour = Ground-Truth Ink Location",
        color="#94A3B8",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1, 0].axis("off")

    for idx, (title, prob, gt, valid, color, m_val, m_test) in enumerate(panel_data, start=1):
        p_lo, p_hi = np.percentile(prob[valid], [50, 99.5])
        norm_prob = np.clip((prob - p_lo) / max(1e-5, p_hi - p_lo), 0.0, 1.0) * valid
        # TOP ROW: RAW AI PROBABILITY HEATMAP - NO GROUND TRUTH CONTOUR OVERLAY
        axes[0, idx].imshow(norm_prob, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0, idx].set_title(
            f"{title}\nVal AP={m_val['ap']:.4f} (Lift {m_val['lift_over_chance']:+.4f} / x{m_val['lift_over_chance']/0.00952:.2f})\n[Raw Prediction — No Ground Truth Overlay]",
            color=color,
            fontsize=10.0,
            fontweight="bold",
        )
        axes[0, idx].axis("off")

        # BOTTOM ROW: DETAILED CLASSIFICATION WITH CONTOUR
        thresh = np.percentile(prob[valid], 85.0)
        pred_bin = (prob >= thresh) & valid
        rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
        tp = pred_bin & (gt > 0.5)
        fp = pred_bin & (gt <= 0.5)
        fn = (~pred_bin) & (gt > 0.5) & valid
        rgb[tp] = [0.13, 0.77, 0.37]  # Green = True Ink Hit
        rgb[fp] = [0.94, 0.27, 0.27]  # Red = False Alarm on Fiber
        rgb[fn] = [0.15, 0.35, 0.55]  # Dim Blue = Missed Ink
        axes[1, idx].imshow(rgb)
        axes[1, idx].contour(gt, levels=[0.5], colors=["#38BDF8"], linewidths=0.6, alpha=0.6)
        axes[1, idx].set_title(
            f"Hits (Green) vs Fiber False Alarms (Red)\nCyan Contour = GT Ink | Val FP@0.5: {m_val['fp_per_mpix_at_05']:,.0f}/Mpix",
            color=color,
            fontsize=9.5,
            fontweight="bold",
        )
        axes[1, idx].axis("off")

    fig.suptitle(
        "Vesuvius Challenge: Gate A Held-Out Fragment 3 Diagnostic Evaluation\nTop Row: Raw AI Probability Heatmaps (Unassisted, No GT Contours) | Bottom Row: Physical Artifact Classification",
        fontsize=13,
        color="#F8FAFC",
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output_png, dpi=140, facecolor="#0D1117")
    plt.close(fig)
    print(f"Saved macroscopic comparison to {args.output_png}")


if __name__ == "__main__":
    main()
