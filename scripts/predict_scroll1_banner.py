#!/usr/bin/env python3
"""Run Kintsugi Physical inference directly on canonical Scroll 1 (PHercParis4) text chunks from S3.

Streams chunks on-the-fly from s3://vesuvius-challenge-open-data/
Target: Segment 20231012184424 (The 2023 Grand Prize Banner segment)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from vesuvius_code.checkpoints import load_checkpoint, read_checkpoint_metadata
from vesuvius_code.inference import predict_region
from vesuvius_code.model_factory import build_model_from_config

S3_BASE = (
    "s3://vesuvius-challenge-open-data/PHercParis4/segments/20231012184424/"
    "surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/0/0"
)


def download_chunk(cy: int, cx: int, out_dir: Path) -> tuple[int, int, np.ndarray]:
    chunk_file = out_dir / f"chunk_{cy}_{cx}.bin"
    if not chunk_file.exists():
        url = f"{S3_BASE}/{cy}/{cx}"
        cmd = ["aws", "s3", "cp", "--no-sign-request", url, str(chunk_file)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = chunk_file.read_bytes()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((109, 128, 128))
    return cy, cx, arr


def load_scroll1_volume(
    y_chunks: range,
    x_chunks: range,
    scratch_dir: Path,
    max_workers: int = 16,
) -> np.ndarray:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    num_y = len(y_chunks)
    num_x = len(x_chunks)
    print(f"Streaming {num_y * num_x} chunks ({num_y * num_x * 1.78:.1f} MB) from Scroll 1 S3...")

    full_vol = np.zeros((109, num_y * 128, num_x * 128), dtype=np.uint8)

    tasks = [(cy, cx) for cy in y_chunks for cx in x_chunks]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_chunk, cy, cx, scratch_dir)
            for cy, cx in tasks
        ]
        for f in futures:
            cy, cx, arr = f.result()
            iy = cy - y_chunks.start
            ix = cx - x_chunks.start
            full_vol[:, iy * 128 : (iy + 1) * 128, ix * 128 : (ix + 1) * 128] = arr

    print("Volume assembled! Shape:", full_vol.shape)
    return full_vol


def main():
    parser = argparse.ArgumentParser(description="Predict Greek letters on Scroll 1 with Kintsugi Physical")
    parser.add_argument("--ckpt", default="runs/cloud_kintsugi_heavy_seed1337/best.strict.ckpt")
    parser.add_argument("--out-png", default="runs/scroll1_banner_kintsugi_prediction.png")
    parser.add_argument("--y-start", type=int, default=351, help="Starting Y chunk index")
    parser.add_argument("--x-start", type=int, default=142, help="Starting X chunk index")
    parser.add_argument("--num-chunks", type=int, default=8, help="Number of chunks along Y and X (8 = 1024 px)")
    parser.add_argument("--tta", action="store_true", default=True, help="Use 8-fold D4 dihedral TTA")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    )
    print(f"Using device: {device}")

    scratch_dir = Path("/tmp/scroll1_s3_chunks")
    y_chunks = range(args.y_start, args.y_start + args.num_chunks)
    x_chunks = range(args.x_start, args.x_start + args.num_chunks)

    vol = load_scroll1_volume(y_chunks, x_chunks, scratch_dir)

    # Load model
    ckpt_path = Path(args.ckpt)
    meta = read_checkpoint_metadata(ckpt_path)
    model = build_model_from_config(meta["run_config"]["model"]).to(device)
    load_checkpoint(ckpt_path, model=model, strict=True, map_location=device)
    model.eval()

    # The surface mesh is at slice 54. A 16-slice slab around the mesh surface:
    z_start = 54 - 8
    z_end = 54 + 8
    slab = vol[z_start:z_end, :, :]  # shape: (16, H, W)

    h, w = slab.shape[1], slab.shape[2]
    print(f"Running Kintsugi Physical inference on {h}x{w} Scroll 1 volume slab (depth={slab.shape[0]})...")

    prob = predict_region(
        model,
        slab,
        region=(0, 0, h, w),
        patch_size=(16, 128, 128),
        stride=(64, 64),
        device=device,
        batch_size=16,
        tta=args.tta,
    )

    # Render figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=140, facecolor="#0D1117")
    
    # Left: X-ray mean attenuation
    mean_xray = slab.mean(axis=0)
    p5, p99 = np.percentile(mean_xray, [5, 99])
    norm_xray = np.clip((mean_xray - p5) / max(1e-5, p99 - p5), 0, 1)
    axes[0].imshow(norm_xray, cmap="gray")
    axes[0].set_title(
        f"Scroll 1 Synchrotron X-Ray (Slab Z={z_start}..{z_end})\nCanvas y={args.y_start*128}..{(args.y_start+args.num_chunks)*128}, x={args.x_start*128}..{(args.x_start+args.num_chunks)*128}",
        color="#94A3B8",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].axis("off")

    # Right: Kintsugi Physical AI Probability Heatmap
    p_lo, p_hi = np.percentile(prob, [40, 99.5])
    norm_prob = np.clip((prob - p_lo) / max(1e-5, p_hi - p_lo), 0, 1)
    im = axes[1].imshow(norm_prob, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title(
        "Kintsugi Physical Heavy (410k params + 8-fold TTA)\nDirect Unassisted Greek Text Extraction from Scroll 1",
        color="#22C55E",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].axis("off")

    fig.suptitle(
        "Vesuvius Challenge: Scroll 1 (PHercParis4, Buried 79 AD)\nPhysics-Guided Carbon Ink Detection in Sealed Papyrus Scroll",
        fontsize=13,
        color="#F8FAFC",
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor="#0D1117")
    plt.close(fig)
    print(f"Saved Scroll 1 prediction to {out_path}")


if __name__ == "__main__":
    main()
