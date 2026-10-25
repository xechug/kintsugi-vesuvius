#!/usr/bin/env python3
"""Run Kintsugi Physical inference on VIRGIN, unread Scroll 4 (PHerc1667) segment from S3.

Scroll 4 was scanned at Diamond Light Source and fully unwrapped in 2024-2026.
Target: Segment 20260205070000-w041_2026020507_flatboi (Virgin, no public ink labels).
Streams chunks directly from s3://vesuvius-challenge-open-data/
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
import urllib.request

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

DEFAULT_SEGMENT = "20240304161941-w023_20240304161941_flatboi"


def get_level_url(segment: str) -> str:
    return (
        f"https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/"
        f"{segment}/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/0"
    )


_ZARR_META: dict = {}


def get_zarr_meta(level_url: str) -> dict:
    """Read .zarray once to learn the chunk-key separator and compressor (segments differ)."""
    if level_url not in _ZARR_META:
        import json
        with urllib.request.urlopen(f"{level_url}/.zarray", timeout=30) as resp:
            _ZARR_META[level_url] = json.loads(resp.read())
    return _ZARR_META[level_url]


def decode_chunk(raw: bytes, meta: dict) -> np.ndarray | None:
    if meta.get("compressor"):
        import numcodecs
        raw = numcodecs.get_codec(meta["compressor"]).decode(raw)
    raw = bytes(raw)
    if len(raw) != 109 * 128 * 128:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((109, 128, 128))


def download_chunk(cy: int, cx: int, out_dir: Path, http_base: str) -> tuple[int, int, np.ndarray]:
    meta = get_zarr_meta(http_base)
    sep = meta.get("dimension_separator", ".")
    chunk_file = out_dir / f"chunk_{cy}_{cx}.bin"
    if not chunk_file.exists():
        url = f"{http_base}/{sep.join(['0', str(cy), str(cx)])}"
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "KintsugiPhysical/1.0 (Vesuvius Challenge Prize Scan)"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                chunk_file.write_bytes(data)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # Sparse Zarr chunk: represents unsegmented region or empty background
                    return cy, cx, np.zeros((109, 128, 128), dtype=np.uint8)
                if attempt == 4:
                    raise RuntimeError(f"Failed to download {url}: {e}")
                time.sleep(1 + attempt)
            except Exception as e:
                if attempt == 4:
                    raise RuntimeError(f"Failed to download {url}: {e}")
                time.sleep(1 + attempt)
    if chunk_file.exists():
        arr = decode_chunk(chunk_file.read_bytes(), meta)
        if arr is not None:
            return cy, cx, arr
    return cy, cx, np.zeros((109, 128, 128), dtype=np.uint8)


def load_scroll4_volume(
    y_chunks: range,
    x_chunks: range,
    scratch_dir: Path,
    http_base: str,
    max_workers: int = 16,
) -> np.ndarray:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    num_y = len(y_chunks)
    num_x = len(x_chunks)
    print(f"Streaming {num_y * num_x} chunks ({num_y * num_x * 1.78:.1f} MB) from Scroll 4 S3 ({http_base})...")

    full_vol = np.zeros((109, num_y * 128, num_x * 128), dtype=np.uint8)

    tasks = [(cy, cx) for cy in y_chunks for cx in x_chunks]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_chunk, cy, cx, scratch_dir, http_base)
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
    parser = argparse.ArgumentParser(description="Scan virgin Scroll 4 segment with Kintsugi Physical")
    parser.add_argument("--segment", default=DEFAULT_SEGMENT, help="Scroll 4 segment name")
    parser.add_argument("--ckpt", default="runs/cloud_kintsugi_heavy_seed1337/best.strict.ckpt")
    parser.add_argument("--out-png", default="runs/scroll4_w023_hotspot1_kintsugi_prediction.png")
    parser.add_argument("--y-start", type=int, default=65, help="Starting Y chunk index")
    parser.add_argument("--x-start", type=int, default=345, help="Starting X chunk index")
    parser.add_argument("--num-chunks", type=int, default=16, help="Number of chunks along Y and X (16 = 2048 px)")
    parser.add_argument("--num-y-chunks", type=int, default=20, help="Number of chunks along Y (overrides num-chunks)")
    parser.add_argument("--num-x-chunks", type=int, default=25, help="Number of chunks along X (overrides num-chunks)")
    parser.add_argument("--tta", action="store_true", default=True, help="Use 8-fold D4 dihedral TTA")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    )
    print(f"Using device: {device}")

    num_y = args.num_y_chunks if args.num_y_chunks is not None else args.num_chunks
    num_x = args.num_x_chunks if args.num_x_chunks is not None else args.num_chunks

    scratch_dir = Path("/tmp/scroll4_s3_chunks")
    y_chunks = range(args.y_start, args.y_start + num_y)
    x_chunks = range(args.x_start, args.x_start + num_x)

    http_base = get_level_url(args.segment)
    vol = load_scroll4_volume(y_chunks, x_chunks, scratch_dir, http_base)

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
    print(f"Running Kintsugi Physical inference on {h}x{w} Scroll 4 volume slab (depth={slab.shape[0]})...")

    prob = predict_region(
        model,
        slab,
        region=(0, 0, h, w),
        patch_size=(16, 128, 128),
        stride=(64, 64),
        device=device,
        batch_size=args.batch_size,
        tta=args.tta,
    )

    # Save raw probabilities
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path = out_path.with_suffix(".npy")
    np.save(npy_path, prob)
    print(f"Saved raw probabilities to {npy_path}")

    # Process images for visualization
    mean_xray = slab.mean(axis=0)
    p5, p99 = np.percentile(mean_xray, [5, 99])
    norm_xray = np.clip((mean_xray - p5) / max(1e-5, p99 - p5), 0, 1)

    p_lo, p_hi = np.percentile(prob, [40, 99.5])
    norm_prob = np.clip((prob - p_lo) / max(1e-5, p_hi - p_lo), 0, 1)

    ink_mask = np.clip((norm_prob - 0.5) / 0.5, 0, 1)
    overlay = np.dstack([norm_xray * 0.7, norm_xray * 0.7, norm_xray * 0.7])
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + ink_mask * 0.95, 0, 1)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + ink_mask * 0.75, 0, 1)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + ink_mask * 0.15, 0, 1)

    # Save 1:1 pixel native overlay image
    from PIL import Image
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    native_overlay_path = out_path.with_name(out_path.stem + "_native_overlay.png")
    Image.fromarray(overlay_uint8).save(native_overlay_path)
    print(f"Saved 1:1 native resolution overlay to {native_overlay_path}")

    # Render figure
    is_wide = (w / max(1, h)) > 1.3
    if is_wide:
        fig_h = max(18, int(22 * (h / w) * 3 + 2))
        fig, axes = plt.subplots(3, 1, figsize=(24, fig_h), dpi=140, facecolor="#0D1117")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=140, facecolor="#0D1117")
    
    # 1. X-ray
    axes[0].imshow(norm_xray, cmap="gray")
    axes[0].set_title(
        f"1. Synchrotron X-Ray (Diamond Light Source, 78keV)\nSlab Z={z_start}..{z_end} | Y={args.y_start*128}..{(args.y_start+num_y)*128}, X={args.x_start*128}..{(args.x_start+num_x)*128}",
        color="#94A3B8",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].axis("off")

    # 2. Probability Heatmap
    axes[1].imshow(norm_prob, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title(
        f"2. Kintsugi Physical Heavy (410k params + 8-fold TTA)\nProbability Map ({h}x{w} px)",
        color="#22C55E",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].axis("off")

    # 3. Ink Stroke Composite
    axes[2].imshow(overlay)
    axes[2].set_title(
        "3. High-Contrast Ink Stroke Overlay\nReconstructed Carbon Papyrus Letterforms",
        color="#F59E0B",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].axis("off")

    fig.suptitle(
        f"Vesuvius Challenge: VIRGIN Scroll 4 (PHerc1667, Buried 79 AD)\nSegment: {args.segment} | Prime Hotspot Text Exploration",
        fontsize=13,
        color="#F8FAFC",
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(out_path, dpi=140, facecolor="#0D1117")
    plt.close(fig)
    print(f"Saved Scroll 4 prediction to {out_path}")


if __name__ == "__main__":
    main()
