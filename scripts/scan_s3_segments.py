#!/usr/bin/env python3
"""Stream-scan candidate scroll segments from s3://vesuvius-challenge-open-data/ one by one.

Downloads a single candidate segment volume from S3 into a temporary scratch folder,
runs KintsugiPhysical sliding-window inference, saves the lightweight PNG preview
(and optional probability TIFF), and immediately deletes the raw 3D volume so
hundreds of scroll segments can be scanned with < 20 GB of free disk space.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from scripts.public_data_inventory import (
    PUBLIC_BUCKET,
    aws_download_command,
    inventory_prefix,
    unsigned_aws_environment,
)
from vesuvius_code.dataset import open_volume_array
from vesuvius_code.inference import (
    load_kintsugi_model,
    predict_region,
    save_probability_preview,
    save_probability_tiff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download-scan-delete S3 scroll segments one at a time."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained KintsugiPhysical checkpoint (.pt)",
    )
    parser.add_argument(
        "--s3-prefix",
        required=True,
        help="S3 prefix under s3://vesuvius-challenge-open-data/ (e.g. PHercParis4/segments/)",
    )
    parser.add_argument(
        "--scratch-dir",
        default="/workspace/scratch_s3",
        help="Temporary directory for single-segment download (cleaned after each scan)",
    )
    parser.add_argument(
        "--out-dir",
        default="/workspace/scans_out",
        help="Output directory for lightweight PNG previews",
    )
    parser.add_argument(
        "--max-segment-gb",
        type=float,
        default=8.0,
        help="Skip any single object larger than this limit in GB",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of candidate volumes to scan in this batch",
    )
    parser.add_argument(
        "--center-size",
        type=int,
        default=1024,
        help="Center crop size (Y, X) to scan per segment",
    )
    parser.add_argument(
        "--save-tiff",
        action="store_true",
        help="Also keep the float32 probability TIFF alongside the PNG preview",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, metadata = load_kintsugi_model(args.checkpoint, device=device)
    saved_patch = tuple(
        int(v) for v in metadata["run_config"]["data"]["patch_size"]
    )
    stride = (max(1, saved_patch[1] // 2), max(1, saved_patch[2] // 2))

    objects = inventory_prefix(args.s3_prefix, bucket=PUBLIC_BUCKET)
    max_bytes = int(args.max_segment_gb * (1024**3))
    candidates = [
        obj
        for obj in objects
        if (obj.key.endswith(".tif") or obj.key.endswith(".tiff"))
        and 0 < obj.size <= max_bytes
    ][: args.limit]

    scratch_root = Path(args.scratch_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Found {len(candidates)} candidate volumes under s3://{PUBLIC_BUCKET}/{args.s3_prefix}"
    )
    for idx, item in enumerate(candidates, start=1):
        safe_stem = item.key.replace("/", "__")
        preview_path = out_root / f"{safe_stem}.png"
        if preview_path.exists():
            print(f"[{idx}/{len(candidates)}] Skipping already scanned: {item.key}")
            continue

        scratch_root.mkdir(parents=True, exist_ok=True)
        local_file = scratch_root / Path(item.key).name
        print(
            f"[{idx}/{len(candidates)}] Downloading {item.key} ({item.size / (1024**2):.1f} MB)..."
        )
        try:
            subprocess.run(
                aws_download_command(item.key, local_file, bucket=PUBLIC_BUCKET),
                env=unsigned_aws_environment(),
                check=True,
            )
            volume = open_volume_array(local_file)
            depth, height, width = (int(v) for v in volume.shape)
            region_h = min(height, max(args.center_size, saved_patch[1]))
            region_w = min(width, max(args.center_size, saved_patch[2]))
            y0 = (height - region_h) // 2
            x0 = (width - region_w) // 2
            region = (y0, x0, y0 + region_h, x0 + region_w)

            probs = predict_region(
                model,
                volume,
                region=region,
                patch_size=saved_patch,
                stride=stride,
                device=device,
                batch_size=args.batch_size,
            )
            save_probability_preview(preview_path, probs)
            if args.save_tiff:
                save_probability_tiff(out_root / f"{safe_stem}.tif", probs)
            print(
                f"  -> Saved preview: {preview_path} (max_prob={float(probs.max()):.4f})"
            )
        finally:
            if scratch_root.exists():
                shutil.rmtree(scratch_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
