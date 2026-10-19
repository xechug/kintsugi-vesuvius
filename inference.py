import argparse
from pathlib import Path

import torch

from vesuvius_code.dataset import open_volume_array
from vesuvius_code.inference import (
    load_kintsugi_model,
    predict_region,
    save_probability_preview,
    save_probability_tiff,
)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_region(value):
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be y0,x0,y1,x1")
    return tuple(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Strict Kintsugi sliding-window inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zarr", default="sample_data/volume.zarr")
    parser.add_argument("--output", default="prediction_probability.tif")
    parser.add_argument("--preview")
    parser.add_argument("--region", type=parse_region)
    parser.add_argument("--center-size", type=int, default=1000)
    parser.add_argument("--patch-depth", type=int)
    parser.add_argument("--patch-height", type=int)
    parser.add_argument("--patch-width", type=int)
    parser.add_argument("--stride-y", type=int)
    parser.add_argument("--stride-x", type=int)
    parser.add_argument("--z-start", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tta", action="store_true", help="Apply 8-fold D4 dihedral Test-Time Augmentation")
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else choose_device()
    model, metadata = load_kintsugi_model(args.checkpoint, device=device)
    volume = open_volume_array(args.zarr)
    depth, height, width = (int(value) for value in volume.shape)
    saved_patch = tuple(int(value) for value in metadata["run_config"]["data"]["patch_size"])
    patch_size = (
        args.patch_depth or saved_patch[0],
        args.patch_height or saved_patch[1],
        args.patch_width or saved_patch[2],
    )

    if args.region is None:
        region_height = min(height, max(args.center_size, patch_size[1]))
        region_width = min(width, max(args.center_size, patch_size[2]))
        y0 = (height - region_height) // 2
        x0 = (width - region_width) // 2
        region = (y0, x0, y0 + region_height, x0 + region_width)
    else:
        region = args.region

    stride = (
        args.stride_y or max(1, patch_size[1] // 2),
        args.stride_x or max(1, patch_size[2] // 2),
    )
    print(f"device={device}")
    print(f"volume_shape={(depth, height, width)} dtype={volume.dtype}")
    print(f"region={region} patch={patch_size} stride={stride} tta={args.tta}")
    probabilities = predict_region(
        model,
        volume,
        region=region,
        patch_size=patch_size,
        stride=stride,
        device=device,
        batch_size=args.batch_size,
        z_start=args.z_start,
        tta=args.tta,
    )
    output = save_probability_tiff(
        args.output,
        probabilities,
        metadata={
            "checkpoint": str(Path(args.checkpoint)),
            "checkpoint_config_sha256": metadata["run_config_sha256"],
            "region_yx": list(region),
            "patch_zyx": list(patch_size),
            "stride_yx": list(stride),
        },
    )
    if args.preview:
        save_probability_preview(args.preview, probabilities)
    print(
        f"saved={output} min={float(probabilities.min()):.6f} "
        f"max={float(probabilities.max()):.6f} mean={float(probabilities.mean()):.6f}"
    )


if __name__ == "__main__":
    main()
