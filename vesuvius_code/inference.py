from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

from .checkpoints import load_checkpoint, read_checkpoint_metadata
from .model_factory import build_model_from_config


def normalization_divisor(dtype) -> float:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    if np.issubdtype(dtype, np.floating):
        return 1.0
    raise TypeError(f"Unsupported volume dtype: {dtype}")


def sliding_origins(*, length: int, patch: int, stride: int) -> list[int]:
    length = int(length)
    patch = int(patch)
    stride = int(stride)
    if length <= 0 or patch <= 0 or stride <= 0:
        raise ValueError("length, patch, and stride must be positive")
    if patch > length:
        raise ValueError(f"patch {patch} exceeds length {length}")
    final = length - patch
    origins = list(range(0, final + 1, stride))
    if origins[-1] != final:
        origins.append(final)
    return origins


def hann_window_2d(height: int, width: int, minimum: float = 1e-3) -> np.ndarray:
    def one_dimensional(size):
        if size == 1:
            return np.ones(1, dtype=np.float32)
        return np.hanning(size).astype(np.float32)

    window = np.outer(one_dimensional(int(height)), one_dimensional(int(width)))
    return np.maximum(window, float(minimum)).astype(np.float32)


def load_kintsugi_model(path: str | Path, *, device="cpu"):
    device = torch.device(device)
    metadata = read_checkpoint_metadata(path, map_location="cpu")
    run_config = metadata["run_config"]
    model = build_model_from_config(run_config["model"]).to(device)
    load_checkpoint(
        path,
        model=model,
        expected_config=run_config,
        strict=True,
        map_location=device,
    )
    model.eval()
    return model, metadata


def predict_batch_with_tta(
    model, tensor: torch.Tensor, use_tta: bool = False
) -> torch.Tensor:
    """
    Evaluates model probabilities on 5D tensor (B, 1, D, H, W).
    If use_tta is True, averages predictions across the 8 dihedral transformations (D4 symmetry)
    of the spatial plane (H, W).
    """
    if not use_tta:
        return model.predict_proba(tensor)

    probs = []
    for flip in [False, True]:
        for k in range(4):
            x = tensor
            if flip:
                x = torch.flip(x, dims=[4])
            if k > 0:
                x = torch.rot90(x, k=k, dims=[3, 4])

            p = model.predict_proba(x)

            if k > 0:
                p = torch.rot90(p, k=-k, dims=[2, 3])
            if flip:
                p = torch.flip(p, dims=[3])
            probs.append(p)

    return torch.stack(probs, dim=0).mean(dim=0)


def predict_region(
    model,
    volume,
    *,
    region: tuple[int, int, int, int],
    patch_size: tuple[int, int, int],
    stride: tuple[int, int] | int,
    device="cpu",
    batch_size: int = 1,
    z_start: int | None = None,
    tta: bool = False,
) -> np.ndarray:
    device = torch.device(device)
    depth, height, width = (int(value) for value in volume.shape)
    patch_depth, patch_height, patch_width = (int(value) for value in patch_size)
    y0, x0, y1, x1 = (int(value) for value in region)
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError(f"Region {region} is outside volume YX {(height, width)}")
    if patch_depth > depth:
        raise ValueError("patch depth exceeds volume depth")
    if isinstance(stride, int):
        stride_y = stride_x = int(stride)
    else:
        stride_y, stride_x = (int(value) for value in stride)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    if z_start is None:
        z_start = (depth - patch_depth) // 2
    z_start = int(z_start)
    if not 0 <= z_start <= depth - patch_depth:
        raise ValueError("z_start does not fit the requested patch depth")

    y_origins = [y0 + value for value in sliding_origins(
        length=y1 - y0, patch=patch_height, stride=stride_y
    )]
    x_origins = [x0 + value for value in sliding_origins(
        length=x1 - x0, patch=patch_width, stride=stride_x
    )]
    locations = [(y, x) for y in y_origins for x in x_origins]
    prediction_sum = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    weight_sum = np.zeros_like(prediction_sum)
    window = hann_window_2d(patch_height, patch_width)
    divisor = normalization_divisor(volume.dtype)

    def process_batch(batch_locations):
        patches = []
        for y, x in batch_locations:
            patch = np.asarray(
                volume[
                    z_start : z_start + patch_depth,
                    y : y + patch_height,
                    x : x + patch_width,
                ]
            )
            if patch.shape != (patch_depth, patch_height, patch_width):
                raise RuntimeError(f"Incomplete patch at {(z_start, y, x)}: {patch.shape}")
            patches.append(np.ascontiguousarray(patch.astype(np.float32) / divisor))
        tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device)
        probabilities = (
            predict_batch_with_tta(model, tensor, use_tta=tta)
            .detach()
            .float()
            .cpu()
            .numpy()[:, 0]
        )
        for probability, (y, x) in zip(probabilities, batch_locations, strict=True):
            local_y = y - y0
            local_x = x - x0
            prediction_sum[
                local_y : local_y + patch_height,
                local_x : local_x + patch_width,
            ] += probability * window
            weight_sum[
                local_y : local_y + patch_height,
                local_x : local_x + patch_width,
            ] += window

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(locations), int(batch_size)):
            process_batch(locations[start : start + int(batch_size)])
    if np.any(weight_sum <= 0):
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    return prediction_sum / weight_sum


def save_probability_tiff(path: str | Path, probabilities, metadata: dict | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(probabilities, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("probabilities must be a finite 2D array")
    tifffile.imwrite(
        destination,
        array,
        metadata=metadata or {},
        bigtiff=array.nbytes >= 4_000_000_000,
    )
    return destination


def save_probability_preview(path: str | Path, probabilities) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(probabilities, dtype=np.float32)
    preview = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(preview).save(destination)
    return destination
