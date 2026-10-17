from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint, read_checkpoint_metadata, save_checkpoint
from .data_contract import audit_training_data, open_raster
from .dataset import PapyriDataset, open_volume_array
from .inference import predict_region
from .losses import AnisotropicFiberSuppressionLoss, FableLoss
from .metrics import (
    average_precision,
    binary_segmentation_metrics,
    probability_calibration_metrics,
    threshold_sweep_metrics,
)
from .model_factory import build_model_from_config, create_model
from .splits import SpatialSplitManifest


@dataclass(frozen=True)
class SmokeTestConfig:
    volume: str
    valid_mask: str
    labels: str
    split_manifest: str
    output_dir: str
    model_name: str = "residual_unet"
    backend: str = "fallback"
    base_channels: int = 4
    patch_size: tuple[int, int, int] = (16, 64, 64)
    coord_scale: int = 8
    train_steps: int = 2
    validation_samples: int = 2
    seed: int = 1337
    device: str = "cpu"
    z_jitter: int = 0
    synthetic_ink_prob: float = 0.0
    fiber_weight: float = 0.0


def _regions_overlap(first, second):
    return not (
        first.y1 <= second.y0
        or second.y1 <= first.y0
        or first.x1 <= second.x0
        or second.x1 <= first.x0
    )


def _batch(sample, device):
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if key in {"image", "target", "valid_mask"}
    }


def select_informative_validation_sample(validation, max_candidates=512):
    available = len(getattr(validation, "valid_coords", ())) or int(max_candidates)
    positive_fallback = None
    for index in range(min(int(max_candidates), int(available))):
        sample = validation[index]
        valid = sample["valid_mask"].bool()
        truth = sample["target"] >= 0.5
        positives = int((truth & valid).sum())
        negatives = int((~truth & valid).sum())
        if positives > 0 and negatives > 0:
            return index, sample
        if positives > 0 and positive_fallback is None:
            positive_fallback = (index, sample)
    if positive_fallback is not None:
        return positive_fallback
    raise ValueError("smoke validation candidates contain no positive pixels")


def run_smoke_test(config: SmokeTestConfig) -> dict:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_training_data(
        config.volume,
        config.valid_mask,
        config.labels,
        label_kind="ground_truth",
    )
    audit.raise_for_training()
    split = SpatialSplitManifest.load(config.split_manifest)
    train_regions = split.regions("train")
    validation_regions = split.regions("validation")
    if not train_regions or not validation_regions:
        raise ValueError("smoke test requires non-empty train and validation regions")
    split_overlap = any(
        _regions_overlap(train_region, validation_region)
        for train_region in train_regions
        for validation_region in validation_regions
    )
    if split_overlap:
        raise ValueError("training and validation regions overlap")

    common = {
        "zarr_path": config.volume,
        "mask_path": config.valid_mask,
        "ink_path": config.labels,
        "patch_size": config.patch_size,
        "coord_scale": config.coord_scale,
        "positive_fraction": 0.5,
        "z_policy": "center",
        "label_kind": "ground_truth",
        "data_audit": audit,
        "quiet": True,
    }
    training = PapyriDataset(
        **common,
        epoch_size=max(config.train_steps, 1),
        base_seed=config.seed,
        allowed_regions=train_regions,
        z_jitter=config.z_jitter,
        synthetic_ink_prob=config.synthetic_ink_prob,
        resample_each_epoch=True,
    )
    validation = PapyriDataset(
        **common,
        epoch_size=max(config.validation_samples, 1),
        base_seed=config.seed + 10_000,
        allowed_regions=validation_regions,
        sampling_mode="uniform",
        resample_each_epoch=False,
    )

    model = create_model(
        config.model_name,
        in_channels=1,
        num_classes=1,
        base_channels=config.base_channels,
        backend=config.backend,
        bidirectional=True,
    ).to(device)
    base_loss = FableLoss(pos_weight=3.0)
    loss_fn = (
        AnisotropicFiberSuppressionLoss(base_loss=base_loss, fiber_weight=config.fiber_weight)
        if config.fiber_weight > 0.0
        else base_loss
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.train_steps)
    )
    losses = []
    model.train()
    for step in range(config.train_steps):
        batch = _batch(training[step], device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image"])
        if isinstance(loss_fn, AnisotropicFiberSuppressionLoss):
            loss = loss_fn(
                logits,
                batch["target"],
                valid_mask=batch["valid_mask"],
                volume=batch["image"],
            )
        else:
            loss = loss_fn(logits, batch["target"], batch["valid_mask"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

    run_config = {
        "model": model.get_config(),
        "data": {
            "patch_size": list(config.patch_size),
            "coord_scale": config.coord_scale,
            "split_manifest": split.to_dict(),
            "normalization_divisor": float(training.normalize_divisor),
        },
        "smoke": asdict(config),
    }
    checkpoint_path = output_dir / "smoke_model.ckpt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        run_config=run_config,
        data_audit=audit,
        epoch=0,
        global_step=config.train_steps,
    )

    restored = build_model_from_config(run_config["model"]).to(device)
    load_checkpoint(
        checkpoint_path,
        model=restored,
        expected_config=run_config,
        strict=True,
        map_location=device,
    )
    checkpoint_round_trip = all(
        torch.equal(expected.detach().cpu(), actual.detach().cpu())
        for expected, actual in zip(model.parameters(), restored.parameters(), strict=True)
    )
    if not checkpoint_round_trip:
        raise RuntimeError("checkpoint round-trip changed model parameters")

    validation_sample_index, validation_sample = select_informative_validation_sample(
        validation
    )
    _, sample_y, sample_x = (int(value) for value in validation_sample["coords"])
    _, patch_height, patch_width = config.patch_size
    containing_region = next(
        region
        for region in validation_regions
        if region.y0 <= sample_y and region.x0 <= sample_x
        and sample_y + patch_height <= region.y1
        and sample_x + patch_width <= region.x1
    )
    evaluation_height = min(patch_height * 2, containing_region.y1 - sample_y)
    evaluation_width = min(patch_width * 2, containing_region.x1 - sample_x)
    region = (
        sample_y,
        sample_x,
        sample_y + evaluation_height,
        sample_x + evaluation_width,
    )
    volume = open_volume_array(config.volume)
    probabilities = predict_region(
        restored,
        volume,
        region=region,
        patch_size=config.patch_size,
        stride=(max(1, patch_height // 2), max(1, patch_width // 2)),
        device=device,
        batch_size=1,
    )
    label_raster = open_raster(config.labels)
    valid_raster = open_raster(config.valid_mask)
    y0, x0, y1, x1 = region
    target = np.asarray(label_raster[y0:y1, x0:x1], dtype=np.float32)
    target /= max(1.0, max(audit.label.observed_values, default=1.0))
    valid = np.asarray(valid_raster[y0:y1, x0:x1]) > 0
    probability_tensor = torch.from_numpy(probabilities)
    target_tensor = torch.from_numpy(target)
    valid_tensor = torch.from_numpy(valid)
    counts = binary_segmentation_metrics(
        probability_tensor,
        target_tensor,
        threshold=0.5,
        valid_mask=valid_tensor,
    )
    calibration = probability_calibration_metrics(
        probability_tensor,
        target_tensor,
        valid_mask=valid_tensor,
    )
    ranking = average_precision(
        probability_tensor,
        target_tensor,
        valid_mask=valid_tensor,
    )
    selected = threshold_sweep_metrics(
        probability_tensor,
        target_tensor,
        valid_mask=valid_tensor,
    )
    label_raster.close()
    valid_raster.close()
    if not np.isfinite(calibration["brier"]):
        raise RuntimeError("validation produced a non-finite Brier score")

    metadata = read_checkpoint_metadata(checkpoint_path)
    report = {
        "data_audit_ok": audit.ok,
        "checkpoint": str(checkpoint_path),
        "checkpoint_schema_version": metadata["schema_version"],
        "checkpoint_round_trip": checkpoint_round_trip,
        "split_overlap": split_overlap,
        "split_regions": {
            name: len(split.regions(name)) for name in ("train", "validation", "test")
        },
        "model": model.get_config(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_steps": config.train_steps,
        "train_loss": losses,
        "evaluation_region": list(region),
        "evaluation_sample_index": validation_sample_index,
        "evaluation_sampling": "mixed_positive_diagnostic",
        "validation": {
            **counts,
            **calibration,
            "fixed_threshold": 0.5,
            "average_precision": ranking["average_precision"],
            "selected_threshold": selected["threshold"],
            "precision_at_selected_threshold": selected["precision"],
            "recall_at_selected_threshold": selected["recall"],
            "f1_at_selected_threshold": selected["f1"],
            "fp_per_mpix_at_selected_threshold": selected["fp_per_mpix"],
        },
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def config_from_json(path, output_dir, steps):
    payload = json.loads(Path(path).read_text())
    data = payload["data"]
    model = payload["model"]
    training = payload["training"]
    return SmokeTestConfig(
        volume=data["volume"],
        valid_mask=data["valid_mask"],
        labels=data["ground_truth"],
        split_manifest=data["split"],
        output_dir=output_dir,
        model_name=model["name"],
        backend=model["backend"],
        base_channels=model["base_channels"],
        patch_size=tuple(data["patch_size_zyx"]),
        coord_scale=data["coord_scale"],
        train_steps=steps,
        validation_samples=min(training["validation_size"], 8),
        seed=training["seed"],
        device=training["device"],
        z_jitter=int(training.get("z_jitter", 0)),
        synthetic_ink_prob=float(training.get("synthetic_ink_prob", 0.0)),
        fiber_weight=float(training.get("fiber_weight", 0.0)),
    )


def main():
    parser = argparse.ArgumentParser(description="Run a leakage-resistant scientific smoke test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="runs/smoke")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    report = run_smoke_test(config_from_json(args.config, args.output_dir, args.steps))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
