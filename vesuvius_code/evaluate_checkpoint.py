from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from vesuvius_code.checkpoints import (
    CheckpointCompatibilityError,
    load_checkpoint,
    read_checkpoint_metadata,
)
from vesuvius_code.dataset import PapyriDataset, make_dataloader
from vesuvius_code.metrics import (
    aggregate_binary_segmentation_metrics,
    average_precision,
    binary_segmentation_metrics,
    probability_calibration_metrics,
)
from vesuvius_code.model_factory import build_model_from_config
from vesuvius_code.splits import SpatialSplitManifest
from vesuvius_code.train import choose_device


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a strict Kintsugi checkpoint on a frozen spatial split"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--ink", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Maximum number of deterministic non-overlapping evaluation patches.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-depth", type=int)
    parser.add_argument("--patch-height", type=int)
    parser.add_argument("--patch-width", type=int)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{value / 20:.2f}" for value in range(1, 20)),
        help="Validation-only threshold candidates.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Frozen threshold. Mandatory for test; normally selected on validation.",
    )
    parser.add_argument("--require-training-data-match", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=9001)
    return parser


def parse_args():
    args = build_parser().parse_args()
    validate_args(args)
    return args


def _parse_thresholds(value):
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be between zero and one")
    return thresholds


def validate_args(args):
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch size must be positive")
    if not 0.0 <= args.positive_fraction <= 1.0:
        raise ValueError("positive fraction must be between zero and one")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if args.split == "test" and args.threshold is None:
        raise ValueError("test evaluation requires a frozen threshold from validation")
    if args.threshold is None:
        _parse_thresholds(args.thresholds)


def enforce_data_identity(checkpoint_audit, current_audit, *, require_match):
    def fingerprint(audit):
        return (
            audit.get("volume", {}).get("metadata_sha256"),
            audit.get("valid_mask", {}).get("sha256"),
            audit.get("label", {}).get("sha256"),
        )

    actual = fingerprint(current_audit)
    training_audits = checkpoint_audit.get("sources", {}).values()
    if not training_audits:
        training_audits = (checkpoint_audit,)
    matches = any(
        all(left is not None and left == right for left, right in zip(fingerprint(audit), actual))
        for audit in training_audits
    )
    if require_match and not matches:
        raise CheckpointCompatibilityError(
            "evaluation data differs from the checkpoint data audit"
        )
    return matches


def _select_threshold(rows_by_threshold):
    rows = [
        {"threshold": threshold, **aggregate_binary_segmentation_metrics(metric_rows)}
        for threshold, metric_rows in rows_by_threshold.items()
    ]
    return max(rows, key=lambda row: (row["f1"], -row["fp"], row["threshold"])), rows


def main():
    args = parse_args()
    device = choose_device()
    metadata = read_checkpoint_metadata(args.checkpoint)
    run_config = metadata["run_config"]
    model_config = run_config["model"]
    data_config = run_config["data"]
    saved_patch = tuple(int(value) for value in data_config["patch_size"])
    patch_size = (
        args.patch_depth or saved_patch[0],
        args.patch_height or saved_patch[1],
        args.patch_width or saved_patch[2],
    )

    split_manifest = SpatialSplitManifest.load(args.split_manifest)
    regions = split_manifest.regions(args.split)
    if not regions:
        raise ValueError(f"split manifest contains no {args.split} regions")
    dataset = PapyriDataset(
        args.zarr,
        args.mask,
        args.ink,
        patch_size=patch_size,
        epoch_size=args.samples,
        positive_fraction=args.positive_fraction,
        z_policy="center",
        coord_scale=int(data_config["coord_scale"]),
        label_kind="ground_truth",
        allowed_regions=regions,
        sampling_mode="grid",
        resample_each_epoch=False,
        base_seed=args.seed,
        quiet=True,
    )
    if tuple(dataset.volume_shape[-2:]) != (
        split_manifest.height,
        split_manifest.width,
    ):
        raise ValueError("split manifest dimensions do not match evaluation volume")
    loader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = build_model_from_config(model_config).to(device)
    load_checkpoint(
        args.checkpoint,
        model=model,
        expected_config=run_config,
        strict=True,
        map_location=device,
    )
    current_audit = dataset.data_audit.to_dict()
    identity_matches = enforce_data_identity(
        metadata.get("data_audit", {}),
        current_audit,
        require_match=args.require_training_data_match,
    )

    thresholds = (
        [float(args.threshold)]
        if args.threshold is not None
        else _parse_thresholds(args.thresholds)
    )
    rows_by_threshold = {threshold: [] for threshold in thresholds}
    probability_batches = []
    target_batches = []
    valid_batches = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            volumes = batch["image"].to(device)
            targets = batch["target"].cpu()
            valid = batch["valid_mask"].cpu()
            probs = model.predict_proba(volumes).cpu()
            probability_batches.append(probs)
            target_batches.append(targets)
            valid_batches.append(valid)
            for threshold in thresholds:
                rows_by_threshold[threshold].append(
                    binary_segmentation_metrics(
                        probs,
                        targets,
                        threshold=threshold,
                        valid_mask=valid,
                    )
                )

    probabilities = torch.cat(probability_batches)
    targets = torch.cat(target_batches)
    valid = torch.cat(valid_batches)
    selected, threshold_rows = _select_threshold(rows_by_threshold)
    ranking = average_precision(probabilities, targets, valid_mask=valid)
    if not math.isfinite(ranking["average_precision"]):
        raise ValueError(
            "evaluation patches contain no positive ink pixels; increase --samples "
            "or inspect the selected split"
        )
    calibration = probability_calibration_metrics(
        probabilities,
        targets,
        valid_mask=valid,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "model": model_config,
        "device": str(device),
        "source": {
            "zarr": str(args.zarr),
            "mask": str(args.mask),
            "ink": str(args.ink),
            "split_manifest": str(args.split_manifest),
            "split": args.split,
            "data_identity_matches_training": identity_matches,
            "data_audit": current_audit,
        },
        "requested_samples": args.samples,
        "evaluated_samples": len(dataset),
        "evaluation_sampling": "non_overlapping_grid",
        "patch_size": list(patch_size),
        "threshold_policy": (
            "frozen" if args.threshold is not None else "selected_on_validation"
        ),
        "selected": selected,
        "threshold_metrics": threshold_rows,
        "ranking": ranking,
        "calibration": calibration,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
