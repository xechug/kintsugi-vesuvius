from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from vesuvius_code.data_contract import audit_training_data
from vesuvius_code.dataset import PapyriDataset
from vesuvius_code.losses import FableLoss
from vesuvius_code.metrics import average_precision, threshold_sweep_metrics
from vesuvius_code.model_factory import create_model
from vesuvius_code.smoke_test import config_from_json
from vesuvius_code.splits import SpatialSplitManifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="Verify that a model can memorize one audited positive patch"
    )
    parser.add_argument("--config", default="configs/residual_local_baseline.json")
    parser.add_argument("--output-dir", default="runs/overfit_diagnostic")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser


def _batch(sample, device):
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if key in {"image", "target", "valid_mask"}
    }


def _evaluate(model, batch):
    model.eval()
    with torch.no_grad():
        probabilities = model.predict_proba(batch["image"])
    ranking = average_precision(
        probabilities,
        batch["target"],
        valid_mask=batch["valid_mask"],
    )
    selected = threshold_sweep_metrics(
        probabilities,
        batch["target"],
        valid_mask=batch["valid_mask"],
    )
    return {**ranking, "selected": selected}


def choose_informative_sample(
    dataset,
    *,
    candidates=128,
    target_positive_rate=0.1,
):
    best = None
    for index in range(int(candidates)):
        sample = dataset[index]
        valid = sample["valid_mask"].bool()
        if not bool(valid.any()):
            continue
        rate = float(sample["target"][valid].float().mean())
        distance = abs(rate - float(target_positive_rate))
        if best is None or distance < best[0]:
            best = (distance, sample, rate)
    if best is None:
        raise ValueError("no valid patch was found for the overfit diagnostic")
    return best[1], best[2]


def run_overfit_diagnostic(config, *, iterations=300, learning_rate=1e-3):
    if iterations <= 0 or learning_rate <= 0:
        raise ValueError("iterations and learning rate must be positive")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    audit = audit_training_data(
        config.volume,
        config.valid_mask,
        config.labels,
        label_kind="ground_truth",
    )
    audit.raise_for_training()
    split = SpatialSplitManifest.load(config.split_manifest)
    training_regions = split.regions("train")
    if not training_regions:
        raise ValueError("split manifest contains no training regions")

    dataset = PapyriDataset(
        config.volume,
        config.valid_mask,
        config.labels,
        patch_size=config.patch_size,
        coord_scale=config.coord_scale,
        epoch_size=128,
        positive_fraction=1.0,
        z_policy="center",
        data_audit=audit,
        allowed_regions=training_regions,
        resample_each_epoch=False,
        base_seed=config.seed,
        quiet=True,
    )
    sample, selected_positive_rate = choose_informative_sample(dataset)
    batch = _batch(sample, device)
    model = create_model(
        config.model_name,
        in_channels=1,
        num_classes=1,
        base_channels=config.base_channels,
        backend=config.backend,
        bidirectional=True,
    ).to(device)
    loss_fn = FableLoss(pos_weight=3.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    initial = _evaluate(model, batch)
    losses = []
    model.train()
    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image"])
        loss = loss_fn(logits, batch["target"], batch["valid_mask"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("overfit diagnostic produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    final = _evaluate(model, batch)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": asdict(config),
        "model": model.get_config(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "coords": sample["coords"].tolist(),
        "positive_rate": selected_positive_rate,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "initial": initial,
        "final": final,
    }
    (output_dir / "overfit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main():
    args = build_parser().parse_args()
    config = config_from_json(args.config, args.output_dir, 1)
    report = run_overfit_diagnostic(
        config,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
