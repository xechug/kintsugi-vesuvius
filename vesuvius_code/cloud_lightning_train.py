import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from vesuvius_code.checkpoints import export_lightning_checkpoint
from vesuvius_code.data_contract import audit_training_data
from vesuvius_code.data_manifest import DataManifest, DataSource
from vesuvius_code.dataset import (
    BalancedMultiSourceDataset,
    PapyriDataset,
    make_dataloader,
)
from vesuvius_code.losses import AnisotropicFiberSuppressionLoss, FableLoss
from vesuvius_code.model_factory import create_model
from vesuvius_code.splits import SpatialSplitManifest
from vesuvius_code.training_config import TrainingConfig


def import_lightning():
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
        from torchmetrics.classification import BinaryAveragePrecision
    except ModuleNotFoundError:
        try:
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
            from torchmetrics.classification import BinaryAveragePrecision
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Lightning is not installed. Install the pinned cloud environment first."
            ) from error
    return pl, ModelCheckpoint, LearningRateMonitor, BinaryAveragePrecision


def build_parser():
    parser = argparse.ArgumentParser(description="Validation-first Kintsugi Lightning trainer")
    parser.add_argument("--data-manifest")
    parser.add_argument("--zarr")
    parser.add_argument("--mask")
    parser.add_argument("--ink")
    parser.add_argument("--split-manifest")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--gpu-model", default="unknown")
    parser.add_argument("--cpu-ram-budget-gib", type=float, default=32.0)
    parser.add_argument(
        "--model",
        choices=(
            "residual_unet",
            "residual_physical",
            "kintsugi_axial",
            "kintsugi_physical",
            "legacy_kintsugi",
        ),
        default="kintsugi_axial",
    )
    parser.add_argument(
        "--backend", choices=("mamba", "fallback", "auto"), default="mamba"
    )
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--positive-fraction", type=float, default=0.33)
    parser.add_argument(
        "--augment-xy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--intensity-jitter", type=float, default=0.10)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--z-jitter", type=int, default=0)
    parser.add_argument("--synthetic-ink-prob", type=float, default=0.0)
    parser.add_argument("--fiber-weight", type=float, default=0.0)
    parser.add_argument(
        "--align-to-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epoch-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=2048)
    parser.add_argument("--patch-depth", type=int, default=64)
    parser.add_argument("--patch-height", type=int, default=128)
    parser.add_argument("--patch-width", type=int, default=128)
    parser.add_argument("--coord-scale", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def parse_args():
    args = build_parser().parse_args()
    validate_data_args(args)
    return args


def validate_data_args(args):
    single_values = {
        "zarr": args.zarr,
        "mask": args.mask,
        "ink": args.ink,
        "split-manifest": args.split_manifest,
    }
    provided_single = [name for name, value in single_values.items() if value]
    if args.data_manifest and provided_single:
        raise ValueError("provide either --data-manifest or single-source paths, not both")
    if not 0.0 <= args.positive_fraction <= 1.0:
        raise ValueError("positive fraction must be between zero and one")
    if not 0.0 <= args.intensity_jitter <= 1.0 or not 0.0 <= args.noise_std <= 1.0:
        raise ValueError("augmentation magnitudes must be between zero and one")
    if args.z_jitter < 0 or not 0.0 <= args.synthetic_ink_prob <= 1.0 or args.fiber_weight < 0.0:
        raise ValueError("z_jitter >= 0, synthetic_ink_prob in [0, 1], and fiber_weight >= 0 required")
    if args.data_manifest:
        return
    missing = [name for name, value in single_values.items() if not value]
    if missing:
        raise ValueError(
            "single-source training requires "
            + ", ".join(f"--{name}" for name in missing)
        )


def training_config_from_args(args, *, validation_regions):
    return TrainingConfig(
        gpu_model=args.gpu_model,
        devices=args.devices,
        precision=args.precision,
        batch_size_per_device=args.batch_size,
        accumulate_grad_batches=args.accumulate,
        workers_per_device=args.workers,
        prefetch_factor=args.prefetch_factor,
        patch_size=(args.patch_depth, args.patch_height, args.patch_width),
        cpu_ram_budget_gib=args.cpu_ram_budget_gib,
        validation_regions=int(validation_regions),
    )


def _distributed_counts(probs, target, valid, threshold=0.5):
    predicted = probs >= threshold
    truth = target >= 0.5
    valid = valid.to(dtype=torch.bool)
    return {
        "tp": (predicted & truth & valid).sum().float(),
        "fp": (predicted & ~truth & valid).sum().float(),
        "fn": (~predicted & truth & valid).sum().float(),
        "valid": valid.sum().float(),
    }


def load_data_contexts(args):
    if args.data_manifest:
        manifest = DataManifest.load(args.data_manifest)
        sources = manifest.sources
        manifest_payload = manifest.to_dict()
    else:
        source = DataSource(
            name="single_source",
            zarr=str(Path(args.zarr).resolve()),
            mask=str(Path(args.mask).resolve()),
            ink=str(Path(args.ink).resolve()),
            split_manifest=str(Path(args.split_manifest).resolve()),
            splits={"train": "train", "validation": "validation", "test": "test"},
        )
        sources = (source,)
        manifest_payload = {"schema_version": 1, "sources": [source.__dict__]}

    contexts = []
    for source in sources:
        split_manifest = SpatialSplitManifest.load(source.split_manifest)
        data_audit = audit_training_data(
            source.zarr,
            source.mask,
            source.ink,
            label_kind="ground_truth",
        )
        data_audit.raise_for_training()
        if tuple(data_audit.volume.shape[-2:]) != (
            split_manifest.height,
            split_manifest.width,
        ):
            raise ValueError(
                f"split manifest dimensions do not match source {source.name}"
            )
        for role, split_name in source.splits.items():
            if role in {"train", "validation"} and not split_manifest.regions(split_name):
                raise ValueError(
                    f"source {source.name} has no regions for {role}:{split_name}"
                )
        contexts.append(
            {
                "source": source,
                "split_manifest": split_manifest,
                "data_audit": data_audit,
            }
        )
    return contexts, manifest_payload


def main():
    args = parse_args()
    data_contexts, data_manifest_payload = load_data_contexts(args)
    train_contexts = [
        context
        for context in data_contexts
        if context["source"].split_for("train") is not None
    ]
    validation_contexts = [
        context
        for context in data_contexts
        if context["source"].split_for("validation") is not None
    ]
    validation_region_count = sum(
        len(
            context["split_manifest"].regions(
                context["source"].split_for("validation")
            )
        )
        for context in validation_contexts
    )
    runtime_config = training_config_from_args(
        args, validation_regions=validation_region_count
    )
    runtime_config.validate()
    if not train_contexts or not validation_contexts:
        raise ValueError("training requires train and validation data sources")
    data_audits = {
        context["source"].name: context["data_audit"].to_dict()
        for context in data_contexts
    }

    pl, ModelCheckpoint, LearningRateMonitor, BinaryAveragePrecision = import_lightning()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    class KintsugiModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = create_model(
                args.model,
                in_channels=1,
                num_classes=1,
                base_channels=args.base_channels,
                backend=args.backend,
                bidirectional=True,
            )
            base_loss = FableLoss(pos_weight=args.pos_weight, tversky_beta=0.7)
            self.loss_fn = (
                AnisotropicFiberSuppressionLoss(
                    base_loss=base_loss, fiber_weight=args.fiber_weight
                )
                if args.fiber_weight > 0.0
                else base_loss
            )
            self.val_average_precision = BinaryAveragePrecision(thresholds=None)
            self.run_config = {
                "model": self.model.get_config(),
                "training": runtime_config.to_dict(),
                "optimization": {
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "pos_weight": args.pos_weight,
                    "fiber_weight": args.fiber_weight,
                    "seed": args.seed,
                },
                "data": {
                    "patch_size": [args.patch_depth, args.patch_height, args.patch_width],
                    "coord_scale": args.coord_scale,
                    "epoch_size": args.epoch_size,
                    "validation_size": args.validation_size,
                    "validation_sampling": "non_overlapping_grid",
                    "validation_average_precision": "exact",
                    "positive_fraction": args.positive_fraction,
                    "augmentation": {
                        "xy": args.augment_xy,
                        "intensity_jitter": args.intensity_jitter,
                        "noise_std": args.noise_std,
                        "z_jitter": args.z_jitter,
                        "synthetic_ink_prob": args.synthetic_ink_prob,
                    },
                    "align_to_chunks": args.align_to_chunks,
                    "manifest": data_manifest_payload,
                },
            }
            self.save_hyperparameters({"run_config": self.run_config})

        def forward(self, image):
            return self.model(image)

        def training_step(self, batch, batch_idx):
            logits = self.model(batch["image"])
            if isinstance(self.loss_fn, AnisotropicFiberSuppressionLoss):
                loss = self.loss_fn(
                    logits,
                    batch["target"],
                    valid_mask=batch["valid_mask"],
                    volume=batch["image"],
                )
            else:
                loss = self.loss_fn(logits, batch["target"], batch["valid_mask"])
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
                batch_size=batch["image"].shape[0],
            )
            return loss

        def validation_step(self, batch, batch_idx):
            logits = self.model(batch["image"])
            if isinstance(self.loss_fn, AnisotropicFiberSuppressionLoss):
                loss = self.loss_fn(
                    logits,
                    batch["target"],
                    valid_mask=batch["valid_mask"],
                    volume=batch["image"],
                )
            else:
                loss = self.loss_fn(logits, batch["target"], batch["valid_mask"])
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch["image"].shape[0],
            )
            probabilities = torch.sigmoid(logits)
            counts = _distributed_counts(
                probabilities, batch["target"], batch["valid_mask"]
            )
            valid = batch["valid_mask"].bool()
            self.val_average_precision.update(
                probabilities[valid],
                (batch["target"][valid] >= 0.5).long(),
            )
            self.log(
                "val_ap",
                self.val_average_precision,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            for name, value in counts.items():
                self.log(
                    f"val_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx="sum",
                    sync_dist=True,
                )

        def on_train_epoch_start(self):
            self.trainer.datamodule.train_dataset.set_epoch(self.current_epoch)

        def on_save_checkpoint(self, checkpoint):
            checkpoint["kintsugi_schema_version"] = 2
            checkpoint["kintsugi_run_config"] = self.run_config
            checkpoint["data_audits"] = data_audits

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.trainer.estimated_stepping_batches),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    class KintsugiDataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            def make_source_dataset(
                context,
                *,
                role,
                epoch_size,
                source_index,
            ):
                source = context["source"]
                split_name = source.split_for(role)
                regions = context["split_manifest"].regions(split_name)
                return PapyriDataset(
                    zarr_path=source.zarr,
                    mask_path=source.mask,
                    ink_path=source.ink,
                    patch_size=(
                        args.patch_depth,
                        args.patch_height,
                        args.patch_width,
                    ),
                    coord_scale=args.coord_scale,
                    positive_fraction=args.positive_fraction,
                    z_policy="random" if role == "train" else "center",
                    label_kind="ground_truth",
                    data_audit=context["data_audit"],
                    quiet=True,
                    epoch_size=epoch_size,
                    base_seed=args.seed + source_index * 100_003,
                    allowed_regions=regions,
                    sampling_mode="balanced" if role == "train" else "grid",
                    augment_xy=args.augment_xy if role == "train" else False,
                    intensity_jitter=args.intensity_jitter if role == "train" else 0.0,
                    noise_std=args.noise_std if role == "train" else 0.0,
                    z_jitter=args.z_jitter if role == "train" else 0,
                    synthetic_ink_prob=args.synthetic_ink_prob if role == "train" else 0.0,
                    align_to_chunks=args.align_to_chunks if role == "train" else False,
                    resample_each_epoch=role == "train",
                )

            train_per_source = math.ceil(args.epoch_size / len(train_contexts))
            validation_per_source = math.ceil(
                args.validation_size / len(validation_contexts)
            )
            train_parts = [
                make_source_dataset(
                    context,
                    role="train",
                    epoch_size=train_per_source,
                    source_index=index,
                )
                for index, context in enumerate(train_contexts)
            ]
            validation_parts = [
                make_source_dataset(
                    context,
                    role="validation",
                    epoch_size=validation_per_source,
                    source_index=index + len(train_contexts),
                )
                for index, context in enumerate(validation_contexts)
            ]
            self.train_dataset = BalancedMultiSourceDataset(
                train_parts,
                epoch_size=args.epoch_size,
            )
            self.validation_dataset = ConcatDataset(validation_parts)

        def train_dataloader(self):
            return make_dataloader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.workers > 0,
                drop_last=True,
            )

        def val_dataloader(self):
            return make_dataloader(
                self.validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.workers > 0,
                drop_last=False,
            )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="kintsugi-{epoch:03d}-{val_ap:.5f}",
        save_top_k=3,
        monitor="val_ap",
        mode="max",
        save_last=True,
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=args.devices,
        strategy="ddp_find_unused_parameters_false" if args.devices > 1 else "auto",
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate,
        gradient_clip_val=1.0,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=25,
        deterministic=True,
        benchmark=False,
        use_distributed_sampler=True,
    )
    module = KintsugiModule()
    data_module = KintsugiDataModule()
    (output_dir / "run_config.json").write_text(
        json.dumps(module.run_config, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(module.run_config, indent=2, sort_keys=True))
    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=args.resume,
    )
    if trainer.is_global_zero:
        if not checkpoint.best_model_path:
            raise RuntimeError("Lightning did not produce a best checkpoint")
        export_model = create_model(
            args.model,
            in_channels=1,
            num_classes=1,
            base_channels=args.base_channels,
            backend=args.backend,
            bidirectional=True,
        )
        strict_path = export_lightning_checkpoint(
            checkpoint.best_model_path,
            output_dir / "best.strict.ckpt",
            model=export_model,
            run_config=module.run_config,
            data_audit={"sources": data_audits},
        )
        print(f"strict_checkpoint={strict_path}")


if __name__ == "__main__":
    main()
