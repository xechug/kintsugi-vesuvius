import os

import torch

try:
    from .checkpoints import load_checkpoint, save_checkpoint
    from .dataset import PapyriDataset, make_dataloader
    from .losses import FableLoss
    from .metrics import binary_segmentation_metrics, format_metrics
    from .model_factory import create_model
except ImportError:
    from checkpoints import load_checkpoint, save_checkpoint
    from dataset import PapyriDataset, make_dataloader
    from losses import FableLoss
    from metrics import binary_segmentation_metrics, format_metrics
    from model_factory import create_model


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    zarr_dir=None,
    mask_file=None,
    ink_file=None,
    save_path="runs/manual/model.ckpt",
    epochs=20,
    batch_size=2,
    num_workers=None,
    learning_rate=2e-4,
    pos_weight=3.0,
    base_channels=16,
    patch_size=(32, 64, 64),
    coord_scale=16,
    epoch_size=1000,
    metric_threshold=0.5,
    positive_fraction=0.33,
    label_kind="ground_truth",
    resume_path=None,
    device_override=None,
    model_name="kintsugi_axial",
    model_backend="fallback",
):
    if zarr_dir is None or mask_file is None or ink_file is None:
        raise ValueError("zarr_dir, mask_file, and ink_file must be explicitly provided")
    if not os.path.exists(zarr_dir) or not os.path.exists(mask_file) or not os.path.exists(ink_file):
        raise FileNotFoundError("Missing zarr, papyrus mask, or ink labels")

    device = torch.device(device_override) if device_override is not None else choose_device()
    print(f"Device: {device}", flush=True)

    if num_workers is None:
        num_workers = 4 if device.type == "cuda" else 0

    dataset = PapyriDataset(
        zarr_dir,
        mask_file,
        ink_file,
        patch_size=patch_size,
        stride=(32, 32, 32),
        downsample_factor=1,
        epoch_size=epoch_size,
        positive_fraction=positive_fraction,
        z_policy="random",
        coord_scale=coord_scale,
        label_kind=label_kind,
    )
    dataloader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=2,
        drop_last=True,
    )

    model = create_model(
        model_name,
        in_channels=1,
        num_classes=1,
        base_channels=base_channels,
        backend=model_backend,
        bidirectional=True,
    ).to(device)

    criterion = FableLoss(pos_weight=pos_weight, tversky_beta=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * len(dataloader)),
    )

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None

    run_config = {
        "model": model.get_config(),
        "data": {
            "patch_size": list(patch_size),
            "coord_scale": int(coord_scale),
            "epoch_size": int(epoch_size),
            "positive_fraction": float(positive_fraction),
            "label_kind": str(label_kind),
            "normalization_divisor": float(dataset.normalize_divisor),
        },
        "optimization": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "pos_weight": float(pos_weight),
            "weight_decay": 1e-4,
        },
    }

    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        resumed = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config=run_config,
            strict=True,
            restore_rng=True,
            map_location=device,
        )
        start_epoch = int(resumed["epoch"]) + 1
        global_step = int(resumed["global_step"])

    model.train()
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            volumes = batch["image"].to(device, non_blocking=True)
            masks = batch["target"].to(device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model(volumes)
                loss = criterion(logits, masks, valid_mask=valid_masks)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += float(loss.detach().cpu())

            if step % 25 == 0:
                lr = scheduler.get_last_lr()[0]
                with torch.no_grad():
                    probs = torch.sigmoid(logits.detach())
                    metrics = binary_segmentation_metrics(
                        probs.cpu(),
                        masks.detach().cpu(),
                        threshold=metric_threshold,
                        valid_mask=valid_masks.detach().cpu(),
                    )
                print(
                    f"epoch={epoch + 1}/{epochs} step={step} "
                    f"loss={loss.item():.4f} lr={lr:.6g} {format_metrics(metrics)}"
                    ,
                    flush=True,
                )
            global_step += 1

        print(
            f"epoch={epoch + 1} mean_loss={epoch_loss / max(1, len(dataloader)):.4f}",
            flush=True,
        )
        save_checkpoint(
            save_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            run_config=run_config,
            data_audit=dataset.data_audit,
            epoch=epoch,
            global_step=global_step,
        )

    return save_path


if __name__ == "__main__":
    train_model()
