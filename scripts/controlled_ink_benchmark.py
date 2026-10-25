"""Budget-sized native-9um ablation with frozen segment holdout and raw artifacts."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import numcodecs
import numpy as np
import torch
import torch.nn.functional as F

from vesuvius_code.baselines import KintsugiPhysicalUNet3DTo2D, ResidualUNet3DTo2D
from vesuvius_code.losses import FableLoss
from vesuvius_code.numpy_metrics import binary_report, precision_recall_summary

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
SEGMENTS = {
    "w035": "20260317000000-w035_2026031718",
    "w039": "20260302000000-w039_2026030210",
    "w040": "20250831000000-w040_2025083102",
    "w041": "20260108000000-w041_2026010816",
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_url(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt == 2:
                raise
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(0.5 * (attempt + 1))


def download_window(url, window, destination, reviewed_mask=None):
    """Read native uint8 Zarr v2 chunks, failing on missing reviewed pixels."""
    destination = Path(destination)
    if destination.exists():
        return np.load(destination, mmap_mode="r")
    meta = json.loads(read_url(url + "/0/.zarray"))
    if meta["dtype"] != "|u1" or meta["order"] != "C":
        raise ValueError(f"Unsupported input contract: {meta}")
    depth, height, width = meta["shape"]
    dz, dy, dx = meta["chunks"]
    cy0, cy1, cx0, cx1 = window
    if cy1 * dy > height or cx1 * dx > width or dz < depth:
        raise ValueError("Window must contain full chunks and all native depth slices")
    volume = np.zeros((depth, (cy1 - cy0) * dy, (cx1 - cx0) * dx), np.uint8)
    codec = numcodecs.get_codec(meta["compressor"]) if meta.get("compressor") else None
    sep = meta.get("dimension_separator", ".")
    tasks = [(cy, cx) for cy in range(cy0, cy1) for cx in range(cx0, cx1)]

    def fetch(coord):
        cy, cx = coord
        try:
            raw = read_url(url + "/0/" + sep.join(map(str, (0, cy, cx))))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            return cy, cx, None
        if codec:
            raw = codec.decode(raw)
        return cy, cx, np.frombuffer(raw, np.uint8).reshape(dz, dy, dx)[:depth]

    missing = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for cy, cx, array in pool.map(fetch, tasks):
            y, x = (cy - cy0) * dy, (cx - cx0) * dx
            if array is None:
                missing.append([cy, cx])
                if reviewed_mask is not None and reviewed_mask[y:y+dy, x:x+dx].any():
                    raise RuntimeError(f"Missing CT chunk over reviewed labels: {cy},{cx}")
            else:
                volume[:, y:y+dy, x:x+dx] = array
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, volume)
    write_json(destination.with_suffix(".source.json"), {
        "url": url, "level": 0, "window_chunks": window, "metadata": meta,
        "missing_unreviewed_chunks": missing, "sha256": sha256(destination),
    })
    print(f"Prepared {destination.name}: {volume.shape}, missing={len(missing)}", flush=True)
    return volume


def box_sum(array, radius):
    padded = np.pad(array.astype(np.float64), radius, mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    size = 2 * radius + 1
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


def prepare_slab(volume, offset=0):
    start = (volume.shape[0] - 16) // 2 + int(offset)
    if start < 0 or start + 16 > volume.shape[0]:
        raise ValueError("Depth window does not fit")
    slab = np.asarray(volume[start:start+16], np.float32) / 255.0
    valid = slab.max(0) > 0.02
    if not valid.any():
        raise ValueError("Empty papyrus crop")
    means = slab[:, valid].mean(1)
    slab[:, ~valid] = means[:, None]
    if slab[:6].mean() < slab[10:].mean():
        slab = slab[::-1].copy()
    values = slab[:, valid]
    slab = (slab - values.mean()) / max(float(values.std()), 1e-4)
    slab[:, ~valid] = 0
    drift = box_sum(slab.mean(0) * valid, 64) / np.maximum(box_sum(valid, 64), 1e-4)
    slab -= 0.7 * drift.astype(np.float32)[None]
    slab = (np.clip(slab, -3.5, 3.5) + 3.5) / 7
    slab[:, ~valid] = 0.5
    return slab.astype(np.float32), valid


def prepare(config, data, repo):
    data.mkdir(parents=True, exist_ok=True)
    write_json(data / "protocol.json", config)
    for name, window in config["windows"].items():
        cy0, cy1, cx0, cx1 = window
        crop = np.s_[cy0*128:cy1*128, cx0*128:cx1*128]
        mask = np.load(repo / f"data/scroll_labels/0139_{name}_sup_l0.npy", mmap_mode="r")[crop] > 0
        ink = np.load(repo / f"data/scroll_labels/0139_{name}_ink_l0.npy", mmap_mode="r")[crop] > 0
        np.savez_compressed(data / f"{name}_labels.npz", ink=ink, supervision=mask)
        url = f"{S3}/PHerc0139/segments/{SEGMENTS[name]}/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
        download_window(url, window, data / f"{name}_volume.npy", mask)
    seg = config["p800_segment"]
    url = f"{S3}/PHerc0800/segments/{seg}/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr"
    download_window(url, config["p800_window"], data / "p800_volume.npy")
    provenance = {f.name: sha256(f) for f in sorted(data.glob("*.npy"))}
    provenance.update({f.name: sha256(f) for f in sorted(data.glob("*_labels.npz"))})
    write_json(data / "hashes.json", provenance)


def build_model(name, config):
    physical = name != "residual"
    options = dict(base_channels=config["base_channels"], physical_stem=physical,
                   sheet_unwarp=physical, multi_scale_stem=True)
    if name == "kintsugi_physical":
        return KintsugiPhysicalUNet3DTo2D(backend=config["backend"], **options)
    if name not in ("residual", "residual_physical"):
        raise ValueError(name)
    return ResidualUNet3DTo2D(**options)


def validate_protocol(config):
    train_names = set(config["train"])
    if config["validation"] in train_names or config["test"] in train_names or config["test"] == config["validation"]:
        raise ValueError("Training, validation, and test segments must be disjoint")
    if config["backend"] != "fallback" or config["tta"]:
        raise ValueError("This protocol fixes the fallback backend and disables TTA for every model")


@torch.inference_mode()
def predict(model, slab, device, patch=128, stride=96, batch_size=8):
    model.eval()
    _, height, width = slab.shape
    if min(height, width) < patch:
        raise ValueError("Crop smaller than inference tile")
    ys = sorted(set(range(0, height-patch+1, stride)) | {height-patch})
    xs = sorted(set(range(0, width-patch+1, stride)) | {width-patch})
    coords = [(y, x) for y in ys for x in xs]
    axis = np.linspace(-1, 1, patch, dtype=np.float32)
    weights = np.exp(-2*(axis[:, None]**2 + axis[None, :]**2))
    total = np.zeros((height, width), np.float32)
    count = np.zeros_like(total)
    for index in range(0, len(coords), batch_size):
        batch_coords = coords[index:index+batch_size]
        x = torch.from_numpy(np.stack([slab[:, y:y+patch, xx:xx+patch][None] for y, xx in batch_coords])).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        probability = logits.float().sigmoid().squeeze(1).cpu().numpy()
        for (y, xx), values in zip(batch_coords, probability):
            total[y:y+patch, xx:xx+patch] += values * weights
            count[y:y+patch, xx:xx+patch] += weights
    if (count == 0).any():
        raise RuntimeError("Inference left uncovered pixels")
    return total / count


def load_item(data, name):
    volume = np.load(data / f"{name}_volume.npy", mmap_mode="r")
    slab, valid = prepare_slab(volume)
    labels = np.load(data / f"{name}_labels.npz")
    mask = labels["supervision"] & valid
    return slab, labels["ink"], mask


def train(config, data, out, device):
    if (out / "test_opened.json").exists():
        raise RuntimeError("Test already opened: use a new protocol/output for additional training")
    train_items = [load_item(data, name) for name in config["train"]]
    val_slab, val_ink, val_mask = load_item(data, config["validation"])
    centers = [np.column_stack(np.where(item[2])) for item in train_items]
    deadline = time.monotonic() + config["max_train_minutes"] * 60
    index = []
    for seed in config["seeds"]:
        for name in config["models"]:
            if time.monotonic() > deadline:
                raise TimeoutError("Training time budget exhausted")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            model = build_model(name, config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"], eta_min=1e-4)
            criterion = FableLoss(pos_weight=1.2, tversky_alpha=0.4, tversky_beta=0.6)
            size = config["patch_size"]
            best_ap = -1.0
            history = []
            started = time.monotonic()
            checkpoint = out / f"{name}_seed{seed}.ckpt"
            for step in range(1, config["steps"]+1):
                if time.monotonic() > deadline:
                    raise TimeoutError("Training time budget exhausted")
                model.train()
                volumes, inks, masks = [], [], []
                for _ in range(config["batch_size"]):
                    source = int(rng.integers(len(train_items)))
                    slab, ink, mask = train_items[source]
                    cy, cx = centers[source][int(rng.integers(len(centers[source])))]
                    y = int(np.clip(cy-size//2+rng.integers(-32,33), 0, slab.shape[1]-size))
                    x = int(np.clip(cx-size//2+rng.integers(-32,33), 0, slab.shape[2]-size))
                    v = slab[:, y:y+size, x:x+size]
                    i, m = ink[y:y+size, x:x+size], mask[y:y+size, x:x+size]
                    rotation = int(rng.integers(4))
                    v, i, m = np.rot90(v, rotation, (1,2)), np.rot90(i, rotation), np.rot90(m, rotation)
                    if rng.random() < 0.5:
                        v, i, m = v[:,:,::-1], i[:,::-1], m[:,::-1]
                    if rng.random() < 0.5:
                        v = v[::-1]
                    volumes.append(np.ascontiguousarray(v[None]))
                    inks.append(np.ascontiguousarray(i[None], dtype=np.float32))
                    masks.append(np.ascontiguousarray(m[None], dtype=np.float32))
                x_t = torch.from_numpy(np.stack(volumes)).to(device)
                y_t = torch.from_numpy(np.stack(inks)).to(device)
                m_t = torch.from_numpy(np.stack(masks)).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(x_t)
                    loss = criterion(logits, y_t, valid_mask=m_t)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if step % 25 == 0 or step == 1:
                    print(f"TRAIN {name} seed={seed} step={step} loss={loss.item():.5f} elapsed={time.monotonic()-started:.1f}s", flush=True)
                if step % config["validate_every"] == 0 or step == config["steps"]:
                    probability = predict(model, val_slab, device, size, config["inference_stride"])
                    ap, _, threshold = precision_recall_summary(val_ink[val_mask], probability[val_mask])
                    result = binary_report(val_ink[val_mask], probability[val_mask], threshold)
                    history.append({"step": step, **result})
                    if ap > best_ap:
                        best_ap = ap
                        negative = probability[val_mask & ~val_ink]
                        strict_threshold = float(np.quantile(negative, 0.99))
                        torch.save({"model_state": model.state_dict(), "model_config": model.get_config(),
                                    "seed": seed, "step": step, "validation_threshold": threshold,
                                    "validation_1pct_fpr_threshold": strict_threshold, "validation": result,
                                    "protocol": config}, checkpoint)
                        np.save(out / f"{name}_seed{seed}_validation.npy", probability)
                    print(f"VALIDATION {name} seed={seed} step={step} AP={ap:.5f} best={best_ap:.5f}", flush=True)
            write_json(out / f"{name}_seed{seed}_history.json", history)
            index.append({"model": name, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                          "checkpoint": checkpoint.name, "sha256": sha256(checkpoint),
                          "train_seconds": time.monotonic()-started})
            write_json(out / "frozen_checkpoints.json", index)
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_json(out / "training_complete.json", {"protocol": config["protocol"], "runs": len(index)})


def evaluate(config, data, out, device):
    frozen = json.loads((out / "frozen_checkpoints.json").read_text())
    if len(frozen) != len(config["seeds"]) * len(config["models"]):
        raise RuntimeError("All models must be frozen before opening test")
    if not (out / "training_complete.json").exists():
        raise RuntimeError("Training incomplete")
    write_json(out / "test_opened.json", {"at_unix": time.time(), "checkpoints": frozen})
    slab, ink, mask = load_item(data, config["test"])
    report = {"protocol": config, "test": config["test"], "runs": [], "baselines": {}}
    raw = np.asarray(np.load(data / f"{config['test']}_volume.npy", mmap_mode="r"), np.float32)
    for name, values in [("raw_mid_brightness", raw[raw.shape[0]//2]/255.0), ("raw_depth_mean", raw.mean(0)/255.0)]:
        report["baselines"][name] = binary_report(ink[mask], values[mask])
    for entry in frozen:
        checkpoint = out / entry["checkpoint"]
        if sha256(checkpoint) != entry["sha256"]:
            raise RuntimeError("Frozen checkpoint changed")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(entry["model"], config).to(device)
        model.load_state_dict(state["model_state"], strict=True)
        probability = predict(model, slab, device, config["patch_size"], config["inference_stride"])
        filename = f"{entry['model']}_seed{entry['seed']}_test.npy"
        np.save(out / filename, probability)
        result = {**entry, "validation": state["validation"],
                  "test_at_validation_f1_threshold": binary_report(ink[mask], probability[mask], state["validation_threshold"]),
                  "test_at_validation_1pct_fpr": binary_report(ink[mask], probability[mask], state["validation_1pct_fpr_threshold"]),
                  "prediction": filename}
        report["runs"].append(result)
        write_json(out / "metrics.json", report)
        print("TEST", entry["model"], entry["seed"], json.dumps(result["test_at_validation_f1_threshold"]), flush=True)
        del model
    np.savez_compressed(out / "test_labels.npz", ink=ink, supervision=mask, surface=raw[raw.shape[0]//2])
    return report


def scan(config, data, out, device):
    report = json.loads((out / "metrics.json").read_text())
    # The deployment choice is made using validation only, never the opened test.
    scores = {name: np.mean([r["validation"]["ap"] for r in report["runs"] if r["model"] == name]) for name in config["models"]}
    selected = max(scores, key=scores.get)
    volume = np.load(data / "p800_volume.npy", mmap_mode="r")
    predictions = {}
    for offset in config["p800_depth_offsets"]:
        slab, valid = prepare_slab(volume, offset)
        values = []
        for run in report["runs"]:
            if run["model"] != selected:
                continue
            state = torch.load(out / run["checkpoint"], map_location="cpu", weights_only=False)
            model = build_model(selected, config).to(device)
            model.load_state_dict(state["model_state"], strict=True)
            values.append(predict(model, slab, device, config["patch_size"], config["inference_stride"]))
            del model
        predictions[f"z{offset:+d}"] = np.mean(values, axis=0)
    predictions["surface"] = np.asarray(volume[volume.shape[0]//2])
    predictions["valid"] = valid
    np.savez_compressed(out / "p800_depth_sweep.npz", **predictions)
    write_json(out / "p800_summary.json", {"selected_by_validation": selected, "validation_ap_means": scores,
        "segment": config["p800_segment"], "window_chunks": config["p800_window"], "native_um": 8.64,
        "depth_offsets": config["p800_depth_offsets"], "interpretation": config["p800_policy"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/controlled_ink_20261025.json")
    parser.add_argument("--data-dir", default="data/controlled_ink_20261025")
    parser.add_argument("--out-dir", default="runs/controlled_ink_20261025")
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate", "scan", "run", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_protocol(config)
    data, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous_protocol = out / "protocol.json"
    if previous_protocol.exists() and json.loads(previous_protocol.read_text()) != config:
        raise RuntimeError("Existing output belongs to a different frozen protocol")
    write_json(previous_protocol, config)
    device = torch.device(args.device)
    if args.stage != "prepare":
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but unavailable")
        write_json(out / "runtime.json", {
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256(path)
                for path in [Path(__file__).resolve(), Path(__file__).resolve().parents[1]/"vesuvius_code/baselines.py",
                             Path(__file__).resolve().parents[1]/"vesuvius_code/losses.py",
                             Path(__file__).resolve().parents[1]/"vesuvius_code/numpy_metrics.py"]},
        })
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.stage in ("prepare", "all"):
        prepare(config, data, Path(__file__).resolve().parents[1])
    if args.stage in ("train", "run", "all"):
        train(config, data, out, device)
    if args.stage in ("evaluate", "run", "all"):
        evaluate(config, data, out, device)
    if args.stage in ("scan", "run", "all"):
        scan(config, data, out, device)
    print("STAGE_COMPLETE", args.stage, flush=True)


if __name__ == "__main__":
    main()
