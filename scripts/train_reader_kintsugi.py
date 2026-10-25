"""Human-supervised Kintsugi and frozen-Reader correction with a sealed winding test."""
from __future__ import annotations

import json
import random
import time

import numpy as np
import torch

from scripts.reader_v3_survey import REPO, infer_volume, load_official, sha256, write_json
from vesuvius_code.losses import AnisotropicFiberSuppressionLoss, FableLoss
from vesuvius_code.numpy_metrics import binary_report
from vesuvius_code.reader_correction import build_correction_model

STUDY = "reader_kintsugi_20261025"
DATA = REPO / "data" / STUDY
OUT = REPO / "runs" / STUDY


def labels(name):
    value = np.load(DATA / f"{name}_labels.npz")
    return value["ink"] > 127, value["supervision"].astype(bool)


def report_at_recall(ink, mask, scores, recall=0.8):
    positives = scores[mask & ink]
    if not len(positives):
        raise ValueError("No reviewed ink for setting a recall threshold")
    threshold = float(np.quantile(positives, 1 - recall))
    result = binary_report(ink[mask], scores[mask], threshold)
    result["false_positive_rate"] = result["fp"] / max(1, result["pixels"] - result["positive_pixels"])
    return result


def oriented_slab(name, direction):
    volume = np.load(DATA / f"{name}.npy", mmap_mode="r")
    start = volume.shape[0] // 2 - 8
    slab = np.asarray(volume[start:start+17], np.float32) / 255
    if direction == "reverse":
        slab = slab[::-1].copy()
    valid = slab.max(0) > 0
    return slab, valid


@torch.inference_mode()
def predict(model, name, slab, reader_scores, device, patch=128, stride=96):
    model.eval()
    height, width = reader_scores.shape
    ys = sorted(set(range(0, height-patch+1, stride)) | {height-patch})
    xs = sorted(set(range(0, width-patch+1, stride)) | {width-patch})
    positions = [(y, x) for y in ys for x in xs]
    weight = np.maximum(np.outer(np.hanning(patch), np.hanning(patch)), 1e-3).astype(np.float32)
    total, count = np.zeros((height, width), np.float32), np.zeros((height, width), np.float32)
    for start in range(0, len(positions), 8):
        coords = positions[start:start+8]
        volume = torch.from_numpy(np.stack([slab[:, y:y+patch, x:x+patch][None] for y, x in coords])).to(device)
        scores = torch.from_numpy(np.stack([reader_scores[y:y+patch, x:x+patch][None] for y, x in coords])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(volume) if name == "kintsugi_alone" else model(volume, scores)
        output = logits.float().sigmoid().squeeze(1).cpu().numpy()
        for (y, x), array in zip(coords, output):
            total[y:y+patch, x:x+patch] += array * weight
            count[y:y+patch, x:x+patch] += weight
    return total / np.maximum(count, 1e-9)


def train_one(name, seed, config, train_items, val_item, device, deadline):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = build_correction_model(name, config["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"], eta_min=1e-5)
    base_loss = FableLoss(pos_weight=1, tversky_alpha=0.6, tversky_beta=0.4)
    criterion = AnisotropicFiberSuppressionLoss(base_loss,
        fiber_weight=config["fiber_penalty_weight"] if name == "reader_kintsugi_fiber" else 0).to(device)
    pools = [[np.argwhere(mask & (ink == positive)) for positive in (False, True)]
             for _, _, ink, mask in train_items]
    if any(not len(pool) for item in pools for pool in item):
        raise ValueError("Every training crop must contain reviewed ink and background")
    best_ap, history = -1, []
    checkpoint = OUT / f"{name}_seed{seed}.ckpt"
    if (OUT / f"{name}_seed{seed}_history.json").exists() and checkpoint.exists() and (OUT / f"{name}_seed{seed}_validation.npy").exists():
        print("RESUMED_COMPLETED", name, seed, flush=True)
        return {"model": name, "seed": seed, "checkpoint": checkpoint.name,
            "sha256": sha256(checkpoint), "parameters": sum(p.numel() for p in model.parameters()),
            "train_seconds": 0.0}
    size, started = config["patch_size"], time.monotonic()
    val_slab, val_reader, val_ink, val_mask = val_item
    for step in range(1, config["steps"]+1):
        if time.monotonic() > deadline:
            raise TimeoutError("Training time guard reached")
        model.train(); volumes, readers, inks, masks = [], [], [], []
        for _ in range(config["batch_size"]):
            source = int(rng.integers(len(train_items)))
            slab, scores, ink, mask = train_items[source]
            pool = pools[source][int(rng.integers(2))]
            cy, cx = pool[int(rng.integers(len(pool)))]
            y = int(np.clip(cy-size//2+rng.integers(-32, 33), 0, ink.shape[0]-size))
            x = int(np.clip(cx-size//2+rng.integers(-32, 33), 0, ink.shape[1]-size))
            arrays = [slab[:, y:y+size, x:x+size], scores[y:y+size, x:x+size][None],
                      ink[y:y+size, x:x+size][None], mask[y:y+size, x:x+size][None]]
            rotation, flip = int(rng.integers(4)), rng.random() < 0.5
            for target, array in zip((volumes, readers, inks, masks), arrays):
                array = np.rot90(array, rotation, (-2, -1))
                if flip:
                    array = array[..., ::-1]
                target.append(np.ascontiguousarray(array, dtype=np.float32))
        volume = torch.from_numpy(np.stack(volumes)[:, None]).to(device)
        scores = torch.from_numpy(np.stack(readers)).to(device)
        target = torch.from_numpy(np.stack(inks)).to(device)
        mask = torch.from_numpy(np.stack(masks)).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(volume) if name == "kintsugi_alone" else model(volume, scores)
            loss = criterion(logits, target, valid_mask=mask, volume=volume)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite supervised loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        if step == 1 or step % 100 == 0:
            print(f"TRAIN {name} seed={seed} step={step} loss={loss.item():.5f}", flush=True)
        if step % config["validate_every"] == 0 or step == config["steps"]:
            probability = predict(model, name, val_slab, val_reader, device,
                                   size, config["inference_stride"])
            result = report_at_recall(val_ink, val_mask, probability, config["target_validation_recall"])
            history.append({"step": step, **result})
            if result["ap"] > best_ap:
                best_ap = result["ap"]
                torch.save({"state": model.state_dict(), "model": name, "seed": seed,
                    "step": step, "validation": result, "protocol": config,
                    "reader_sha256": sha256(DATA / "reader-v3.pth")}, checkpoint)
                np.save(OUT / f"{name}_seed{seed}_validation.npy", probability)
            print("VALIDATION", name, seed, step, "AP", result["ap"], flush=True)
    write_json(OUT / f"{name}_seed{seed}_history.json", history)
    frozen = {"model": name, "seed": seed, "checkpoint": checkpoint.name,
        "sha256": sha256(checkpoint), "parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.monotonic() - started}
    del model, optimizer
    torch.cuda.empty_cache()
    return frozen


def main():
    if (OUT / "test_opened.json").exists() or (OUT / "artifacts.json").exists():
        raise RuntimeError("This experiment has already opened its test; use a new protocol")
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((DATA / "protocol.json").read_text())
    for relative, digest in json.loads((DATA / "hashes.json").read_text()).items():
        if sha256(DATA / relative) != digest:
            raise ValueError("Input checksum mismatch: " + relative)
    if not torch.cuda.is_available():
        raise RuntimeError("This worker requires CUDA")
    device = torch.device("cuda"); torch.set_num_threads(8)
    official = load_official(DATA / "official")
    official.configure_logging("WARNING")
    args = official.parse_args(["unused", str(DATA / "reader-v3.pth"), "unused.tif", "--no-compile"])
    bundle = official.configure_model(args)
    payload = torch.load(DATA / "reader-v3.pth", map_location="cpu", weights_only=False)
    bundle.model.model.load_state_dict(payload["model"], strict=True)
    bundle.model = bundle.model.to(device).eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    del payload
    reader_config = {"overlap": 0.5, "blend_mode": "hann", "batch_size": 4}
    direction_scores = {}
    train_name = "train_1447_w058"
    ink, mask = labels(train_name)
    for direction in ("forward", "reverse"):
        orient_path = OUT / f"orientation_train_w058_{direction}.npy"
        if orient_path.exists():
            scores = np.load(orient_path)
        else:
            scores, _ = infer_volume(official, bundle, np.load(DATA / f"{train_name}.npy", mmap_mode="r"),
                                     device, reader_config, direction)
            np.save(orient_path, scores)
        direction_scores[direction] = report_at_recall(ink, mask, scores)
    direction_1447 = max(direction_scores, key=lambda d: direction_scores[d]["ap"])
    orientations = {name: "forward" if "0139" in name else direction_1447
                    for name in config["train"] + [config["validation"], config["test"]] + config["search"]}
    write_json(OUT / "orientation.json", {"training_only_metrics": direction_scores,
        "chosen_for_1447": direction_1447, "directions": orientations})
    print("FROZEN_DIRECTION_1447", direction_1447, flush=True)

    def reader_predict(name, direction=None):
        direction = orientations[name] if direction is None else direction
        path = OUT / f"reader_{name}_{direction}.npy"
        if not path.exists():
            result, _ = infer_volume(official, bundle, np.load(DATA / f"{name}.npy", mmap_mode="r"),
                                     device, reader_config, direction)
            np.save(path, result)
        return np.load(path)

    items = {}
    for name in config["train"] + [config["validation"]]:
        slab, valid = oriented_slab(name, orientations[name])
        ink, mask = labels(name)
        items[name] = slab, reader_predict(name), ink, mask & valid
    val_slab, val_reader, val_ink, val_mask = items[config["validation"]]
    baseline_val = report_at_recall(val_ink, val_mask, val_reader, config["target_validation_recall"])
    summary = {"protocol": config, "reader_sha256": sha256(DATA / "reader-v3.pth"),
        "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                    "training_precision": "bf16", "reader_precision": str(bundle.amp_dtype)},
        "baseline_validation": baseline_val, "runs": [], "started_at": time.time()}
    write_json(OUT / "summary.json", summary)
    frozen, deadline = [], time.monotonic() + config["max_train_minutes"] * 60
    for seed in config["seeds"]:
        for name in config["models"]:
            frozen.append(train_one(name, seed, config, [items[n] for n in config["train"]],
                                    items[config["validation"]], device, deadline))
            write_json(OUT / "frozen_checkpoints.json", frozen)
    write_json(OUT / "training_complete.json", {"at_unix": time.time(), "runs": len(frozen)})
    ensembles = {}
    for name in config["models"]:
        mean_map = np.mean([np.load(OUT / f"{name}_seed{s}_validation.npy") for s in config["seeds"]], axis=0)
        np.save(OUT / f"{name}_ensemble_validation.npy", mean_map)
        ensembles[name] = report_at_recall(val_ink, val_mask, mean_map, config["target_validation_recall"])
    physics_names = [n for n in config["models"] if n.startswith("reader_kintsugi")]
    exploration_model = max(physics_names, key=lambda n: ensembles[n]["ap"])
    qualified = [n for n in config["models"] if n != "kintsugi_alone"
        and ensembles[n]["ap"] >= baseline_val["ap"] + 0.01
        and ensembles[n]["precision"] >= baseline_val["precision"] + 0.03]
    candidate = max(qualified, key=lambda n: ensembles[n]["ap"]) if qualified else None
    selection = {"candidate": candidate, "exploratory_physical_model": exploration_model,
                 "ensemble_validation": ensembles, "test_policy": "Candidate fixed using validation only, before test labels or predictions are loaded."}
    write_json(OUT / "model_selection.json", selection)
    write_json(OUT / "test_opened.json", {"at_unix": time.time(), "frozen": frozen, "selection": selection})
    # Only now open the held-out winding and its labels.
    test_name = config["test"]
    test_reader = reader_predict(test_name)
    test_slab, valid = oriented_slab(test_name, orientations[test_name])
    test_ink, test_mask = labels(test_name); test_mask &= valid
    baseline_test = binary_report(test_ink[test_mask], test_reader[test_mask], baseline_val["threshold"])
    summary["baseline_test"] = baseline_test
    test_maps = {name: [] for name in config["models"]}
    for entry in frozen:
        if sha256(OUT / entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("Frozen checkpoint changed")
        saved = torch.load(OUT / entry["checkpoint"], map_location="cpu", weights_only=False)
        model = build_correction_model(entry["model"], config["base_channels"]).to(device)
        model.load_state_dict(saved["state"], strict=True)
        scores = predict(model, entry["model"], test_slab, test_reader, device,
                          config["patch_size"], config["inference_stride"])
        filename = f"{entry['model']}_seed{entry['seed']}_test.npy"
        np.save(OUT / filename, scores)
        test_maps[entry["model"]].append(scores)
        summary["runs"].append({**entry, "selected_step": saved["step"],
            "validation": saved["validation"], "test": binary_report(test_ink[test_mask], scores[test_mask], saved["validation"]["threshold"]),
            "prediction": filename})
        write_json(OUT / "summary.json", summary)
        print("TEST", entry["model"], entry["seed"], summary["runs"][-1]["test"]["ap"], flush=True)
        del model, saved
        torch.cuda.empty_cache()
    summary["ensemble_test"] = {}
    for name, maps in test_maps.items():
        scores = np.mean(maps, axis=0)
        np.save(OUT / f"{name}_ensemble_test.npy", scores)
        summary["ensemble_test"][name] = binary_report(test_ink[test_mask], scores[test_mask], ensembles[name]["threshold"])
    passed = False
    if candidate:
        result = summary["ensemble_test"][candidate]
        passed = (result["ap"] > baseline_test["ap"] and result["precision"] > baseline_test["precision"]
                  and result["recall"] >= baseline_test["recall"] - 0.03)
    summary["promoted_model"] = candidate if passed else None
    summary["model_selection"] = selection
    write_json(OUT / "summary.json", summary)
    print("PROMOTED_MODEL", summary["promoted_model"], flush=True)
    del items, test_maps

    # Explore both surface directions, keeping all outputs separate. A failed
    # model remains explicitly experimental; it is never sold as a precision gain.
    search_name = exploration_model
    summary["search"] = []
    for name in config["search"]:
        for direction in ("forward", "reverse"):
            base = reader_predict(name, direction)
            slab, _ = oriented_slab(name, direction)
            predictions = []
            for seed in config["seeds"]:
                state = torch.load(OUT / f"{search_name}_seed{seed}.ckpt", map_location="cpu", weights_only=False)
                model = build_correction_model(search_name, config["base_channels"]).to(device)
                model.load_state_dict(state["state"], strict=True)
                predictions.append(predict(model, search_name, slab, base, device,
                                            config["patch_size"], config["inference_stride"]))
                del model, state
            filename = f"{search_name}_{name}_{direction}.npy"
            np.save(OUT / filename, np.mean(predictions, axis=0))
            summary["search"].append({"item": name, "direction": direction,
                "model": search_name, "prediction": filename,
                "status": "validated_gain" if summary["promoted_model"] == search_name else "experimental_not_promoted"})
            write_json(OUT / "summary.json", summary)
            print("SEARCH", name, direction, flush=True)
    summary["finished_at"] = time.time()
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "protocol.json", config)
    write_json(OUT / "artifacts.json", {p.name: sha256(p) for p in sorted(OUT.iterdir())
        if p.suffix in (".npy", ".json", ".ckpt") and p.name not in ("runpod_state.json", "worker_status.json", "artifacts.json")})
    print("READER_KINTSUGI_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
