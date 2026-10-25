"""Native-resolution Reader v3 survey using pinned, unmodified Villa inference helpers.

The known-letter control is in Reader's training set: it tests the inference path,
not generalization. No training, probability calibration, OCR, or discovery claims.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STUDY = "reader_v3_survey_20261025"
DATA = REPO / "data" / STUDY
OUT = REPO / "runs" / STUDY
S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_url(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404 or attempt == 3:
                raise
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(attempt + 1)


def download_native(url, destination):
    """Decode every native Zarr-v2 chunk; retain declared fill for absent chunks.

    A 404 chunk denotes the Zarr fill value, never a failed request silently
    replaced by zero. All other failures abort preparation. Edge chunks are
    clipped to the array shape after full-chunk decoding.
    """
    destination = Path(destination)
    source_path = destination.with_suffix(".source.json")
    if destination.exists() and source_path.exists():
        source = json.loads(source_path.read_text())
        if sha256(destination) != source["sha256"]:
            raise ValueError(f"Cached input checksum mismatch: {destination}")
        return source
    metadata = json.loads(read_url(url + "/0/.zarray"))
    if (metadata["dtype"] != "|u1" or metadata["order"] != "C"
            or metadata.get("filters") or metadata["chunks"][0] < metadata["shape"][0]):
        raise ValueError(f"Unsupported native volume contract: {metadata}")
    if np.prod(metadata["shape"]) > 500_000_000:
        raise ValueError("Input exceeds this small survey's per-volume limit")
    depth, height, width = metadata["shape"]
    dz, dy, dx = metadata["chunks"]
    codec = None
    if metadata.get("compressor"):
        import numcodecs
        codec = numcodecs.get_codec(metadata["compressor"])
    separator = metadata.get("dimension_separator", ".")
    tasks = [(cy, cx) for cy in range((height + dy - 1) // dy)
             for cx in range((width + dx - 1) // dx)]
    temporary = destination.with_suffix(".partial.npy")
    volume = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                       shape=(depth, height, width))
    volume[:] = metadata.get("fill_value", 0) or 0

    def fetch(coord):
        key = separator.join(map(str, (0, *coord)))
        try:
            raw = read_url(url + "/0/" + key)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            return coord, None, 0
        nbytes = len(raw)
        decoded = codec.decode(raw) if codec else raw
        chunk = np.frombuffer(decoded, np.uint8).reshape(dz, dy, dx)[:depth]
        return coord, chunk, nbytes

    missing, received = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for (cy, cx), chunk, nbytes in pool.map(fetch, tasks):
            received += nbytes
            if chunk is None:
                missing.append([cy, cx])
            else:
                y, x = cy * dy, cx * dx
                h, w = min(dy, height - y), min(dx, width - x)
                volume[:, y:y+h, x:x+w] = chunk[:, :h, :w]
    volume.flush()
    del volume
    temporary.replace(destination)
    try:
        attrs = json.loads(read_url(url + "/.zattrs"))
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        attrs = {}
    source = {"url": url, "level": 0, "metadata": metadata, "attributes": attrs,
              "chunks_total": len(tasks), "absent_fill_chunks": missing,
              "download_bytes": received, "sha256": sha256(destination)}
    write_json(source_path, source)
    print(f"PREPARED {destination.name} shape={metadata['shape']} fill_chunks={len(missing)}", flush=True)
    return source


def prepare(config):
    DATA.mkdir(parents=True, exist_ok=True)
    old = REPO / "data/controlled_ink_20261025"
    vendor = DATA / "official"
    shutil.copytree(old / "official", vendor, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.pth", "__pycache__", "*.pyc"))
    inference = vendor / "ink-detection/koine_machines/inference"
    inference.mkdir(exist_ok=True)
    (inference / "__init__.py").touch()
    infer_url = ("https://raw.githubusercontent.com/ScrollPrize/villa/"
                 + config["villa_runtime_commit"] + "/ink-detection/koine_machines/inference/infer.py")
    (inference / "infer.py").write_bytes(read_url(infer_url))
    for name in ("w039_volume.npy", "w039_labels.npz", "w039_volume.source.json"):
        shutil.copyfile(old / name, DATA / name)
    if not (DATA / "reader-v3.pth").is_file():
        model_url = ("https://huggingface.co/" + config["model_repository"] + "/resolve/"
                     + config["model_revision"] + "/reader-v3.pth")
        (DATA / "reader-v3.pth").write_bytes(read_url(model_url))
    sources = []
    for index, segment in enumerate(config["segments"]):
        url = f"{S3}/{config['scroll']}/segments/{segment}/surface-volumes/{config['surface_volume']}"
        source = download_native(url, DATA / f"p800_{index:02d}.npy")
        source["segment"] = segment
        sources.append(source)
    write_json(DATA / "sources.json", sources)
    write_json(DATA / "protocol.json", config)
    hashes = {str(path.relative_to(DATA)): sha256(path) for path in sorted(DATA.rglob("*"))
              if path.is_file() and path.name != "hashes.json" and "__pycache__" not in path.parts}
    write_json(DATA / "hashes.json", hashes)
    print("READER_INPUTS_READY", flush=True)


def load_official(vendor=None):
    vendor = Path(vendor) if vendor is not None else DATA / "official"
    sys.path[:0] = [str(vendor / "ink-detection"), str(vendor / "vesuvius/src")]
    # Avoid eager initialization of the unrelated remote-volume SDK. All model,
    # normalization and inference source files are original pinned Villa files.
    package = types.ModuleType("vesuvius")
    package.__path__ = [str(vendor / "vesuvius/src/vesuvius")]
    sys.modules["vesuvius"] = package
    from koine_machines.inference import infer
    return infer


class NumpyAccumulator:
    """Float32 output adapter for the original Villa inference loop."""
    def __init__(self, shape):
        self.total = np.zeros(shape, np.float32)
        self.weight = np.zeros(shape, np.float32)

    def add_tile(self, *, y0, x0, tile, tile_weights):
        height, width = tile.shape
        self.total[y0:y0+height, x0:x0+width] += tile * tile_weights
        self.weight[y0:y0+height, x0:x0+width] += tile_weights

    def result(self):
        return np.divide(self.total, self.weight, out=np.zeros_like(self.total),
                         where=self.weight > 0)


def infer_volume(official, bundle, volume, device, config, direction, offset=0, tta=False):
    from torch.utils.data import DataLoader
    indices = official.center_crop_layer_indices(np.arange(volume.shape[0]),
                                                  output_depth=bundle.in_chans) + offset
    if indices.min() < 0 or indices.max() >= volume.shape[0]:
        raise ValueError("Requested depth window exceeds native slices")
    if direction not in ("forward", "reverse"):
        raise ValueError("Unknown layer direction")
    if direction == "reverse":
        indices = indices[::-1]
    height, width = volume.shape[1:]
    reader = official.OmeZarrPatchReader(input_path="cached_native_array", resolution="0",
        depth_axis_first=True, height=height, width=width, layer_indices=indices,
        output_depth=bundle.in_chans, preprocessing=bundle.preprocessing)
    reader._array = volume  # Same array interface; no resampling or intensity edits.
    size = bundle.roi_size
    stride = official.resolve_patch_stride(patch_size=size, overlap=config["overlap"], explicit_stride=None)
    blocks = official.iter_blocks(image_shape=(height, width), patch_size=size, stride=stride,
                                  mask_lowres=None, occupancy_scale=(1, 1))
    dataset = official.OmeZarrBlockDataset(reader=reader, blocks=blocks, patch_size=size,
                                          preprocessing=bundle.preprocessing)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                         num_workers=0, pin_memory=device.type == "cuda")
    weight = official.compute_importance_map_2d(patch_size=(size, size),
                    mode=config["blend_mode"], sigma_scale=0.125).numpy()
    accumulator = NumpyAccumulator((height, width))
    axes = official.compute_equal_length_mirror_axes((bundle.in_chans, size, size)) if tta else ()
    official.run_block_inference(loader=loader, model=bundle.model, accumulator=accumulator,
        weight_map=weight, mask=None, device=device, amp=True, amp_dtype=bundle.amp_dtype,
        tta_axes=axes, tta_batch_size=1)
    result = accumulator.result()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite model output")
    return result, indices.tolist()


def run(config, device_name="cuda"):
    import torch
    from vesuvius_code.numpy_metrics import binary_report

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "artifacts.json").exists():
        raise RuntimeError("Completed survey already exists; do not overwrite it")
    for relative, expected in json.loads((DATA / "hashes.json").read_text()).items():
        if sha256(DATA / relative) != expected:
            raise ValueError(f"Input checksum mismatch: {relative}")
    official = load_official()
    official.configure_logging("INFO")
    checkpoint = DATA / "reader-v3.pth"
    args = official.parse_args(["unused", str(checkpoint), "unused.tif", "--no-compile"])
    bundle = official.configure_model(args)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Villa logs incompatibilities but tolerates them. This survey requires an exact load.
    bundle.model.model.load_state_dict(payload["model"], strict=True)
    if bundle.roi_size != config["patch_size"] or bundle.in_chans != 17 or bundle.preprocessing != "tifxyz_robust":
        raise ValueError("Checkpoint architecture/preprocessing differs from the frozen protocol")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the rented inference worker")
    torch.set_num_threads(8)
    bundle.model = bundle.model.to(device).eval()
    metadata = {"protocol": config, "model_sha256": sha256(checkpoint),
        "checkpoint_config": payload["config"], "checkpoint_step": payload["step"],
        "strict_state_load": True, "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "amp_dtype": str(bundle.amp_dtype), "parameters": sum(p.numel() for p in bundle.model.parameters()),
        "control": [], "survey": [], "started_at": time.time()}
    write_json(OUT / "protocol.json", config)
    del payload
    control = np.load(DATA / "w039_volume.npy", mmap_mode="r")
    labels = np.load(DATA / "w039_labels.npz")
    reviewed = labels["supervision"] > 0
    for direction in config["directions"]:
        started = time.monotonic()
        scores, indices = infer_volume(official, bundle, control, device, config, direction)
        name = f"control_w039_{direction}.npy"
        np.save(OUT / name, scores)
        report = binary_report(labels["ink"][reviewed], scores[reviewed])
        metadata["control"].append({"direction": direction, "prediction": name,
            "layers": indices, "metrics": report, "seconds": time.monotonic() - started})
        write_json(OUT / "summary.json", metadata)
        print("CONTROL", direction, json.dumps(report), flush=True)
    if max(item["metrics"]["ap"] for item in metadata["control"]) < config["control_min_ap"]:
        raise RuntimeError("Known-letter control failed: stop before interpreting unseen-scroll scores")
    for index, segment in enumerate(config["segments"]):
        volume = np.load(DATA / f"p800_{index:02d}.npy", mmap_mode="r")
        valid = volume[volume.shape[0] // 2] > 0
        for direction in config["directions"]:
            started = time.monotonic()
            scores, indices = infer_volume(official, bundle, volume, device, config, direction)
            name = f"p800_{index:02d}_{direction}.npy"
            np.save(OUT / name, scores)
            metadata["survey"].append({"segment": segment, "index": index,
                "direction": direction, "prediction": name, "layers": indices,
                "shape": list(scores.shape), "nonzero_surface_pixels": int(valid.sum()),
                "score_percentiles": np.percentile(scores[valid], [0, 25, 50, 75, 90, 99, 99.5, 100]).tolist(),
                "seconds": time.monotonic() - started})
            write_json(OUT / "summary.json", metadata)
            print("SURVEY", index, direction, "seconds", round(time.monotonic() - started, 2), flush=True)
    metadata["finished_at"] = time.time()
    write_json(OUT / "summary.json", metadata)
    artifacts = {path.name: sha256(path) for path in sorted(OUT.iterdir())
                 if path.suffix in (".npy", ".json") and path.name not in ("runpod_state.json", "worker_status.json")}
    write_json(OUT / "artifacts.json", artifacts)
    print("READER_SURVEY_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = json.loads((REPO / f"configs/{STUDY}.json").read_text())
    if args.stage == "prepare":
        prepare(protocol)
    else:
        run(protocol, args.device)
