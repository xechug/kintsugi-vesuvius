"""Prepare independently selected native crops, human masks and geometry audit."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
from scipy.spatial import cKDTree

from scripts.reader_v3_survey import REPO, sha256, write_json

DATA = REPO / "data/reader_kintsugi_20261025"
CACHE = REPO / "data/reader_kintsugi_sources_20261025"


def best_window(mask, size, min_x=0, max_x=None):
    height, width = mask.shape
    max_x = width if max_x is None else max_x
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = sorted(set(range(0, height - size + 1, 128)) | {height - size})
    xs = sorted(set(range(min_x, max_x - size + 1, 128)) | {max_x - size})
    xs = [x for x in xs if x >= min_x and x + size <= max_x]
    candidates = [(int(integral[y+size, x+size] - integral[y, x+size] - integral[y+size, x] + integral[y, x]), y, x)
                  for y in ys for x in xs]
    if not candidates:
        raise ValueError("No window satisfies the spatial split")
    count, y, x = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
    if count < 30000:
        raise ValueError("Too few reviewed pixels in the selected window")
    return {"y0": y, "x0": x, "height": size, "width": size, "reviewed_pixels": count}


def gray(path):
    return np.asarray(Image.open(path).convert("L"))


def geometry(winding):
    return np.stack([tifffile.imread(CACHE / winding / f"tifxyz/{axis}.tif") for axis in "xyz"], -1)


def region_slice(region):
    y, x, h, w = (region[key] for key in ("y0", "x0", "height", "width"))
    return np.s_[y:y+h, x:x+w]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    config = json.loads((REPO / "configs/reader_kintsugi_20261025.json").read_text())
    certainty = {w: gray(CACHE / w / "labels/mask.png") > 0 for w in ("w058", "w060")}
    counts = certainty["w058"].sum(0)
    cut = int(np.searchsorted(counts.cumsum(), counts.sum() // 2))
    train_region = best_window(certainty["w058"], 1536, max_x=cut - 384)
    val_region = best_window(certainty["w058"], 1536, min_x=cut + 384)
    xyz_train_full = geometry("w058")
    ys, xs = region_slice(train_region)
    xyz_train = xyz_train_full[ys.start//4:ys.stop//4, xs.start//4:xs.stop//4].reshape(-1, 3)
    xyz_train = xyz_train[np.isfinite(xyz_train).all(1) & (xyz_train > 0).all(1)]
    tree = cKDTree(xyz_train)
    xyz_test = geometry("w060")
    distance, _ = tree.query(xyz_test.reshape(-1, 3), workers=4)
    distance = distance.reshape(xyz_test.shape[:2])
    safe = np.isfinite(xyz_test).all(-1) & (xyz_test > 0).all(-1) & (distance > 20)
    safe = np.repeat(np.repeat(safe, 4, 0), 4, 1)
    safe = safe[:certainty["w060"].shape[0], :certainty["w060"].shape[1]]
    test_region = best_window(certainty["w060"] & safe, 2048)
    regions = {"train_1447_w058": train_region, "val_1447_w058": val_region,
               "test_1447_w060": test_region}
    geometry_audit = {"minimum_allowed_surface_distance_voxels": 20,
        "native_um": 8.64, "training_geometry_points": len(xyz_train),
        "w060_reviewed_before": int(certainty["w060"].sum()),
        "w060_reviewed_after_geometry_exclusion": int((certainty["w060"] & safe).sum()),
        "w058_train_val_x_gap": val_region["x0"] - train_region["x0"] - train_region["width"],
        "policy": "Nearest source-surface vertices at 4-pixel mesh spacing; 20-voxel separation includes margin around two 17-layer slabs. Conservative exclusion, not a full volumetric overlap proof."}
    for index, x in enumerate((1920, 4864)):
        regions[f"search_1447_w062_{index:02d}"] = {"x0": x, "y0": 1280, "height": 2048, "width": 2048}
    # Commit all window choices before loading ink-label intensities or model scores.
    write_json(DATA / "regions.json", regions)
    write_json(DATA / "geometry_audit.json", geometry_audit)
    write_json(DATA / "protocol.json", config)
    print("REGIONS_FROZEN", json.dumps(regions), flush=True)
    for winding in ("w058", "w060", "w062"):
        selected = {name: r for name, r in regions.items() if winding in name}
        arrays = {name: np.empty((24, r["height"], r["width"]), np.uint8) for name, r in selected.items()}
        for layer in range(24):
            image = tifffile.imread(CACHE / winding / f"layers/{layer:02d}.tif")
            if image.dtype != np.uint8 or image.ndim != 2:
                raise ValueError("Native CT must remain grayscale uint8")
            for name, region in selected.items():
                arrays[name][layer] = image[region_slice(region)]
        for name, volume in arrays.items():
            np.save(DATA / f"{name}.npy", volume)
            if not name.startswith("search"):
                mask = certainty[winding].copy()
                if winding == "w060":
                    mask &= safe
                ink = gray(CACHE / winding / "labels/inklabels.png")
                crop = region_slice(regions[name])
                np.savez_compressed(DATA / f"{name}_labels.npz", ink=ink[crop], supervision=mask[crop])
        print("PREPARED", winding, flush=True)
    previous = REPO / "data/controlled_ink_20261025"
    for winding in ("w035", "w040"):
        shutil.copyfile(previous / f"{winding}_volume.npy", DATA / f"train_0139_{winding}.npy")
        original = np.load(previous / f"{winding}_labels.npz")
        np.savez_compressed(DATA / f"train_0139_{winding}_labels.npz",
            ink=original["ink"].astype(np.uint8) * 255, supervision=original["supervision"])
    reader = REPO / "data/reader_v3_survey_20261025"
    shutil.copytree(reader / "official", DATA / "official", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(reader / "reader-v3.pth", DATA / "reader-v3.pth")
    sources = {"dataset_repository": config["dataset_repository"], "revision": config["dataset_revision"],
        "license": "CC-BY-NC-4.0", "author": "Youssef Nader", "source_readme": (CACHE / "README.md").read_text(),
        "original_checksums": (CACHE / "SHA256SUMS").read_text(),
        "human_label_rule": "ink > 127 inside nonzero human certainty mask; no published model maps used as targets"}
    write_json(DATA / "sources.json", sources)
    files = [path for path in DATA.rglob("*") if path.is_file() and path.name != "hashes.json" and "__pycache__" not in path.parts]
    write_json(DATA / "hashes.json", {str(path.relative_to(DATA)): sha256(path) for path in sorted(files)})
    print("READER_KINTSUGI_INPUTS_READY", flush=True)


if __name__ == "__main__":
    main()
