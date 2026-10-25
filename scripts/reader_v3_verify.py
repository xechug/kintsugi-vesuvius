"""Check the strongest P800 cluster with depth shifts and mirror TTA, locally."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from scripts.reader_v3_survey import DATA, OUT, infer_volume, load_official, sha256, write_json


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    config = summary["protocol"]
    official = load_official()
    official.configure_logging("WARNING")
    checkpoint = DATA / "reader-v3.pth"
    args = official.parse_args(["unused", str(checkpoint), "unused.tif", "--no-compile"])
    bundle = official.configure_model(args)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bundle.model.model.load_state_dict(payload["model"], strict=True)
    bundle.model.eval()
    torch.set_num_threads(8)
    # Chosen for inspection, not confirmed ink. Coordinates refer to native level 0.
    region = {"segment_index": 3, "y0": 1536, "y1": 2304, "x0": 768, "x1": 1920}
    crop = np.s_[region["y0"]:region["y1"], region["x0"]:region["x1"]]
    volume = np.load(DATA / "p800_03.npy", mmap_mode="r")[:, crop[0], crop[1]]
    result = {"region": region, "model_sha256": sha256(checkpoint),
        "selection": "Strongest forward cluster with little reverse response; exploratory ROI selection after inspecting all twelve maps.",
        "interpretation": "Stability is a screening check, not proof of ink or novelty.",
        "device": "cpu", "amp": False, "variants": []}
    for offset, tta in ((-4, False), (0, False), (4, False), (0, True)):
        started = time.monotonic()
        scores, layers = infer_volume(official, bundle, volume, torch.device("cpu"),
            config, "forward", offset=offset, tta=tta)
        name = f"verify_p800_03_z{offset:+d}_{'mirror' if tta else 'plain'}.npy"
        np.save(OUT / name, scores)
        result["variants"].append({"offset_slices": offset, "offset_um": offset * 8.64,
            "mirror_tta": tta, "layers": layers, "prediction": name,
            "sha256": sha256(OUT / name), "seconds": time.monotonic() - started})
        write_json(OUT / "verification.json", result)
        print("VERIFICATION", offset, "mirror", tta, "seconds", round(time.monotonic() - started, 2), flush=True)
    print("READER_VERIFICATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
