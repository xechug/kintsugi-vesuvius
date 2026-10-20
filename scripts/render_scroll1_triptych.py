#!/usr/bin/env python3
"""Render 3-panel side-by-side comparison on Scroll 1 (PHercParis4):

1. Raw Synchrotron X-Ray (Mean Z)
2. Kintsugi Physical Heavy (Physics-Guided 410k + 8-fold TTA)
3. 2023 Grand Prize Winner Canonical Model (Farritor / Nader / Schilliger)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_DIR = Path(__file__).resolve().parents[1]

# 1. Load our prediction and raw X-ray
kintsugi_img_path = REPO_DIR / "runs/scroll1_multiline_greek_kintsugi.png"
full_fig = Image.open(kintsugi_img_path)
w, h = full_fig.size

left_crop = full_fig.crop((int(w * 0.04), int(h * 0.15), int(w * 0.46), int(h * 0.98)))
right_crop = full_fig.crop((int(w * 0.54), int(h * 0.15), int(w * 0.96), int(h * 0.98)))

# 2. Load the canonical Grand Prize prediction for this exact crop
canon_ds8 = Image.open("/tmp/canon_pred_ds8.jpg")
canon_crop = canon_ds8.crop((2288, 5616, 2416, 5744))
canon_resized = canon_crop.resize(right_crop.size, Image.Resampling.BILINEAR)

# 3. Create the 3-panel triptych
fig, axes = plt.subplots(1, 3, figsize=(21, 8), dpi=140, facecolor="#0D1117")

axes[0].imshow(left_crop)
axes[0].set_title(
    "1. Raw Synchrotron X-Ray (Scroll 1)\n3D Density Tomography (Diamond Light Source)",
    color="#94A3B8",
    fontsize=11,
    fontweight="bold",
)
axes[0].axis("off")

axes[1].imshow(right_crop)
axes[1].set_title(
    "2. Kintsugi Physical Heavy (Our Physics-Informed AI)\n410k params + 4 Physical Lenses + 8-fold TTA",
    color="#22C55E",
    fontsize=11,
    fontweight="bold",
)
axes[1].axis("off")

canon_arr = np.asarray(canon_resized, dtype=np.float32) / 255.0
axes[2].imshow(canon_arr, cmap="inferno", vmin=0.0, vmax=1.0)
axes[2].set_title(
    "3. 2023 Grand Prize Canonical Reference\n(Farritor / Nader / Schilliger Benchmark)",
    color="#38BDF8",
    fontsize=11,
    fontweight="bold",
)
axes[2].axis("off")

fig.suptitle(
    "Vesuvius Challenge: Direct Validation on Sealed Scroll 1 (PHercParis4, Segment 20231012184424)\nComparing Physics-Guided Kintsugi AI Against the 2023 Grand Prize Benchmark",
    fontsize=13,
    color="#F8FAFC",
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = REPO_DIR / "runs/scroll1_triptych_benchmark.png"
fig.savefig(out_path, dpi=140, facecolor="#0D1117")
plt.close(fig)
print(f"Saved triptych to {out_path}")
