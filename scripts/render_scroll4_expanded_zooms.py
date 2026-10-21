#!/usr/bin/env python3
"""Render zoomed multi-line Greek text bands from the expanded Scroll 4 virgin scan."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_DIR = Path("/Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge")
ARTIFACT_DIR = Path("/Users/jsmcore/.gemini/antigravity/brain/facf1ebf-2061-4f7f-b872-c0129914aef4")

# Load native overlay and raw probability
overlay_img = Image.open(REPO_DIR / "runs/scroll4_virgin_expanded_prediction_native_overlay.png")
prob = np.load(REPO_DIR / "runs/scroll4_virgin_expanded_prediction.npy")
w, h = overlay_img.size

# Multi-line Scribal Column Band:
# y: 650..1850 (height: 1200 px), x: 2200..5000 (width: 2800 px)
y1, y2 = 650, 1850
x1, x2 = 2200, 5000

sub_overlay = overlay_img.crop((x1, y1, x2, y2))
sub_prob = prob[y1:y2, x1:x2]

# High-detail character crops (4 close-ups)
char_crops = [
    ("Line 1: Character Sequence α", (750, 2600, 1150, 3200)),
    ("Line 1: Character Sequence β", (750, 3400, 1150, 4000)),
    ("Line 2: Mid-Band Inscriptions γ", (1250, 2500, 1650, 3100)),
    ("Line 2: Right Margin Inscriptions δ", (1200, 3800, 1600, 4400)),
]

fig = plt.figure(figsize=(24, 16), dpi=140, facecolor="#0D1117")
gs = fig.add_gridspec(3, 4, height_ratios=[1.8, 1, 1], hspace=0.25, wspace=0.15)

# Top: Wide Multi-Line Inscription Banner
ax_banner = fig.add_subplot(gs[0, :])
ax_banner.imshow(sub_overlay)
ax_banner.set_title(
    "Scroll 4 (PHerc. 1667) — Consecutive Horizontal Scribal Lines (Expanded Column Crop: 2800x1200 px)\nReconstructed Carbon Ink Pigment (Gold) Superimposed on Diamond Light Source Synchrotron X-Ray",
    color="#F59E0B",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax_banner.axis("off")

# Bottom: 4 Zoomed Close-ups (Top row: Heatmap, Bottom row: Overlay)
for i, (title, (cy1, cx1, cy2, cx2)) in enumerate(char_crops):
    crop_prob = prob[cy1:cy2, cx1:cx2]
    crop_overlay = overlay_img.crop((cx1, cy1, cx2, cy2))
    
    p_lo, p_hi = np.percentile(crop_prob, [35, 99.5])
    norm_p = np.clip((crop_prob - p_lo) / max(1e-5, p_hi - p_lo), 0, 1)
    
    ax_heat = fig.add_subplot(gs[1, i])
    ax_heat.imshow(norm_p, cmap="inferno", vmin=0, vmax=1)
    ax_heat.set_title(f"{title}\nKintsugi Heatmap (P_max={crop_prob.max():.2f})", color="#22C55E", fontsize=9, fontweight="bold")
    ax_heat.axis("off")
    
    ax_comp = fig.add_subplot(gs[2, i])
    ax_comp.imshow(crop_overlay)
    ax_comp.set_title(f"Composite Stroke Trace", color="#38BDF8", fontsize=9, fontweight="bold")
    ax_comp.axis("off")

fig.suptitle(
    "Vesuvius Challenge: Virgin Scroll 4 (PHerc. 1667) Multi-Line Scribal Text Analysis\nKintsugi Physical Heavy Model (410k params + 8-fold TTA) | First Letters Prize Candidate",
    fontsize=14,
    color="#F8FAFC",
    fontweight="bold",
    y=0.98,
)

out_local = REPO_DIR / "runs/scroll4_virgin_multiline_letters.png"
out_artifact = ARTIFACT_DIR / "scroll4_virgin_multiline_letters.png"
fig.savefig(out_local, dpi=140, facecolor="#0D1117")
fig.savefig(out_artifact, dpi=140, facecolor="#0D1117")
plt.close(fig)
print(f"Saved multi-line analysis to {out_local} and {out_artifact}")
