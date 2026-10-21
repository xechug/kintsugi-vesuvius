#!/usr/bin/env python3
"""Render high-resolution zoomed crops of detected letter candidates on Virgin Scroll 4."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_DIR = Path("/Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge")
ARTIFACT_DIR = Path("/Users/jsmcore/.gemini/antigravity/brain/facf1ebf-2061-4f7f-b872-c0129914aef4")

# Load raw probability
prob = np.load(REPO_DIR / "runs/scroll4_virgin_kintsugi_prediction.npy")

# We can extract the raw X-ray crop from the saved 3-panel figure
full_img = Image.open(REPO_DIR / "runs/scroll4_virgin_kintsugi_prediction.png")
w, h = full_img.size

# Panel 1 (left) is roughly x: 0.03*w to 0.31*w, y: 0.15*h to 0.98*h
p1_crop = full_img.crop((int(w * 0.03), int(h * 0.15), int(w * 0.31), int(h * 0.98))).resize((2048, 2048))
xray = np.asarray(p1_crop.convert("L"), dtype=np.float32) / 255.0

# Define Candidate Box 1 (Upper Right Text Column: y=150..950, x=1200..2000)
# Define Candidate Box 2 (Mid Left / Central Trajectory: y=400..1100, x=100..900)
boxes = [
    ("Candidate Area A: Upper Right Scribal Margin", (150, 1150, 950, 1950)),
    ("Candidate Area B: Central Text Band", (600, 200, 1400, 1000)),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=140, facecolor="#0D1117")

for idx, (title, (y1, x1, y2, x2)) in enumerate(boxes):
    sub_xray = xray[y1:y2, x1:x2]
    sub_prob = prob[y1:y2, x1:x2]
    
    # Normalized probability for this patch
    p_lo, p_hi = np.percentile(sub_prob, [35, 99])
    norm_p = np.clip((sub_prob - p_lo) / max(1e-5, p_hi - p_lo), 0, 1)
    
    # Col 0: X-ray
    axes[idx, 0].imshow(sub_xray, cmap="gray")
    axes[idx, 0].set_title(f"{title}\nRaw Synchrotron X-Ray", color="#94A3B8", fontsize=10, fontweight="bold")
    axes[idx, 0].axis("off")
    
    # Col 1: Kintsugi Heatmap
    axes[idx, 1].imshow(norm_p, cmap="inferno", vmin=0, vmax=1)
    axes[idx, 1].set_title(f"Kintsugi Physical Probability\nMax P = {sub_prob.max():.3f}", color="#22C55E", fontsize=10, fontweight="bold")
    axes[idx, 1].axis("off")
    
    # Col 2: Composite Overlay
    mask = np.clip((norm_p - 0.5) / 0.5, 0, 1)
    comp = np.dstack([sub_xray * 0.75, sub_xray * 0.75, sub_xray * 0.75])
    comp[:, :, 0] = np.clip(comp[:, :, 0] + mask * 0.95, 0, 1)
    comp[:, :, 1] = np.clip(comp[:, :, 1] + mask * 0.75, 0, 1)
    comp[:, :, 2] = np.clip(comp[:, :, 2] + mask * 0.15, 0, 1)
    
    axes[idx, 2].imshow(comp)
    axes[idx, 2].set_title("Reconstructed Ink Strokes Overlay", color="#F59E0B", fontsize=10, fontweight="bold")
    axes[idx, 2].axis("off")

fig.suptitle(
    "Scroll 4 (PHerc. 1667, Buried 79 AD) — High-Resolution Letter Candidate Details\nKintsugi Physical Reconstruction (Gate A 410k + 8-fold TTA)",
    fontsize=13,
    color="#F8FAFC",
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_local = REPO_DIR / "runs/scroll4_virgin_zoomed_letters.png"
out_artifact = ARTIFACT_DIR / "scroll4_virgin_zoomed_letters.png"
fig.savefig(out_local, dpi=140, facecolor="#0D1117")
fig.savefig(out_artifact, dpi=140, facecolor="#0D1117")
plt.close(fig)
print(f"Saved zoomed crops to {out_local} and {out_artifact}")
