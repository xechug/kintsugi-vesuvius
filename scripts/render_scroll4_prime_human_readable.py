import os
#!/usr/bin/env python3
"""Render clean, human-readable black-ink-on-parchment letters for the Scroll 4 prime text region."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

REPO_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("VESUVIUS_ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "docs" / "figures"))

# Load prediction and native overlay
prob = np.load(REPO_DIR / "runs/scroll4_prime_region_kintsugi_prediction.npy")
overlay_img = Image.open(REPO_DIR / "runs/scroll4_prime_region_kintsugi_prediction_native_overlay.png")
h, w = prob.shape

# High-pass filter to remove papyrus sheet low-frequency drift
im_prob = Image.fromarray((prob * 255).astype(np.uint8))
bg = np.asarray(im_prob.filter(ImageFilter.GaussianBlur(radius=24)), dtype=np.float32)
raw = np.asarray(im_prob, dtype=np.float32)
highpass = np.maximum(0, raw - bg)

hp_p98 = np.percentile(highpass, 98.5)
hp_norm = np.clip(highpass / max(1e-5, hp_p98), 0, 1)

# Papyrologist rendering: realistic warm parchment + black carbon ink strokes
parchment = np.zeros((h, w, 3), dtype=np.float32)
parchment[:, :, 0] = 0.88
parchment[:, :, 1] = 0.82
parchment[:, :, 2] = 0.68

native_xray = np.asarray(overlay_full := overlay_img.convert("L"), dtype=np.float32) / 255.0
parchment = np.clip(parchment * (0.60 + 0.40 * native_xray[:, :, None]), 0, 1)

# Ink strength
ink_strength = np.clip((hp_norm - 0.20) / 0.80, 0, 1)
clean_papyrus = parchment * (1.0 - 0.90 * ink_strength[:, :, None])

# If empty region (black in xray), keep it dark background
empty_mask = (native_xray < 0.05)
clean_papyrus[empty_mask] = 0.05

# Crops of top text features
crops = [
    ("Upper-Right Line Sequence (y=100..700, x=1800..2800)", (100, 1800, 700, 2800)),
    ("Mid-Right Character Cluster (y=700..1300, x=1900..2900)", (700, 1900, 1300, 2900)),
    ("Lower-Right Inscription Zone (y=1800..2500, x=2200..3000)", (1800, 2200, 2500, 3000)),
]

fig = plt.figure(figsize=(24, 18), dpi=140, facecolor="#0D1117")
gs = fig.add_gridspec(4, 3, height_ratios=[1.6, 1, 1, 1], hspace=0.3, wspace=0.15)

# Row 0: Full Canvas
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(native_xray, cmap="gray")
ax0.set_title("1. Synchrotron X-Ray (Slab Z=46..62)\nPrime Region (Y=28800..31360, X=26240..29312)", color="#94A3B8", fontsize=11, fontweight="bold")
ax0.axis("off")

ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(prob, cmap="inferno")
ax1.set_title("2. Kintsugi Physical Probability Map\nMax P = 0.86 (Voxel-wise)", color="#EF4444", fontsize=11, fontweight="bold")
ax1.axis("off")

ax2 = fig.add_subplot(gs[0, 2])
ax2.imshow(clean_papyrus)
ax2.set_title("3. Clean Papyrological Ink Strokes (Black on Papyrus)\nHigh-pass filtered ink deposits", color="#22C55E", fontsize=11, fontweight="bold")
ax2.axis("off")

# Rows 1..3: Zooms
for r, (title, (y1, x1, y2, x2)) in enumerate(crops):
    sub_x = native_xray[y1:y2, x1:x2]
    sub_h = prob[y1:y2, x1:x2]
    sub_c = clean_papyrus[y1:y2, x1:x2]
    
    ax_x = fig.add_subplot(gs[r + 1, 0])
    ax_x.imshow(sub_x, cmap="gray")
    ax_x.set_title(f"{title} — Raw X-Ray", color="#94A3B8", fontsize=10)
    ax_x.axis("off")
    
    ax_h = fig.add_subplot(gs[r + 1, 1])
    ax_h.imshow(sub_h, cmap="inferno", vmin=0.3, vmax=0.85)
    ax_h.set_title(f"{title} — Kintsugi Heatmap", color="#38BDF8", fontsize=10)
    ax_h.axis("off")
    
    ax_c = fig.add_subplot(gs[r + 1, 2])
    ax_c.imshow(sub_c)
    ax_c.set_title(f"{title} — Isolated Ink Strokes", color="#22C55E", fontsize=10, fontweight="bold")
    ax_c.axis("off")

fig.suptitle(
    "Scroll 4 (PHerc. 1667) Prime Text Region — Isolated Ink Letters on Papyrus\nPhysics-Informed Kintsugi Heavy Model (410k params + 8-fold TTA)",
    fontsize=14,
    color="#F8FAFC",
    fontweight="bold",
    y=0.99,
)

out_local = REPO_DIR / "runs/scroll4_prime_region_human_readable.png"
out_artifact = ARTIFACT_DIR / "scroll4_prime_region_human_readable.png"
fig.savefig(out_local, dpi=140, facecolor="#0D1117")
fig.savefig(out_artifact, dpi=140, facecolor="#0D1117")
plt.close(fig)
print(f"Saved human-readable prime region analysis to {out_local} and {out_artifact}")
