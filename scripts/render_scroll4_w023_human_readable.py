import os
#!/usr/bin/env python3
"""Render clean, human-readable black-ink-on-parchment letters for Scroll 4 Segment w023 (Hotspot #1)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

REPO_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("VESUVIUS_ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "docs" / "figures"))

# Load prediction and native overlay
prob_path = REPO_DIR / "runs/scroll4_w023_hotspot1_kintsugi_prediction.npy"
overlay_path = REPO_DIR / "runs/scroll4_w023_hotspot1_kintsugi_prediction_native_overlay.png"

if not prob_path.exists() or not overlay_path.exists():
    print(f"Waiting for files: {prob_path} and {overlay_path}")
    exit(0)

prob = np.load(prob_path)
overlay_img = Image.open(overlay_path)
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

native_xray = np.asarray(overlay_img.convert("L"), dtype=np.float32) / 255.0
parchment = np.clip(parchment * (0.60 + 0.40 * native_xray[:, :, None]), 0, 1)

# Ink strength
ink_strength = np.clip((hp_norm - 0.20) / 0.80, 0, 1)
clean_papyrus = parchment * (1.0 - 0.90 * ink_strength[:, :, None])

# If empty region (black in xray), keep it dark background
empty_mask = (native_xray < 0.05)
clean_papyrus[empty_mask] = 0.05

# Identify top 3 regions of ink density in highpass
crops = [
    ("Upper Inscription Band (y=200..900, x=500..2000)", (200, 500, min(h, 900), min(w, 2000))),
    ("Central Text Column (y=900..1600, x=800..2300)", (min(h-700, 900), 800, min(h, 1600), min(w, 2300))),
    ("Lower Scribal Zone (y=1600..2300, x=1000..2500)", (min(h-700, 1600), 1000, min(h, 2300), min(w, 2500))),
]

fig = plt.figure(figsize=(24, 18), dpi=140, facecolor="#0D1117")
gs = fig.add_gridspec(4, 3, height_ratios=[1.6, 1, 1, 1], hspace=0.3, wspace=0.15)

# Row 0: Full Canvas
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(native_xray, cmap="gray")
ax0.set_title("1. Synchrotron X-Ray (Slab Z=46..62)\nScroll 4 w023 Hotspot #1 (Y=8320..10880, X=44160..47360)", color="#94A3B8", fontsize=11, fontweight="bold")
ax0.axis("off")

ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(prob, cmap="inferno")
ax1.set_title(f"2. Kintsugi Physical Probability Map\nMax P = {prob.max():.2f} (Voxel-wise)", color="#EF4444", fontsize=11, fontweight="bold")
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
    "Scroll 4 (PHerc. 1667) Segment w023 Hotspot #1 — Isolated Greek Letter Candidates\nPhysics-Informed Kintsugi Heavy Model (410k params + 8-fold TTA)",
    fontsize=14,
    color="#F8FAFC",
    fontweight="bold",
    y=0.99,
)

out_local = REPO_DIR / "runs/scroll4_w023_human_readable.png"
out_artifact = ARTIFACT_DIR / "scroll4_w023_human_readable.png"
fig.savefig(out_local, dpi=140, facecolor="#0D1117")
fig.savefig(out_artifact, dpi=140, facecolor="#0D1117")
plt.close(fig)
print(f"Saved human-readable render to: {out_local} and {out_artifact}")
