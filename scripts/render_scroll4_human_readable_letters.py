import os
#!/usr/bin/env python3
"""Generate human-readable, crisp, papyrologist-style letter rendering for Scroll 4."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

REPO_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("VESUVIUS_ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "docs" / "figures"))

# 1. Load data
prob = np.load(REPO_DIR / "runs/scroll4_virgin_expanded_prediction.npy")
canon_im = Image.open("/tmp/canon_scroll4_ds8.jpg")
canon_arr = np.asarray(canon_im)
# Coordinates in ds8: y: 2960..3344, x: 3040..3712
canon_sub = canon_arr[2960:3344, 3040:3712]
canon_full = np.array(Image.fromarray(canon_sub).resize((5376, 3072), Image.Resampling.BILINEAR))

overlay_full = Image.open(REPO_DIR / "runs/scroll4_virgin_expanded_prediction_native_overlay.png")

# 2. High-pass filter to isolate crisp strokes from papyrus low-frequency drift
im_prob = Image.fromarray((prob * 255).astype(np.uint8))
bg = np.asarray(im_prob.filter(ImageFilter.GaussianBlur(radius=28)), dtype=np.float32)
raw = np.asarray(im_prob, dtype=np.float32)
highpass = np.maximum(0, raw - bg)

# Normalize highpass to [0, 1]
hp_p98 = np.percentile(highpass, 98.5)
hp_norm = np.clip(highpass / max(1e-5, hp_p98), 0, 1)

# Papyrologist rendering: black ink strokes on realistic papyrus background
# Papyrus warm parchment color: (218, 198, 160)
parchment = np.zeros((3072, 5376, 3), dtype=np.float32)
parchment[:, :, 0] = 0.85
parchment[:, :, 1] = 0.78
parchment[:, :, 2] = 0.63

# Modulate parchment with raw X-ray texture
native_xray = np.asarray(overlay_full.convert("L"), dtype=np.float32) / 255.0
parchment = np.clip(parchment * (0.65 + 0.35 * native_xray[:, :, None]), 0, 1)

# Subtract ink (dark carbon ink: R: 0.1, G: 0.08, B: 0.05)
ink_strength = np.clip((hp_norm - 0.25) / 0.75, 0, 1)
clean_papyrus = parchment * (1.0 - 0.85 * ink_strength[:, :, None])

# 3. Choose the most distinct letter regions (crops)
# Crop 1: Line 1 clear characters (x: 3500..4100, y: 800..1200)
# Crop 2: Central scribal cluster (x: 2400..3000, y: 700..1100)
# Crop 3: Line 2 right margin letters (x: 3800..4400, y: 1300..1700)
crops = [
    ("Candidate 1: Horizontal Sequence (Line 1)", (800, 3500, 1200, 4100)),
    ("Candidate 2: Central Character Cluster", (700, 2400, 1100, 3000)),
    ("Candidate 3: Line 2 Lower Register", (1300, 3800, 1700, 4400)),
]

fig = plt.figure(figsize=(24, 18), dpi=140, facecolor="#0D1117")
gs = fig.add_gridspec(4, 3, height_ratios=[1.6, 1, 1, 1], hspace=0.3, wspace=0.15)

# Row 0: Full Column Comparison (Wide)
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(native_xray, cmap="gray")
ax0.set_title("1. Raw Synchrotron X-Ray (Diamond Light Source)\nInvisible to naked eye", color="#94A3B8", fontsize=11, fontweight="bold")
ax0.axis("off")

ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(prob, cmap="inferno")
ax1.set_title("2. Raw Probability Heatmap (What looked like clouds)\nVoxel-by-voxel continuous probability", color="#EF4444", fontsize=11, fontweight="bold")
ax1.axis("off")

ax2 = fig.add_subplot(gs[0, 2])
ax2.imshow(clean_papyrus)
ax2.set_title("3. Papyrological Binarized View (Filtered Strokes)\nBlack carbon ink on papyrus ground (Cleaned)", color="#22C55E", fontsize=11, fontweight="bold")
ax2.axis("off")

# Rows 1 to 3: Detailed Zooms of individual candidate characters
for row_idx, (crop_title, (y1, x1, y2, x2)) in enumerate(crops):
    sub_xray = native_xray[y1:y2, x1:x2]
    sub_heat = prob[y1:y2, x1:x2]
    sub_pap = clean_papyrus[y1:y2, x1:x2]
    sub_canon = canon_full[y1:y2, x1:x2]
    
    # Col 0: Raw X-ray
    ax_r0 = fig.add_subplot(gs[row_idx + 1, 0])
    ax_r0.imshow(sub_xray, cmap="gray")
    ax_r0.set_title(f"{crop_title} — Raw X-Ray", color="#94A3B8", fontsize=10)
    ax_r0.axis("off")
    
    # Col 1: Canon Grand Prize Benchmark
    ax_r1 = fig.add_subplot(gs[row_idx + 1, 1])
    ax_r1.imshow(sub_canon, cmap="inferno", vmin=0, vmax=255)
    ax_r1.set_title(f"{crop_title} — Vesuvius Official Canon Model", color="#38BDF8", fontsize=10)
    ax_r1.axis("off")
    
    # Col 2: Kintsugi Clean Letters (Black on Papyrus)
    ax_r2 = fig.add_subplot(gs[row_idx + 1, 2])
    ax_r2.imshow(sub_pap)
    ax_r2.set_title(f"{crop_title} — Kintsugi Ink Strokes Isolated", color="#22C55E", fontsize=10, fontweight="bold")
    ax_r2.axis("off")

fig.suptitle(
    "Scroll 4 (PHerc. 1667) — Making Ancient Greek Letters Human-Readable\nComparing Raw Heatmaps vs. Filtered Scribal Strokes & Official Vesuvius Benchmark",
    fontsize=14,
    color="#F8FAFC",
    fontweight="bold",
    y=0.99,
)

out_local = REPO_DIR / "runs/scroll4_human_readable_letters.png"
out_artifact = ARTIFACT_DIR / "scroll4_human_readable_letters.png"
fig.savefig(out_local, dpi=140, facecolor="#0D1117")
fig.savefig(out_artifact, dpi=140, facecolor="#0D1117")
plt.close(fig)
print(f"Saved human readable letter analysis to {out_local} and {out_artifact}")
