#!/usr/bin/env python3
"""Validate Kintsugi on Scroll 4 w013 against the human ink labels published 2026-09-18.

Both predictions are compared on the label pyramid level 3 grid (19.2 um/px), restricted to the
`supervision` mask (the area humans actually reviewed). The official canonical prediction
(new_canon_autoresearch_recipe, ds8 jpg) is evaluated on exactly the same pixels as a reference.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("VESUVIUS_ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "docs" / "figures"))
LAB = REPO / "data/w013_labels"

CY, CX, NY, NX = 86, 62, 20, 25          # chunk window that was scanned (128 px chunks)
DS = 8                                    # native -> label level 3


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), no sklearn needed."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_f1(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best = (0.0, 0.0)
    for t in np.quantile(score, np.linspace(0.50, 0.995, 100)):
        p = score >= t
        tp = (p & y).sum()
        f1 = 2 * tp / max(1, p.sum() + y.sum())
        best = max(best, (f1, t))
    return best


def main() -> None:
    prob = np.load(REPO / "runs/scroll4_w013_validation_kintsugi_prediction.npy").astype(np.float32)
    H, W = prob.shape[0] // DS, prob.shape[1] // DS
    k = prob[: H * DS, : W * DS].reshape(H, DS, W, DS).mean(axis=(1, 3))

    y0, x0 = CY * 128 // DS, CX * 128 // DS
    ink = np.load(LAB / "ink_l3.npy")[y0:y0 + H, x0:x0 + W] > 0
    sup = np.load(LAB / "sup_l3.npy")[y0:y0 + H, x0:x0 + W] > 0
    can = np.asarray(Image.open("/tmp/w013_ds8.jpg").convert("L"), dtype=np.float32)[y0:y0 + H, x0:x0 + W] / 255

    m = sup
    rows = []
    for name, s in [("Kintsugi Physical (nuestro)", k), ("Canónico oficial (new_canon)", can)]:
        a = auc(s[m], ink[m])
        f1, _ = best_f1(s[m], ink[m])
        rows.append((name, a, f1))
    # Small registration check: label canvas has a ~-20 native px offset (~2.5 px here).
    shifts = {(dy, dx): auc(np.roll(k, (dy, dx), (0, 1))[m], ink[m]) for dy in range(-4, 5) for dx in range(-4, 5)}
    (bdy, bdx), bauc = max(shifts.items(), key=lambda kv: kv[1])

    print(f"window l3 y={y0}:{y0+H} x={x0}:{x0+W}  supervised px={m.sum()}  ink px={ink[m].sum()}")
    for name, a, f1 in rows:
        print(f"{name:32s} AUC={a:.3f}  bestF1={f1:.3f}")
    print(f"Kintsugi best AUC under +/-4px shift: {bauc:.3f} at (dy,dx)=({bdy},{bdx})  [optimistic, tuned on test]")
    print("Random baseline AUC = 0.500")

    fig, ax = plt.subplots(1, 4, figsize=(26, 6.5), facecolor="#0D1117")
    lo, hi = np.percentile(k, [2, 99.5])
    panels = [
        (ink.astype(float), "gray", "1. Etiquetas humanas (papirólogos, 18-sep-2026)"),
        (can, "inferno", f"2. Modelo canónico oficial  ·  AUC {rows[1][1]:.2f}"),
        ((k - lo) / (hi - lo), "inferno", f"3. Kintsugi Physical (nuestro)  ·  AUC {rows[0][1]:.2f}"),
        (sup.astype(float), "gray", "4. Zona revisada por humanos (máscara de evaluación)"),
    ]
    for a, (img, cm, t) in zip(ax, panels):
        a.imshow(np.clip(img, 0, 1), cmap=cm)
        a.set_title(t, color="white", fontsize=12, fontweight="bold")
        a.axis("off")
    fig.suptitle("Scroll 4 · w013 — Validación contra etiquetas humanas (AUC 0.5 = azar, 1.0 = perfecto)",
                 color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = ART / "scroll4_w013_validation_vs_human_labels.png"
    fig.savefig(out, dpi=110, facecolor="#0D1117")
    print("saved", out)


if __name__ == "__main__":
    main()
