"""Render held-out PHerc1447 w060 benchmark and w062 search maps for Reader v3 + Kintsugi."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
STUDY = "reader_kintsugi_20261025"
DATA = REPO / "data" / STUDY
OUT = REPO / "runs" / STUDY

LABELS = {
    "reader_v3": "Reader v3 (congelado)",
    "kintsugi_alone": "Kintsugi solo",
    "reader_residual": "Reader + Residual",
    "reader_kintsugi": "Reader + Kintsugi",
    "reader_kintsugi_fiber": "Reader + Kintsugi + Fibra",
}


def scale_bar(ax, shape, um=8.64, mm=5.0):
    height, width = shape
    x, y = width * 0.06, height * 0.93
    px = mm * 1000.0 / um
    ax.plot([x, x + px], [y, y], color="#00d2ff", lw=3)
    ax.text(x, y - height * 0.025, f"{mm:.0f} mm", color="#00d2ff", fontsize=8, weight="bold")


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    orientations = json.loads((OUT / "orientation.json").read_text())["directions"]
    test_name = summary["protocol"]["test"]
    direction = orientations[test_name]

    volume = np.load(DATA / f"{test_name}.npy", mmap_mode="r")
    ct_mid = np.asarray(volume[volume.shape[0] // 2])
    lbl = np.load(DATA / f"{test_name}_labels.npz")
    ink = lbl["ink"] > 127
    mask = (lbl["supervision"] > 0) & (volume.max(0) > 0)

    reader_test = np.load(OUT / f"reader_{test_name}_{direction}.npy")
    base_metrics = summary["baseline_test"]

    columns = [
        ("reader_v3", reader_test, base_metrics),
        ("kintsugi_alone", np.load(OUT / "kintsugi_alone_ensemble_test.npy"), summary["ensemble_test"]["kintsugi_alone"]),
        ("reader_residual", np.load(OUT / "reader_residual_ensemble_test.npy"), summary["ensemble_test"]["reader_residual"]),
        ("reader_kintsugi", np.load(OUT / "reader_kintsugi_ensemble_test.npy"), summary["ensemble_test"]["reader_kintsugi"]),
        ("reader_kintsugi_fiber", np.load(OUT / "reader_kintsugi_fiber_ensemble_test.npy"), summary["ensemble_test"]["reader_kintsugi_fiber"]),
    ]

    fig, axes = plt.subplots(2, 6, figsize=(24, 8.5), constrained_layout=True)
    truth = np.zeros((*ink.shape, 3), np.float32)
    truth[mask] = [0.25, 0.25, 0.25]
    truth[mask & ink] = [0.1, 0.9, 0.4]
    axes[0, 0].imshow(truth)
    axes[0, 0].set_title("PHerc1447 w060 (Test ciego)\nVerde: tinta humana; gris: revisado")

    vmin, vmax = np.percentile(ct_mid[ct_mid > 0], [1, 99])
    axes[1, 0].imshow(ct_mid, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"CT nativo 8,64 µm ({direction})\nExcluido a >20 vóxeles de w058")
    scale_bar(axes[1, 0], ink.shape)

    for col_idx, (key, scores, m) in enumerate(columns, start=1):
        axes[0, col_idx].imshow(np.ma.masked_where(~mask, scores), cmap="inferno", vmin=0, vmax=1)
        axes[0, col_idx].set_title(
            f"{LABELS[key]}\nAP {m['ap']:.3f} | AUC {m['roc_auc']:.3f}"
        )
        pred = scores >= m["threshold"]
        rgb = np.zeros((*ink.shape, 3), np.float32)
        rgb[mask & pred & ink] = [0.1, 0.85, 0.35]
        rgb[mask & pred & ~ink] = [0.95, 0.15, 0.10]
        rgb[mask & ~pred & ink] = [0.15, 0.45, 0.85]
        axes[1, col_idx].imshow(rgb)
        axes[1, col_idx].set_title(
            f"Umbral fijado en val (Recall≈0,80)\nPrec {m['precision']:.3f} | Rec {m['recall']:.3f} | FP/Mpx {m['fp_per_mpix']:.0f}"
        )

    for ax in axes.flat:
        ax.axis("off")

    fig.legend(
        handles=[
            Patch(color=[0.1, 0.85, 0.35], label="Tinta detectada (TP)"),
            Patch(color=[0.95, 0.15, 0.10], label="Falsa alarma sobre fibra/fondo (FP)"),
            Patch(color=[0.15, 0.45, 0.85], label="Tinta omitida (FN)"),
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        "Evaluación ciega en PHerc1447 w060: Reader v3 congelado vs. Corrección física Kintsugi (2 semillas)",
        fontsize=15,
    )
    fig.savefig(OUT / "test_w060_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Render search crops on PHerc1447 w062
    search_items = summary.get("search", [])
    if search_items:
        fig, axes = plt.subplots(len(search_items), 4, figsize=(18, 4.3 * len(search_items)), constrained_layout=True)
        if len(search_items) == 1:
            axes = np.expand_dims(axes, 0)
        for row_idx, item in enumerate(search_items):
            sname = item["item"]
            sdir = item["direction"]
            mname = item["model"]
            svol = np.load(DATA / f"{sname}.npy", mmap_mode="r")
            sct = np.asarray(svol[svol.shape[0] // 2])
            svalid = sct > 0
            r_map = np.load(OUT / f"reader_{sname}_{sdir}.npy")
            k_map = np.load(OUT / item["prediction"])
            diff = k_map - r_map

            cmin, cmax = np.percentile(sct[svalid], [1, 99])
            axes[row_idx, 0].imshow(sct, cmap="gray", vmin=cmin, vmax=cmax)
            axes[row_idx, 0].set_title(f"{sname} · CT central ({sdir})")
            scale_bar(axes[row_idx, 0], sct.shape)

            axes[row_idx, 1].imshow(1.0 - np.clip((r_map - 0.25) / 0.5, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 1].set_title(f"Reader v3 ({sdir})\nContraste 0,25–0,75")

            axes[row_idx, 2].imshow(1.0 - np.clip((k_map - 0.25) / 0.5, 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 2].set_title(f"{LABELS.get(mname, mname)} ({sdir})\nEstado: {item['status']}")

            axes[row_idx, 3].imshow(diff, cmap="coolwarm", vmin=-0.25, vmax=0.25)
            axes[row_idx, 3].set_title("Diferencia (Kintsugi − Reader)\nAzul: suprime falsa alarma; Rojo: refuerza")

        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(
            "Exploración en PHerc1447 w062 (sin etiquetas humanas alineadas): Reader v3 vs. Refinador Kintsugi",
            fontsize=14,
        )
        fig.savefig(OUT / "search_w062_overview.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    print("READER_KINTSUGI_FIGURES_READY", flush=True)


if __name__ == "__main__":
    main()
