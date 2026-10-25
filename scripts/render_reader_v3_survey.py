"""Render measured CT and Reader scores; no generated or manually drawn strokes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.reader_v3_survey import DATA, OUT


def score_image(scores, valid, adaptive=False):
    lower, upper = (np.percentile(scores[valid], [50, 99.5]) if adaptive else (0.25, 0.75))
    strength = np.clip((scores - lower) / max(float(upper - lower), 1e-5), 0, 1)
    image = np.ones((*scores.shape, 3), np.float32)
    image[valid] = (1 - strength[valid])[:, None]
    image[~valid] = (0.87, 0.87, 0.87)
    return image, [float(lower), float(upper)]


def scale_bar(axis, shape, um):
    height, width = shape
    length_mm = 5 if width * um > 6000 else 2
    x, y = width * 0.06, height * 0.94
    axis.plot([x, x + length_mm * 1000 / um], [y, y], color="#00a6af", lw=3)
    axis.text(x, y - height * 0.025, f"{length_mm} mm", color="#00848c", fontsize=9)


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    pairs = {index: [item for item in summary["survey"] if item["index"] == index]
             for index in range(len(summary["protocol"]["segments"]))}
    scales = []
    for index, items in pairs.items():
        if len(items) != 2:
            continue
        volume = np.load(DATA / f"p800_{index:02d}.npy", mmap_mode="r")
        ct = np.asarray(volume[volume.shape[0] // 2])
        valid = ct > 0
        ct_min, ct_max = np.percentile(ct[valid], [1, 99])
        figure, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(np.ma.masked_where(~valid, ct), cmap="gray", vmin=ct_min, vmax=ct_max)
        axes[0, 0].set_title("CT original · capa central")
        mask_image = np.full((*valid.shape, 3), 0.87, np.float32)
        mask_image[valid] = 1
        axes[1, 0].imshow(mask_image)
        axes[1, 0].set_title("Área blanca: superficie muestreada")
        for column, item in enumerate(items, start=1):
            scores = np.load(OUT / item["prediction"])
            for row, adaptive in enumerate((False, True)):
                image, limits = score_image(scores, valid, adaptive)
                axes[row, column].imshow(image)
                text = "Contraste común 0,25–0,75" if not adaptive else "Contraste local mediana–p99,5"
                axes[row, column].set_title(f"Reader v3 · {item['direction']}\n{text}")
                scales.append({"index": index, "direction": item["direction"],
                               "adaptive": adaptive, "limits": limits})
                filename = f"p800_{index:02d}_{item['direction']}_{'local' if adaptive else 'common'}.png"
                Image.fromarray(np.uint8(np.round(image * 255))).save(OUT / filename)
        for axis in axes.flat:
            axis.set_xlabel("x (píxeles nativos)")
            axis.set_ylabel("y (píxeles nativos)")
            scale_bar(axis, ct.shape, 8.64)
        figure.suptitle(f"PHerc0800 · segmento {index:02d} · {items[0]['segment']}\n"
                       "Son puntuaciones del modelo: una mancha oscura no demuestra una letra", fontsize=13)
        figure.tight_layout()
        figure.savefig(OUT / f"p800_{index:02d}_comparison.png", dpi=140)
        plt.close(figure)
    labels = np.load(DATA / "w039_labels.npz")
    reviewed = labels["supervision"] > 0
    ink = labels["ink"] > 0
    bounds = (slice(1376, 2048), slice(0, 768))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    truth = np.ones((*ink.shape, 3), np.float32)
    truth[~reviewed] = 0.85
    truth[ink & reviewed] = 0
    axes[0].imshow(truth[bounds]); axes[0].set_title("Tinta ya conocida · anotación humana")
    for axis, item in zip(axes[1:], summary["control"]):
        scores = np.load(OUT / item["prediction"])
        image, _ = score_image(scores, np.ones(scores.shape, bool))
        axis.imshow(image[bounds])
        axis.set_title(f"Reader v3 · {item['direction']}\nAP en zona revisada: {item['metrics']['ap']:.3f}")
    for axis in axes:
        axis.axis("off")
        scale_bar(axis, ink[bounds].shape, 9.362)
    figure.suptitle("Control PHerc0139 w039 · incluido en el entrenamiento de Reader\n"
                   "Comprueba la ejecución; no mide la precisión en otro rollo")
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    figure.savefig(OUT / "known_letters_control.png", dpi=140)
    plt.close(figure)
    write_path = OUT / "display_scales.json"
    write_path.write_text(json.dumps(scales, indent=2) + "\n")
    figure, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row, index in enumerate(range(6)):
        for direction_index, direction in enumerate(("forward", "reverse")):
            axis = axes.flat[index * 2 + direction_index]
            path = OUT / f"p800_{index:02d}_{direction}_local.png"
            if path.exists():
                axis.imshow(Image.open(path))
            axis.set_title(f"Segmento {index:02d} · {direction}", fontsize=11)
            axis.axis("off")
    figure.suptitle("Búsqueda en los seis segmentos públicos de PHerc0800 · Reader v3\n"
                   "Contraste local para inspección: no equivale a probabilidad ni confirma texto", fontsize=14)
    figure.tight_layout()
    figure.savefig(OUT / "p800_survey_overview.png", dpi=130)
    plt.close(figure)
    if (OUT / "verification.json").exists():
        render_verification()
    print("READER_FIGURES_READY")


def render_verification():
    result = json.loads((OUT / "verification.json").read_text())
    if len(result["variants"]) != 4:
        return
    region = result["region"]
    crop = np.s_[region["y0"]:region["y1"], region["x0"]:region["x1"]]
    ct = np.load(DATA / "p800_03.npy", mmap_mode="r")[15][crop]
    valid = ct > 0
    extent = (region["x0"], region["x1"], region["y1"], region["y0"])
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    limits = np.percentile(ct[valid], [1, 99])
    axes.flat[0].imshow(ct, cmap="gray", vmin=limits[0], vmax=limits[1], extent=extent)
    axes.flat[0].set_title("CT original · capa central")
    original = np.load(OUT / "p800_03_forward.npy", mmap_mode="r")[crop]
    image, _ = score_image(original, valid)
    axes.flat[1].imshow(image, extent=extent)
    axes.flat[1].set_title("Barrido completo · orientación forward")
    for axis, item in zip(list(axes.flat)[2:], result["variants"]):
        scores = np.load(OUT / item["prediction"])
        image, _ = score_image(scores, valid)
        axis.imshow(image, extent=extent)
        name = "4 reflejos promediados" if item["mirror_tta"] else "Sin reflejos"
        axis.set_title(f"Profundidad {item['offset_um']:+.2f} µm · {name}")
    for axis in axes.flat:
        axis.set_xlabel("x nativa del segmento")
        axis.set_ylabel("y nativa del segmento")
    figure.suptitle("PHerc0800 · segmento 03 · comprobación de la zona más llamativa\n"
                   "Mismo contraste 0,25–0,75 en todos los mapas. Persistencia no demuestra tinta.", fontsize=13)
    figure.tight_layout()
    figure.savefig(OUT / "depth_and_mirror_verification.png", dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    main()
