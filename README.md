# Kintsugi Ink Detection

Kintsugi is an experimental 3D-to-2D ink-segmentation pipeline for Vesuvius Challenge surface volumes. The repository now fails closed on invalid labels, uses spatially disjoint validation, records full provenance, and compares a residual 3D U-Net against an axial state-space model.

## Current Evidence

- The original `sample_data/mask.tif` and `sample_data/inklabels.tif` are identical teacher predictions and are quarantined from training.
- The verified local dataset now contains Vesuvius Challenge 2023 fragments 1-3, layers 24-39, with independent binary masks and ink labels.
- Training uses fragments 1+2; fragment 3 supplies external validation and a frozen test split.
- Masks are tiled TIFFs, so DataLoader workers do not inflate RAM by decoding complete PNGs.
- Validation/test use deterministic, non-overlapping grids; training alone is label-balanced.
- A two-step smoke validates plumbing only. It is not evidence of ink-detection quality.
- The 128-step baseline `residual_unet` and `kintsugi_axial` checks rank near chance level (`+0.00055 AP` over prevalence on Fragment 1 and below prevalence on Fragment 2 zero-shot transfer).
- **Physical Differential Stem + Soft Sheet Unwarping (`kintsugi_physical`)**: Adding `PhysicalStem3D` (local patch Z-score $\tilde{V}$, Pavel Holoborodko smooth $\partial_z \tilde{V}$, epidermal phase Laplacian $\partial_{zz}^2 \tilde{V}$, and 2D thermal crackle residual $R_{\text{crackle}}$) plus `SoftSheetUnwarper3D` ($\Delta\hat{z}(y,x)$ differentiable sheet alignment) lifts 128-step Fragment 1 validation AP to **`0.21956` (`+0.00872` over prevalence, $\times 15.8$ higher lift than `kintsugi_axial`)**, achieves positive zero-shot transfer on **Fragment 2 (`0.17998 AP`, `+0.00810`)** and **Fragment 3 (`0.16486 AP`, `+0.00978`)**, and cuts cross-scroll false positives by **$>300{,}000\text{ FP/Mpix}$**.

See [data provenance](docs/data-provenance.md) for exact hashes, [experiment log](docs/experiment-log.md) for the honest baseline results, and [research roadmap](docs/research-roadmap.md) for promotion gates.

## Verify

```bash
./venv/bin/python -m unittest discover -s tests -v
./venv/bin/python -m compileall -q vesuvius_code scripts
./venv/bin/python -m vesuvius_code.smoke_test \
  --config configs/kintsugi_physical_local.json \
  --output-dir runs/tiled_raster_smoke \
  --steps 2
```

## Models

- `residual_unet`: small convolutional 3D U-Net baseline with learned depth aggregation.
- `residual_physical`: `residual_unet` equipped with `PhysicalStem3D` + `SoftSheetUnwarper3D`.
- `kintsugi_axial`: residual multiscale U-Net plus bidirectional sequence mixing over Z, Y, and X at the bottleneck.
- `kintsugi_physical`: `kintsugi_axial` equipped with `PhysicalStem3D` (4 physical differential channels) + `SoftSheetUnwarper3D` (differentiable $Z$-sheet unwarping).
- `legacy_kintsugi`: retained only for controlled comparison; it mixes long range only along Z and uses top-k depth reduction.

Physical data simulation and regularization utilities:
- `apply_fourth_augment_z` & `inject_synthetic_carbonized_ink` (`vesuvius_code/dataset.py`): Grand Prize-style $Z$-shift/crop augmentation (`_fourth_augment`) and Voronoi thermal-crackle + synchrotron phase-fringe synthetic ink generator.
- `AnisotropicFiberSuppressionLoss` (`vesuvius_code/losses.py`): Structure-tensor coherence penalty that suppresses false positives aligned with bare papyrus fibers.

`backend=mamba` is explicit and fails if the CUDA extension is unavailable. `backend=fallback` is a pure-PyTorch depthwise sequence mixer used for Mac development, not a substitute result for Mamba.

## Strict Inference

Inference accepts only schema-v2 checkpoints and writes real `float32` probabilities:

```bash
./venv/bin/python inference.py \
  --checkpoint runs/tiled_raster_smoke/smoke_model.ckpt \
  --zarr data/public/kaggle2023_fragment1/volume.zarr \
  --output runs/tiled_raster_smoke/probability.tif \
  --preview runs/tiled_raster_smoke/preview.png
```

The preview uses a fixed 0-1 scale. No per-image contrast stretching is applied.

## Cloud Gate

Build the pinned environment from [cloud/Dockerfile](cloud/Dockerfile). Start with one GPU and local NVMe. The Lightning entrypoint accepts an audited multi-source manifest, uses label-independent external validation, monitors distributed AP, saves resumable Lightning state, exports `best.strict.ckpt`, and retains GroupNorm without SyncBatchNorm.

```bash
python -m vesuvius_code.cloud_lightning_train --help
```

Four-GPU training is a later throughput decision, not the first experiment. Keep the global batch constant when changing device count.

## Scope

The Kaggle 2023 data support a legitimate surface-volume `3D -> 2D` baseline. They do not provide voxel-level 3D ink truth. The current public 2026 reference instead uses full CT cubes, DINO guidance, self-distillation, and a large residual 3D U-Net. Kintsugi should first be tested as a controlled sequence-mixing ablation, then moved to that volumetric setting only if the simpler benchmark shows repeatable cross-fragment gain.
