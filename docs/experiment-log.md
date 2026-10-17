# Experiment Log

## 2026-07-15: Pipeline Audit

The original local labels were invalid: `sample_data/mask.tif` and `sample_data/inklabels.tif` are byte-identical teacher predictions. The legacy checkpoint loaded only 6 of 44 tensors, uint8 inference was divided by 65535, and evaluation patches overlapped training. Results produced by that path are not scientific evidence.

## Verified Data

Kaggle 2023 fragments 1, 2, and 3 now contain independent ground truth, 16 CT layers (24-39), chunked Zarr volumes, frozen spatial splits, and tiled TIFF masks. The active multi-source protocol is:

- Train: guarded train regions from fragments 1 and 2.
- Validation: guarded validation regions from fragment 3.
- Test: guarded test regions from fragment 3, untouched.

## 128-Step Architecture Check

Protocol: fragment 1 train split, 128 steps, seed 1337, base channels 4, and every valid non-overlapping `128x128` patch in the fragment 1 validation regions. AP is compared with validation prevalence; a ranking with no information has expected AP equal to prevalence.

| Model | Parameters | AP | Prevalence | Brier |
|---|---:|---:|---:|---:|
| Residual 3D U-Net | 5,966 | 0.21207 | 0.21084 | 0.23561 |
| Kintsugi axial fallback | 6,822 | 0.21139 | 0.21084 | 0.23043 |
| Kintsugi physical (`PhysicalStem3D` + `SoftSheetUnwarper3D`) | 7,151 | 0.21956 | 0.21084 | 0.32149 |

The raw-intensity models are at random-ranking level, whereas `kintsugi_physical` achieves a `+0.00872` lift over prevalence in 128 steps. Reports:

- `runs/ablation_residual_128/fragment1_validation_grid.json`
- `runs/ablation_kintsugi_128/fragment1_validation_grid.json`
- `runs/ablation_kintsugi_physical_128/fragment1_validation_grid.json`

The earlier `fragment1_validation.json` reports sampled overlapping patches and are retained only as historical diagnostics.

Frozen-threshold transfer to external fragments:

| External fragment | Prevalence | Residual AP | Kintsugi Physical AP | Residual FP/Mpix | Kintsugi Physical FP/Mpix |
|---|---:|---:|---:|---:|---:|
| Fragment 2 | 0.17188 | 0.16755 | 0.17998 | >828,000 | 522,012 |
| Fragment 3 | 0.15508 | 0.15982 | 0.16486 | >828,000 | 501,610 |

The raw-intensity residual checkpoint selected threshold `0.15` on fragment 1 and predicted almost every valid pixel as ink on both external fragments (>828,000 FP/Mpix). Adding `PhysicalStem3D` (per-patch Z-score + Holoborodko vertical derivatives + planar crackle residual) and `SoftSheetUnwarper3D` lifted Fragment 2 transfer from below prevalence (`0.16755`) to above prevalence (`0.17998`) and reduced false positives by >300,000 FP/Mpix.

## Capacity Check

Both models memorized the same audited mixed patch with 9.89% positive pixels after 200 iterations. This proves gradients, normalization, labels, and minimum model capacity work; it does not prove generalization.

| Model | Final AP | Best F1 | FP | FN |
|---|---:|---:|---:|---:|
| Residual 3D U-Net | 0.99982 | 0.99475 | 6 | 11 |
| Kintsugi axial fallback | 0.99996 | 0.99846 | 5 | 0 |

Reports are under `runs/overfit_residual_mixed/` and `runs/overfit_kintsugi_mixed/`.

## I/O Check

Fragment 2, patch `16x128x128`, 256 timed samples after 32 warmups on the M1 Pro:

| Workers | Chunk alignment | Patches/s | Read amplification |
|---:|---|---:|---:|
| 0 | No | 577 | 3.97x |
| 2 | No | 938 | 3.97x |
| 4 | No | 1,306 | 3.97x |
| 4 | Yes | 2,276 | 1.00x |

These are warm-cache local figures, not cloud guarantees. Re-run `scripts/benchmark_zarr_io.py` on the actual mounted RunPod storage. Training aligns to chunks by default; validation remains unaligned and samples valid positions without replacement.

## 2026 Reference Gap

The current public Vesuvius reference is a roughly 142M-parameter residual 3D U-Net trained on `256^3` CT cubes with DINO-guided supervision, self-distillation, EMA, BF16, and Dice+BCE. The public Paris 4 volume is `75784x32693x32693`; the local 109-layer Zarr is a surface-aligned volume, not that full CT. The Kaggle 2023 benchmark remains useful for 3D-to-2D architecture ablation, but it cannot validate voxel-level 3D ink segmentation.

The public 3D prediction Zarr is also a teacher output, not independent ground truth. It may be used for explicitly tracked distillation, never as validation or test truth.

## Decision

Do not scale the current random-level checkpoint. First run the multi-fragment residual baseline for at least three seeds, then run Kintsugi with identical sampling, global batch, schedule, and validation. Promote Kintsugi only if mean external-validation AP improves beyond seed variance and false positives remain controlled. Real CUDA Mamba and fallback results must never be pooled. The staged decision protocol is in `docs/research-roadmap.md`.

## Verification

The final local check passed 84 unit/integration tests, Python compilation, a real Lightning MPS fit using exact AP, schema-v2 strict export, and strict evaluation outside Lightning. The CUDA image and real Mamba extension remain unverified until the first NVIDIA smoke run; Docker is installed locally but its daemon was not running.
