# Kintsugi Research Roadmap

## Honest Verdict

Kintsugi is a plausible research hypothesis: local 3D convolutions model fibre texture while axial state-space mixing may enlarge context at near-linear sequence cost. It is not currently a competitive result. The 128-step residual and Kintsugi models both produce AP approximately equal to prevalence on non-overlapping validation patches.

The overfit experiment proves capacity and gradient flow, not ink understanding. The visual fibre grid proves sensitivity to papyrus structure, not correct ink discrimination.

## Research Question

Test one claim at a time:

> At matched data, parameters, optimizer steps, effective batch, and inference protocol, does axial Mamba improve cross-scroll ink ranking over a residual convolutional control?

Fallback sequence mixing on Apple Silicon and CUDA Mamba are separate model classes for reporting. Never merge their results.

## Gate A: Kaggle 2023 Surface Benchmark

1. Train `residual_unet` on fragments 1+2 for seeds `1337`, `2027`, and `9001`.
2. Train `kintsugi_axial` with the same seeds and exact compute budget.
3. Select checkpoints and thresholds only on fragment 3 validation regions.
4. Open fragment 3 test once model, seed policy, and threshold are frozen.
5. Report AP, prevalence, PR curve, Brier/ECE, and FP/Mpix at the frozen threshold.

Promotion requires all of the following:

- AP above prevalence for every seed.
- Mean Kintsugi AP gain larger than between-seed variation.
- No material regression in FP/Mpix or calibration.
- The gain survives the untouched test split.

If both models remain near prevalence, debug sampling, normalization, depth selection, and optimization before adding capacity.

## Gate B: Current Volumetric Problem

The public 2026 reference segments ink directly in full CT using a roughly 142M-parameter residual 3D U-Net, `256^3` patches, percentile normalization, Dice+BCE with label smoothing, EMA, DINO-guided supervision, and self-distillation. Its released checkpoint is about 1.70 GB.

Use the official `villa/vesuvius` implementation as the control instead of reimplementing it. The next Kintsugi experiment should replace or augment only the bottleneck while preserving the official dataset, supervision, and evaluation path.

Required data before this gate:

- Selected full-CT cubes, not the local 109-layer surface volume.
- Voxel-level expert labels for a small held-out set.
- Teacher predictions stored with model ID, threshold, source volume, and checkpoint hash.
- Spatially disjoint train/validation/test cubes, preferably from different scrolls.

Pseudo-labels may enlarge training data but must never enter independent validation. Uncertain teacher regions should be ignored or confidence-weighted rather than forced to binary truth.

## Gate C: Scientific Contribution

After a repeatable gain, publish an ablation rather than a single attractive render:

- Residual control versus Kintsugi, at matched parameters and at matched wall-clock compute.
- Z-only mixing versus Z/Y/X axial mixing.
- Unidirectional versus bidirectional scans.
- BCE+Dice versus moderate Tversky; no extreme beta without a precision-recall study.
- Three seeds, frozen splits, strict checkpoints, full run configs, and raw probability outputs.
- Failure maps reviewed against papyrus fibres, cracks, compression, and surface-estimation errors.

The useful contribution could be a negative result: showing that long-range sequence mixing does not improve local ink signal is still valuable when the protocol is reproducible.

## Compute Decision

Do not rent four GPUs for Gate A. Start with one 48 GB NVIDIA GPU and local storage, benchmark Zarr throughput, then run the residual control. Rent more GPUs only when profiling shows high GPU utilization and low dataloader wait.

For Gate B, `256^3` patches may justify an A100 80 GB, gradient checkpointing, BF16, batch size 1, and accumulation. GroupNorm remains preferable at tiny per-device batches; SyncBatchNorm is unnecessary. Keep the effective global batch and learning-rate rule fixed across device-count comparisons.

## External References

- Vesuvius Challenge current problems and goal: https://scrollprize.org/
- 2027 Grand Prize rules: https://scrollprize.org/prizes
- Public 3D DINO-guided ink model: https://huggingface.co/scrollprize/ink_3d_dino_guided
- Official implementation: https://github.com/ScrollPrize/villa
- Need for accurate 3D labels: https://github.com/ScrollPrize/villa/issues/192
