# Kintsugi Scientific Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Kintsugi prototype into a leakage-resistant, reproducible ink-detection research pipeline that fails closed on invalid data and can be benchmarked locally before CUDA scaling.

**Status:** Completed and locally verified on 2026-07-15. Checklist items below preserve the original implementation sequence.

**Architecture:** Separate data provenance, sampling, model, checkpointing, inference, and evaluation into tested modules. Keep the current model available for controlled comparison, add a small residual 3D U-Net baseline, and only permit pseudo-label training when teacher scores are explicitly identified and validation uses independent ground truth.

**Tech Stack:** Python 3.14, PyTorch, NumPy, tifffile, Zarr, unittest, optional Lightning and optional mamba-ssm on CUDA.

## Global Constraints

- Public AWS access must use `--no-sign-request` with all AWS profile and credential environment variables unset.
- Never download a complete scroll without a size manifest and an explicit free-space check.
- Training must reject identical validity-mask and label files.
- Validation and test samples must be spatially disjoint from training samples.
- Checkpoint loading must be strict and architecture-versioned.
- Probability outputs must never be min-max stretched before scientific evaluation.
- The project root is not a Git repository, so commit steps are recorded as unavailable rather than simulated.

---

### Task 1: Data Contract And Lazy Raster Access

**Files:**
- Create: `vesuvius_code/data_contract.py`
- Create: `tests/test_data_contract.py`
- Modify: `vesuvius_code/dataset.py`

**Interfaces:**
- Produces: `audit_training_data(volume_path, valid_mask_path, label_path, label_kind) -> DataAuditReport`
- Produces: `open_raster(path) -> LazyRaster`
- Produces: `DataContractError`

- [ ] **Step 1: Write failing tests** proving identical files are rejected, non-binary validity masks are rejected, soft teacher labels require `label_kind="teacher"`, and TIFF tiles are read lazily.
- [ ] **Step 2: Run** `./venv/bin/python -m unittest tests.test_data_contract -v` and confirm failures are caused by missing interfaces.
- [ ] **Step 3: Implement** streamed SHA-256, deterministic chunk sampling, shape/dtype inspection, JSON-serializable reports, and lazy TIFF/Zarr access. The public API must support:

```python
report = audit_training_data(volume, valid_mask, labels, label_kind="ground_truth")
report.raise_for_training()
```

- [ ] **Step 4: Run** the focused test and the complete test suite.

### Task 2: Spatial Splits And Epoch-Aware Sampling

**Files:**
- Create: `vesuvius_code/splits.py`
- Create: `tests/test_spatial_sampling.py`
- Modify: `vesuvius_code/dataset.py`

**Interfaces:**
- Produces: `build_spatial_split(height, width, block_size, guard_band, fractions, seed) -> SpatialSplitManifest`
- Consumes: allowed rectangular regions from `SpatialSplitManifest.regions(split)`
- Produces dataset batches with `image`, `target`, `valid_mask`, and `coords`.

- [ ] **Step 1: Write failing tests** proving train/validation footprints do not overlap after patch expansion, patches are classified by label coverage rather than their top-left pixel, and train samples change after `set_epoch(1)` while validation remains fixed.
- [ ] **Step 2: Run** `./venv/bin/python -m unittest tests.test_spatial_sampling -v` and verify expected failures.
- [ ] **Step 3: Implement** block manifests, guard bands, coarse integral-image coverage, shared epoch state, per-epoch jitter, and valid-mask return values.
- [ ] **Step 4: Run** focused and complete tests.

### Task 3: Valid-Aware Losses And Metrics

**Files:**
- Modify: `vesuvius_code/losses.py`
- Modify: `vesuvius_code/metrics.py`
- Create: `tests/test_losses_metrics.py`

**Interfaces:**
- Produces: `FableLoss.forward(logits, target, valid_mask=None)`
- Produces: `binary_segmentation_metrics(probs, target, threshold, valid_mask=None)`
- Produces: calibration metrics from accumulated probabilities and labels.

- [ ] **Step 1: Write failing tests** proving invalid pixels contribute zero gradient and zero confusion counts, empty-target Tversky is finite, and Brier score uses only valid pixels.
- [ ] **Step 2: Run** the focused test and inspect the expected assertion failures.
- [ ] **Step 3: Implement** masked BCE/Tversky reductions and valid-aware metrics without changing existing unmasked behavior.
- [ ] **Step 4: Run** focused and complete tests.

### Task 4: Versioned Strict Checkpoints

**Files:**
- Create: `vesuvius_code/checkpoints.py`
- Create: `tests/test_checkpoints.py`
- Modify: `vesuvius_code/train.py`
- Modify: `vesuvius_code/evaluate_checkpoint.py`

**Interfaces:**
- Produces: `save_checkpoint(path, model, optimizer, scheduler, scaler, run_config, data_audit, epoch, global_step)`
- Produces: `load_checkpoint(path, model, expected_config=None, strict=True) -> dict`

- [ ] **Step 1: Write failing tests** for architecture mismatch, missing keys, RNG restoration metadata, and atomic save replacement.
- [ ] **Step 2: Run** the focused test and confirm failures.
- [ ] **Step 3: Implement** schema version 2 checkpoints with model config, backend, normalization, data hashes, optimizer/scheduler/scaler state, RNG state, epoch, and step. Reject partial model loads.
- [ ] **Step 4: Update** manual training and evaluation to consume the strict API, then run all tests.

### Task 5: Reproducible Sliding-Window Inference

**Files:**
- Create: `vesuvius_code/inference.py`
- Create: `tests/test_inference.py`
- Modify: `inference.py`

**Interfaces:**
- Produces: `normalization_divisor(dtype)`, `sliding_origins(length, patch, stride)`, `hann_window_2d(height, width)`, and `predict_region(...)`.

- [ ] **Step 1: Write failing tests** proving uint8 and uint16 normalization, final-border coverage, strict checkpoint rejection, and seam-free reconstruction of constant predictions.
- [ ] **Step 2: Run** focused tests and verify missing-function failures.
- [ ] **Step 3: Implement** dtype-aware normalization, halo-safe origins, Hann blending, configurable regions and Z windows, float probability TIFF output, and separate optional visualization export.
- [ ] **Step 4: Run** focused and complete tests.

### Task 6: Honest Model Backends And Baseline

**Files:**
- Modify: `vesuvius_code/model.py`
- Create: `vesuvius_code/baselines.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces: explicit backend values `fallback`, `mamba`, and `auto` with backend metadata.
- Produces: `ResidualUNet3DTo2D` as a convolutional baseline.

- [ ] **Step 1: Write failing tests** proving an explicitly requested unavailable Mamba raises, special `_no_reinit` parameters are preserved, depth reversal has shape parity, and the baseline backpropagates.
- [ ] **Step 2: Run** focused tests and verify failures.
- [ ] **Step 3: Implement** explicit backend reporting, protected initialization, bidirectional depth mixing, and the small residual multiscale baseline. Keep Kintsugi output compatible with 2D labels until verified 3D labels are available.
- [ ] **Step 4: Run** focused and complete tests and record parameter counts.

### Task 7: Validation-First Lightning And DDP Configuration

**Files:**
- Create: `vesuvius_code/training_config.py`
- Create: `tests/test_training_config.py`
- Modify: `vesuvius_code/cloud_lightning_train.py`

**Interfaces:**
- Produces: `TrainingConfig.validate()` and `effective_global_batch_size`.
- Consumes: spatial split manifest, strict data audit, dictionary batches, and versioned checkpoints.

- [ ] **Step 1: Write failing tests** proving T4 rejects BF16, global batch calculation is correct, worker/prefetch memory estimates are bounded, and validation is mandatory.
- [ ] **Step 2: Run** focused tests and verify failures.
- [ ] **Step 3: Implement** deterministic seeding, train and validation dataloaders, epoch propagation, validation loss and counts, `val_loss` checkpoint monitoring, full-state resume, and rank-aware logging. Retain GroupNorm and do not add SyncBatchNorm.
- [ ] **Step 4: Compile** the Lightning module locally and run pure configuration tests; CUDA behavior remains a cloud smoke-test gate.

### Task 8: Safe Public-Data Discovery And Experiment Manifest

**Files:**
- Create: `scripts/public_data_inventory.py`
- Create: `tests/test_public_data_inventory.py`
- Create: `configs/kintsugi_local_baseline.json`
- Create: `docs/data-provenance.md`

**Interfaces:**
- Produces: a size-first public S3 inventory and explicit allow-list download commands.
- Produces: a run manifest binding code configuration, data hashes, voxel size, scroll, segment, and label provenance.

- [ ] **Step 1: Write failing tests** for public unsigned command construction, traversal rejection, byte-budget enforcement, and metadata parsing.
- [ ] **Step 2: Run** focused tests and verify failures.
- [ ] **Step 3: Implement** read-only inventory generation with credentials unset, size summaries, disk-space checks, and opt-in object downloads.
- [ ] **Step 4: Inventory** Paris 4 and small labeled fragments, download only metadata and selected verified samples within the configured byte budget, and write provenance documentation.

### Task 9: End-To-End Scientific Smoke Test

**Files:**
- Create: `vesuvius_code/smoke_test.py`
- Create: `tests/test_scientific_smoke.py`
- Modify: `README.md` if present; otherwise create `README.md`.

**Interfaces:**
- Consumes all preceding public APIs.
- Produces a machine-readable run report and exits nonzero on data leakage, partial checkpoint loads, invalid split overlap, or non-finite metrics.

- [ ] **Step 1: Write a failing synthetic end-to-end test** with two disjoint spatial regions and a tiny Zarr volume.
- [ ] **Step 2: Run** the focused test and verify the smoke interface is missing.
- [ ] **Step 3: Implement** audit, split, one training step, strict checkpoint round-trip, tiled inference, and valid-aware evaluation.
- [ ] **Step 4: Run** `./venv/bin/python -m unittest discover -s tests -v`, `./venv/bin/python -m compileall -q vesuvius_code scripts`, and the local smoke CLI.

## Self-Review

- Spec coverage: data validity, leakage, sampling, masks, loss, metrics, checkpoints, inference, architecture, DDP, I/O, public acquisition, and local verification each have an owning task.
- Placeholder scan: no deferred implementation placeholders are present.
- Type consistency: data reports, split manifests, dictionary batches, checkpoint schema, and training configuration have one named producer and explicit consumers.
