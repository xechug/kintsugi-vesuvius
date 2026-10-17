# CUDA Environment

Build from the repository root. `.dockerignore` excludes datasets, runs, checkpoints, and the local venv.

```bash
docker build -f cloud/Dockerfile -t kintsugi:torch2.6-cu124 .
```

The image lives under `/opt/kintsugi`; RunPod may mount persistent data at `/workspace` without hiding the code. The official Mamba backend is explicit: model construction fails if its CUDA extension cannot load.

## Storage Gate

For the first experiment, use one 48 GB GPU with a local volume disk and back up checkpoints externally. RunPod network volumes are portable but have variable throughput and must be benchmarked before training. With data under `/workspace/data`, test worker counts independently:

```bash
python scripts/benchmark_zarr_io.py \
  --zarr /workspace/data/public/kaggle2023_fragment2/volume.zarr \
  --samples 1024 --warmup-samples 128 \
  --patch-depth 16 --patch-height 128 --patch-width 128 \
  --workers 4 --prefetch-factor 2
```

Repeat with `0`, `2`, `4`, and `8` workers. Select the smallest count near peak throughput. Also run once with `--align-to-chunks`; training enables chunk alignment by default.

Start with an L40S 48 GB or RTX 6000 Ada 48 GB. Use an A100 80 GB only when memory profiling or `256^3` volumetric experiments justify it. Four 24 GB RTX 4090s do not create one 96 GB memory pool and add CPU, I/O, and DDP pressure; they are a throughput option only after a single GPU is demonstrably compute-bound.

## One-GPU Gate

Run the convolutional control first. Fragment 3 is external validation; test remains untouched.

```bash
python -m vesuvius_code.cloud_lightning_train \
  --data-manifest configs/kaggle2023_multifragment.runpod.json \
  --model residual_unet --backend fallback --base-channels 16 \
  --gpu-model "NVIDIA L40S 48GB" --devices 1 --precision bf16-mixed \
  --patch-depth 16 --patch-height 128 --patch-width 128 \
  --batch-size 1 --accumulate 4 \
  --workers 4 --prefetch-factor 2 \
  --epoch-size 20000 --validation-size 2048 --epochs 20 \
  --out-dir /workspace/checkpoints/residual
```

Then change only model/backend/output for the Kintsugi ablation:

```bash
python -m vesuvius_code.cloud_lightning_train \
  --data-manifest configs/kaggle2023_multifragment.runpod.json \
  --model kintsugi_axial --backend mamba --base-channels 16 \
  --gpu-model "NVIDIA L40S 48GB" --devices 1 --precision bf16-mixed \
  --patch-depth 16 --patch-height 128 --patch-width 128 \
  --batch-size 1 --accumulate 4 \
  --workers 4 --prefetch-factor 2 \
  --epoch-size 20000 --validation-size 2048 --epochs 20 \
  --out-dir /workspace/checkpoints/kintsugi
```

Checkpoints are ranked by exact distributed validation AP. Lightning state supports resume; `best.strict.ckpt` is exported for strict inference. GroupNorm is retained, so SyncBatchNorm is neither required nor enabled.

Validation uses a finite non-overlapping grid and never repeats patches to satisfy `--validation-size`; the requested value is a cap. The fragment 3 validation split currently contains 148 valid `128x128` grid patches.

Do not move to four GPUs until one GPU is compute-bound, validation AP beats prevalence across seeds, and the dataloader profiler shows negligible wait. Keep effective global batch constant when changing device count.

This container has been exercised through Lightning on MPS, but the CUDA/Mamba image has not yet been built and run on an NVIDIA host. Building the image, importing the Mamba extension, one forward/backward step, strict export, and strict reload are mandatory before a long rental.
