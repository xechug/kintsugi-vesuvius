#!/usr/bin/env bash
set -euo pipefail

# Gate A Controlled Multi-Fragment Benchmark (Fragments 1+2 -> Fragment 3)
# Compares:
#   1. residual_unet (Convolutional control)
#   2. kintsugi_axial (Sequence-mixing control without physics)
#   3. kintsugi_physical (Full Physics: PhysicalStem3D + SoftSheetUnwarper3D + Z-jitter + Voronoi Crackle + Fiber Loss)

MANIFEST="${MANIFEST:-configs/kaggle2023_multifragment.runpod.json}"
OUT_ROOT="${OUT_ROOT:-/workspace/checkpoints/gate_a}"
BACKEND="${BACKEND:-mamba}"
GPU_MODEL="${GPU_MODEL:-NVIDIA L40S 48GB}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
EPOCHS="${EPOCHS:-20}"
EPOCH_SIZE="${EPOCH_SIZE:-20000}"
VAL_SIZE="${VAL_SIZE:-2048}"
SEEDS="${SEEDS:-1337 2027 9001}"

echo "=== Kintsugi Gate A Cloud Runner ==="
echo "Manifest:      ${MANIFEST}"
echo "Output root:   ${OUT_ROOT}"
echo "Backend:       ${BACKEND}"
echo "Base channels: ${BASE_CHANNELS}"
echo "Seeds:         ${SEEDS}"

for SEED in ${SEEDS}; do
  echo ""
  echo ">>> [Seed ${SEED}] 1/3 Training residual_unet control..."
  python -m vesuvius_code.cloud_lightning_train \
    --data-manifest "${MANIFEST}" \
    --model residual_unet \
    --backend fallback \
    --base-channels "${BASE_CHANNELS}" \
    --gpu-model "${GPU_MODEL}" \
    --devices 1 \
    --precision bf16-mixed \
    --patch-depth 16 --patch-height 128 --patch-width 128 \
    --batch-size 1 --accumulate 4 \
    --workers 4 --prefetch-factor 2 \
    --epoch-size "${EPOCH_SIZE}" --validation-size "${VAL_SIZE}" --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --out-dir "${OUT_ROOT}/residual_unet_seed${SEED}"

  echo ""
  echo ">>> [Seed ${SEED}] 2/3 Training kintsugi_axial control..."
  python -m vesuvius_code.cloud_lightning_train \
    --data-manifest "${MANIFEST}" \
    --model kintsugi_axial \
    --backend "${BACKEND}" \
    --base-channels "${BASE_CHANNELS}" \
    --gpu-model "${GPU_MODEL}" \
    --devices 1 \
    --precision bf16-mixed \
    --patch-depth 16 --patch-height 128 --patch-width 128 \
    --batch-size 1 --accumulate 4 \
    --workers 4 --prefetch-factor 2 \
    --epoch-size "${EPOCH_SIZE}" --validation-size "${VAL_SIZE}" --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --out-dir "${OUT_ROOT}/kintsugi_axial_seed${SEED}"

  echo ""
  echo ">>> [Seed ${SEED}] 3/3 Training kintsugi_physical (Full Physics Options 1+2+3)..."
  python -m vesuvius_code.cloud_lightning_train \
    --data-manifest "${MANIFEST}" \
    --model kintsugi_physical \
    --backend "${BACKEND}" \
    --base-channels "${BASE_CHANNELS}" \
    --z-jitter 2 \
    --synthetic-ink-prob 0.25 \
    --fiber-weight 0.15 \
    --gpu-model "${GPU_MODEL}" \
    --devices 1 \
    --precision bf16-mixed \
    --patch-depth 16 --patch-height 128 --patch-width 128 \
    --batch-size 1 --accumulate 4 \
    --workers 4 --prefetch-factor 2 \
    --epoch-size "${EPOCH_SIZE}" --validation-size "${VAL_SIZE}" --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --out-dir "${OUT_ROOT}/kintsugi_physical_seed${SEED}"
done

echo ""
echo "=== Gate A Training Complete ==="
