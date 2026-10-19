#!/usr/bin/env bash
set -euo pipefail

# Kintsugi Physical Heavy Training & High-Resolution Greek Letter Extraction
# Optimizations:
#   - Model: kintsugi_physical (PhysicalStem3D + SoftSheetUnwarper3D + Mamba Sequence Mixing)
#   - Base channels: 32 (4x capacity over Gate A baseline)
#   - Batch size: 8 (with bf16-mixed on 24 GB / 48 GB GPUs)
#   - Augmentation: Z-jitter (±2 slices) + Synthetic Ink (30% on low-ink patches)
#   - Regularization: Anisotropic fiber-texture suppression loss (weight 0.15)
#   - Evaluation: Full D4 Dihedral Test-Time Augmentation (8-fold TTA)

MANIFEST="${MANIFEST:-configs/kaggle2023_multifragment.runpod.json}"
OUT_ROOT="${OUT_ROOT:-/workspace/checkpoints/kintsugi_heavy}"
BACKEND="${BACKEND:-fallback}"
GPU_MODEL="${GPU_MODEL:-NVIDIA RTX 4090 24GB}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
EPOCHS="${EPOCHS:-10}"
EPOCH_SIZE="${EPOCH_SIZE:-3200}"
VAL_SIZE="${VAL_SIZE:-512}"
SEED="${SEED:-1337}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATE="${ACCUMULATE:-2}"

echo "=========================================================="
echo "  Kintsugi Physical Heavy Cloud Training & Greek Extraction"
echo "=========================================================="
echo "Manifest:       ${MANIFEST}"
echo "Output root:    ${OUT_ROOT}"
echo "Backend:        ${BACKEND}"
echo "Base channels:  ${BASE_CHANNELS}"
echo "Batch size:     ${BATCH_SIZE} (accumulate ${ACCUMULATE})"
echo "Epochs:         ${EPOCHS} (epoch size ${EPOCH_SIZE})"
echo "Seed:           ${SEED}"
echo "GPU:            ${GPU_MODEL}"
echo ""

mkdir -p "${OUT_ROOT}"

echo ">>> Starting heavy training run for kintsugi_physical..."
python -m vesuvius_code.cloud_lightning_train \
  --data-manifest "${MANIFEST}" \
  --model kintsugi_physical \
  --backend "${BACKEND}" \
  --base-channels "${BASE_CHANNELS}" \
  --z-jitter 2 \
  --synthetic-ink-prob 0.30 \
  --fiber-weight 0.15 \
  --gpu-model "${GPU_MODEL}" \
  --devices 1 \
  --precision bf16-mixed \
  --patch-depth 16 --patch-height 128 --patch-width 128 \
  --batch-size "${BATCH_SIZE}" --accumulate "${ACCUMULATE}" \
  --workers 4 --prefetch-factor 2 \
  --epoch-size "${EPOCH_SIZE}" --validation-size "${VAL_SIZE}" --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --out-dir "${OUT_ROOT}/model_seed${SEED}"

echo ""
echo ">>> Training complete. Best checkpoint:"
ls -lh "${OUT_ROOT}/model_seed${SEED}/best.strict.ckpt"

echo ""
echo ">>> Generating high-resolution diagnostic comparison with 8-fold TTA..."
RESIDUAL_CKPT="${RESIDUAL_CKPT:-/workspace/kintsugi-vesuvius/runs/cloud_gate_a_seed1337/residual_unet/best.strict.ckpt}"
AXIAL_CKPT="${AXIAL_CKPT:-/workspace/kintsugi-vesuvius/runs/cloud_gate_a_seed1337/kintsugi_axial/best.strict.ckpt}"

python scripts/render_fragment3_comparison.py \
  --residual-ckpt "${RESIDUAL_CKPT}" \
  --axial-ckpt "${AXIAL_CKPT}" \
  --physical-ckpt "${OUT_ROOT}/model_seed${SEED}/best.strict.ckpt" \
  --output-png "${OUT_ROOT}/fragment3_kintsugi_heavy_tta.png" \
  --metrics-json "${OUT_ROOT}/fragment3_kintsugi_heavy_metrics.json" \
  --tta

echo ""
echo "=========================================================="
echo "  Kintsugi Physical Heavy Pipeline Completed Successfully"
echo "=========================================================="
