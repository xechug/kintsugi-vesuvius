#!/usr/bin/env bash
set -euo pipefail

# Automated Runpod Pod Provisioner & Data Sync for Kintsugi Gate A
# Uses runpodctl v3+ configured with ~/.runpod/config.toml / RUNPOD_API_KEY.
#
# Recommended GPUs (from live catalog):
#   - "NVIDIA GeForce RTX 4090" (24 GB VRAM) — $0.34/hr Community | $0.89/hr Secure
#   - "NVIDIA A40"              (48 GB VRAM) — $0.59/hr Secure
#   - "NVIDIA L4"               (24 GB VRAM) — $0.59/hr Secure
#   - "NVIDIA L40S"             (48 GB VRAM) — $0.79/hr Community | $1.09/hr Secure

POD_NAME="${POD_NAME:-kintsugi-vesuvius-gate-a}"
TEMPLATE_ID="${TEMPLATE_ID:-runpod-torch-v280}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
MIN_CUDA="${MIN_CUDA:-12.8}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-30}"
VOLUME_GB="${VOLUME_GB:-50}"

echo "=== 1. Checking Runpod Account Balance ==="
BALANCE=$(runpodctl user | jq -r '.clientBalance // 0')
echo "Current balance: \$${BALANCE}"
if awk "BEGIN {exit !(${BALANCE} <= 0.5)}"; then
  echo "ERROR: Balance (\$${BALANCE}) is too low to provision a Pod."
  echo "Top up at least \$5-\$10 at https://www.runpod.io/console/user/billing and re-run this script."
  exit 1
fi

echo ""
echo "=== 2. Provisioning Pod (${POD_NAME}) ==="
echo "  Template:   ${TEMPLATE_ID} (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)"
echo "  GPU:        ${GPU_ID} (${CLOUD_TYPE})"
echo "  Disk:       ${CONTAINER_DISK_GB} GB container + ${VOLUME_GB} GB /workspace volume"

CREATE_JSON=$(runpodctl pod create \
  --name "${POD_NAME}" \
  --template-id "${TEMPLATE_ID}" \
  --gpu-id "${GPU_ID}" \
  --cloud-type "${CLOUD_TYPE}" \
  --min-cuda-version "${MIN_CUDA}" \
  --container-disk-in-gb "${CONTAINER_DISK_GB}" \
  --volume-in-gb "${VOLUME_GB}" \
  --volume-mount-path "/workspace" \
  --ports "22/tcp,8888/http" \
  --ssh \
  --wait \
  --wait-timeout 10m)

echo "${CREATE_JSON}" | jq .
POD_ID=$(echo "${CREATE_JSON}" | jq -r '.id')
echo ""
echo "=== 3. Pod Ready: ${POD_ID} ==="
runpodctl ssh info "${POD_ID}"

echo ""
echo "=== 4. Next Step: Sync 4.2 GB Zarr Fragments (Excludes 10.4 GB legacy layers/*.tif) ==="
echo "Run the rsync command using the SSH host/port above:"
echo "  rsync -avP --exclude='layers/' data/public/kaggle2023_fragment1 data/public/kaggle2023_fragment2 data/public/kaggle2023_fragment3 root@<SSH_HOST>:/workspace/data/kaggle2023/"
