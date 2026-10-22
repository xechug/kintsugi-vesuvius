#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge"
cd "${REPO_DIR}"

SSH_KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
chmod 600 "${SSH_KEY}"

POD_ID=""

cleanup() {
    if [ -n "${POD_ID}" ]; then
        echo ""
        echo "=========================================================="
        echo "=== [CLEANUP] Terminating Pod ${POD_ID} to STOP billing ==="
        echo "=========================================================="
        runpodctl pod delete "${POD_ID}" || true
        sleep 2
        echo "=== Final Account & Pod Status ==="
        runpodctl user || true
        runpodctl pod list || true
    fi
}
trap cleanup EXIT INT TERM

echo "=== 1. Checking account balance ==="
runpodctl user

echo ""
echo "=== 2. Creating RunPod Pod ==="
CANDIDATE_GPUS=(
    "NVIDIA RTX PRO 4000 Blackwell"
    "NVIDIA RTX PRO 4500 Blackwell Server Edition"
    "NVIDIA A40"
    "NVIDIA GeForce RTX 4090"
)

CREATE_OUTPUT=""
for GPU in "${CANDIDATE_GPUS[@]}"; do
    echo "Attempting to create pod with GPU: ${GPU}..."
    if CREATE_OUTPUT=$(runpodctl pod create \
        --name "kintsugi-scroll4-scan" \
        --template-id "runpod-torch-v280" \
        --gpu-id "${GPU}" \
        --cloud-type "SECURE" \
        --container-disk-in-gb 30 \
        --volume-in-gb 20 \
        --volume-mount-path "/workspace" \
        --ports "22/tcp" \
        --ssh \
        --wait \
        --wait-timeout "5m" 2>&1); then
        echo "Successfully provisioned with ${GPU}!"
        break
    else
        echo "Failed with ${GPU}, trying next..."
    fi
done

# Extract POD_ID, SSH_HOST, and SSH_PORT directly
POD_ID=$(echo "${CREATE_OUTPUT}" | jq -r '.id // empty' 2>/dev/null || true)
if [ -z "${POD_ID}" ] || [ "${POD_ID}" = "null" ]; then
    POD_ID=$(runpodctl pod list -o json 2>/dev/null | jq -r '.[] | select(.name == "kintsugi-scroll4-scan") | .id' | head -n 1)
fi

if [ -z "${POD_ID}" ] || [ "${POD_ID}" = "null" ]; then
    echo "ERROR: Could not find pod named 'kintsugi-scroll4-scan' in pod list."
    exit 1
fi
echo "Found Pod ID: ${POD_ID}"

# Retrieve SSH details
echo "=== 3. Extracting SSH Connection Details ==="
SSH_HOST=$(echo "${CREATE_OUTPUT}" | jq -r '.ssh.ip // empty' 2>/dev/null || true)
SSH_PORT=$(echo "${CREATE_OUTPUT}" | jq -r '.ssh.port // empty' 2>/dev/null || true)

if [ -z "${SSH_HOST}" ] || [ "${SSH_HOST}" = "null" ] || [ -z "${SSH_PORT}" ] || [ "${SSH_PORT}" = "null" ]; then
    for i in {1..12}; do
        POD_DETAILS=$(runpodctl pod get "${POD_ID}" -o json 2>/dev/null || true)
        SSH_HOST=$(echo "${POD_DETAILS}" | jq -r '.ssh.ip // empty' 2>/dev/null || true)
        SSH_PORT=$(echo "${POD_DETAILS}" | jq -r '.ssh.port // empty' 2>/dev/null || true)
        if [ -n "${SSH_HOST}" ] && [ "${SSH_HOST}" != "null" ] && [ -n "${SSH_PORT}" ] && [ "${SSH_PORT}" != "null" ]; then
            break
        fi
        echo "Waiting for SSH info in pod get (attempt $i/12)..."
        sleep 3
    done
fi

if [ -z "${SSH_HOST}" ] || [ "${SSH_HOST}" = "null" ] || [ -z "${SSH_PORT}" ] || [ "${SSH_PORT}" = "null" ]; then
    SSH_INFO=$(runpodctl ssh info "${POD_ID}" -o json 2>/dev/null || true)
    SSH_HOST=$(echo "${SSH_INFO}" | jq -r '.ip // empty' 2>/dev/null || true)
    SSH_PORT=$(echo "${SSH_INFO}" | jq -r '.port // empty' 2>/dev/null || true)
fi

echo "SSH Host: ${SSH_HOST}, SSH Port: ${SSH_PORT}"

if [ -z "${SSH_HOST}" ] || [ -z "${SSH_PORT}" ]; then
    echo "ERROR: Could not retrieve SSH host and port."
    exit 1
fi

SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i ${SSH_KEY} -p ${SSH_PORT} root@${SSH_HOST}"
SCP_CMD="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i ${SSH_KEY} -P ${SSH_PORT}"

echo "=== 4. Waiting for SSH to be ready ==="
for i in {1..20}; do
    if ${SSH_CMD} "echo 'SSH is up' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null; then
        echo "GPU and SSH confirmed ready!"
        break
    fi
    echo "Waiting for SSH response (attempt $i/20)..."
    sleep 4
done

echo ""
echo "=== 5. Syncing Code & Model Checkpoint to Pod ==="
${SSH_CMD} "mkdir -p /workspace/kintsugi-vesuvius/runs/cloud_kintsugi_heavy_seed1337 /workspace/kintsugi-vesuvius/scripts"

echo "Copying vesuvius_code..."
${SCP_CMD} -r vesuvius_code root@${SSH_HOST}:/workspace/kintsugi-vesuvius/

echo "Copying scripts/scan_scroll4_virgin.py..."
${SCP_CMD} scripts/scan_scroll4_virgin.py root@${SSH_HOST}:/workspace/kintsugi-vesuvius/scripts/

echo "Copying model checkpoint (1.6 MB)..."
${SCP_CMD} runs/cloud_kintsugi_heavy_seed1337/best.strict.ckpt root@${SSH_HOST}:/workspace/kintsugi-vesuvius/runs/cloud_kintsugi_heavy_seed1337/

SEGMENT="${SEGMENT:-20240304161941-w023_20240304161941_flatboi}"
Y_START="${Y_START:-65}"
NUM_Y_CHUNKS="${NUM_Y_CHUNKS:-20}"
X_START="${X_START:-345}"
NUM_X_CHUNKS="${NUM_X_CHUNKS:-25}"
OUT_NAME="${OUT_NAME:-scroll4_w023_hotspot1_kintsugi}"

echo ""
echo "=== 6. Installing Dependencies & Running Inference on Virgin Scroll 4 (${SEGMENT}: Y=${Y_START}, X=${X_START}) ==="
${SSH_CMD} "bash -c '
    set -euo pipefail
    cd /workspace/kintsugi-vesuvius
    pip install -q --break-system-packages matplotlib tifffile pillow
    export PYTHONPATH=.
    python3 scripts/scan_scroll4_virgin.py \
        --segment ${SEGMENT} \
        --ckpt runs/cloud_kintsugi_heavy_seed1337/best.strict.ckpt \
        --y-start ${Y_START} \
        --num-y-chunks ${NUM_Y_CHUNKS} \
        --x-start ${X_START} \
        --num-x-chunks ${NUM_X_CHUNKS} \
        --batch-size 32 \
        --out-png runs/${OUT_NAME}_prediction.png
'"

echo ""
echo "=== 7. Retrieving Results to Local Machine & Artifacts ==="
mkdir -p "${REPO_DIR}/runs"
${SCP_CMD} root@${SSH_HOST}:/workspace/kintsugi-vesuvius/runs/${OUT_NAME}_prediction* "${REPO_DIR}/runs/"

ARTIFACT_DIR="/Users/jsmcore/.gemini/antigravity/brain/facf1ebf-2061-4f7f-b872-c0129914aef4"
cp "${REPO_DIR}/runs/${OUT_NAME}_prediction"* "${ARTIFACT_DIR}/"

echo "Results successfully transferred!"
ls -lh "${REPO_DIR}/runs/${OUT_NAME}_prediction"*
