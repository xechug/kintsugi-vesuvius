#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

SSH_KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
chmod 600 "${SSH_KEY}"

POD_ID=""

cleanup() {
    echo ""
    echo "=========================================================="
    echo "=== [CLEANUP] Terminating all kintsugi-scroll-p800 Pods ==="
    echo "=========================================================="
    if [ -n "${POD_ID}" ]; then
        runpodctl pod delete "${POD_ID}" || true
    fi
    for pid in $(runpodctl pod list -o json 2>/dev/null | jq -r '.[] | select(.name == "kintsugi-scroll-p800") | .id' 2>/dev/null || true); do
        if [ -n "${pid}" ] && [ "${pid}" != "null" ]; then
            runpodctl pod delete "${pid}" || true
        fi
    done
    sleep 2
    echo "=== Final Account & Pod Status ==="
    runpodctl user || true
    runpodctl pod list || true
}
trap cleanup EXIT INT TERM

echo "=== 1. Checking RunPod account balance ==="
runpodctl user

echo ""
echo "=== 2. Finding or Creating RunPod Pod for Kintsugi Scroll Adaptation & PHerc0800 Virgin Scan ==="
EXISTING_ID=$(runpodctl pod list -o json 2>/dev/null | jq -r '.[] | select(.name == "kintsugi-scroll-p800") | .id' | head -n 1 || true)

if [ -n "${EXISTING_ID}" ] && [ "${EXISTING_ID}" != "null" ]; then
    echo "Reusing existing Pod: ${EXISTING_ID}"
    POD_ID="${EXISTING_ID}"
else
    CANDIDATE_GPUS=(
        "NVIDIA GeForce RTX 4090"
        "NVIDIA A40"
        "NVIDIA L40S"
    )
    CREATE_OUTPUT=""
    for GPU in "${CANDIDATE_GPUS[@]}"; do
        for CTYPE in "SECURE" "COMMUNITY"; do
            echo "Attempting to create pod with GPU: ${GPU} (${CTYPE})..."
            if CREATE_OUTPUT=$(runpodctl pod create \
                --name "kintsugi-scroll-p800" \
                --template-id "runpod-torch-v280" \
                --gpu-id "${GPU}" \
                --cloud-type "${CTYPE}" \
                --min-cuda-version "12.8" \
                --container-disk-in-gb 30 \
                --volume-in-gb 20 \
                --volume-mount-path "/workspace" \
                --ports "22/tcp" \
                --ssh 2>&1); then
                echo "Successfully placed Pod with ${GPU} (${CTYPE})!"
                break 2
            else
                echo "Placement unavailable for ${GPU} (${CTYPE}), trying next..."
            fi
        done
    done
    POD_ID=$(echo "${CREATE_OUTPUT}" | jq -r '.id // empty' 2>/dev/null || true)
    if [ -z "${POD_ID}" ] || [ "${POD_ID}" = "null" ]; then
        POD_ID=$(runpodctl pod list -o json 2>/dev/null | jq -r '.[] | select(.name == "kintsugi-scroll-p800") | .id' | head -n 1)
    fi
fi

if [ -z "${POD_ID}" ] || [ "${POD_ID}" = "null" ]; then
    echo "ERROR: Could not find pod named 'kintsugi-scroll-p800' in pod list."
    exit 1
fi
echo "Found Pod ID: ${POD_ID}"

echo "=== 3. Waiting for Container Boot & Direct SSH Connection Details ==="
SSH_HOST=""
SSH_PORT=""
for i in {1..80}; do
    POD_DETAILS=$(runpodctl pod get "${POD_ID}" -o json 2>/dev/null || true)
    SSH_HOST=$(echo "${POD_DETAILS}" | jq -r '.ssh.ip // .ssh.direct.host // empty' 2>/dev/null || true)
    SSH_PORT=$(echo "${POD_DETAILS}" | jq -r '.ssh.port // .ssh.direct.port // empty' 2>/dev/null || true)
    if [ -z "${SSH_HOST}" ] || [ "${SSH_HOST}" = "null" ]; then
        SSH_INFO=$(runpodctl ssh info "${POD_ID}" -o json 2>/dev/null || true)
        SSH_HOST=$(echo "${SSH_INFO}" | jq -r '.ip // empty' 2>/dev/null || true)
        SSH_PORT=$(echo "${SSH_INFO}" | jq -r '.port // empty' 2>/dev/null || true)
    fi
    if [ -n "${SSH_HOST}" ] && [ "${SSH_HOST}" != "null" ] && [ "${SSH_HOST}" != "ssh.runpod.io" ] && [ -n "${SSH_PORT}" ] && [ "${SSH_PORT}" != "null" ]; then
        echo "Container initialized! Direct SSH ready at ${SSH_HOST}:${SSH_PORT}"
        break
    fi
    echo "Container pulling/initializing on host (attempt $i/80)..."
    sleep 5
done

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
echo "=== 5. Syncing Code, Compressed Labels (126 KB) & Model Checkpoint (1.6 MB) to Pod ==="
${SSH_CMD} "mkdir -p /workspace/kintsugi-vesuvius/runs/cloud_kintsugi_heavy_seed1337 /workspace/kintsugi-vesuvius/scripts /workspace/kintsugi-vesuvius/data /workspace/kintsugi-vesuvius/outputs"

${SCP_CMD} -r vesuvius_code root@${SSH_HOST}:/workspace/kintsugi-vesuvius/
${SCP_CMD} scripts/train_and_scan_kintsugi_scrolls.py root@${SSH_HOST}:/workspace/kintsugi-vesuvius/scripts/
${SCP_CMD} runs/cloud_kintsugi_heavy_seed1337/best.strict.ckpt root@${SSH_HOST}:/workspace/kintsugi-vesuvius/runs/cloud_kintsugi_heavy_seed1337/
${SCP_CMD} data/scroll_adaptation_labels.npz root@${SSH_HOST}:/workspace/kintsugi-vesuvius/data/

echo ""
echo "=== 6. Running Phase 1A (2.4um w018->w013), Phase 1B (9.36um PHerc0139->w044) & Phase 2 (Virgin PHerc0800 Scan) ==="
${SSH_CMD} "bash -c '
    set -euo pipefail
    cd /workspace/kintsugi-vesuvius
    pip install -q --break-system-packages pillow numcodecs
    export PYTHONPATH=.
    python3 scripts/train_and_scan_kintsugi_scrolls.py
'"

echo ""
echo "=== 7. Downloading Checkpoints, Validation Maps & Virgin PHerc0800 Scans Locally ==="
LOCAL_OUT="${REPO_DIR}/runs/p800_and_scroll_adaptation"
mkdir -p "${LOCAL_OUT}"
${SCP_CMD} -r "root@${SSH_HOST}:/workspace/kintsugi-vesuvius/outputs/*" "${LOCAL_OUT}/"

echo "Downloaded artifacts in ${LOCAL_OUT}:"
ls -lh "${LOCAL_OUT}"

echo ""
echo "=== 8. Re-rendering Figures and English Presentation with Heavy RunPod Outputs ==="
PYTHONPATH="${REPO_DIR}" "${REPO_DIR}/venv/bin/python" "${REPO_DIR}/scripts/render_scroll_adaptation_and_p800_results.py"
PYTHONPATH="${REPO_DIR}" "${REPO_DIR}/venv/bin/python" "${REPO_DIR}/scripts/generate_english_presentation.py"

echo ""
echo "=== 9. Artifacts ready for local review ==="
echo "Review the figures and metrics before publishing any results."

