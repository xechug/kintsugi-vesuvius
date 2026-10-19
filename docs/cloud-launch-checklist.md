# GPU Cloud Deployment Guide (`Gate A`)

## 1. Cloud Platform & GPU Selection

### Platform Recommendation
- **1st Choice (Most reliable & seamless): RunPod (`Secure Cloud` or `Community Cloud`)**
  - **Why**: High-speed NVMe storage, clean official PyTorch templates, 1-click direct SSH, and persistent `/workspace` volumes.
- **2nd Choice (Lower hourly rate): Vast.ai**
  - **Why**: Lower cost per hour, but relies on community-hosted rigs with varying network/disk performance. If using Vast.ai, always filter by **`Reliability > 99%`**, **`Disk Read > 1500 MB/s`**, and **`Upload/Download > 500 Mbps`**.

### GPU Recommendation (Best Value for Kintsugi)
1. **Best Value / High VRAM (Recommended): `1x NVIDIA L40S (48 GB VRAM)` or `1x RTX 6000 Ada (48 GB VRAM)`**
   - **Cost**: ~\$0.79 – \$1.05 USD/hour on RunPod.
   - **Advantage**: Native `bf16-mixed` support, ample 48 GB VRAM for `base_channels=32`, and high-throughput Ada Lovelace architecture for `mamba-ssm`.
2. **Fast Budget Option: `1x NVIDIA RTX 4090 (24 GB VRAM)`**
   - **Cost**: ~\$0.39 – \$0.59 USD/hour.
   - **Advantage**: Highest throughput per dollar; comfortably fits `16x128x128` patches in `bf16-mixed` within 24 GB.
3. **Storage**: Allocate **50 GB Container Disk / Volume Disk** mounted at `/workspace`.

---

## 2. Cloud Execution Walkthrough

### Step A — Clone Repository & Set Up Environment
Connect via SSH to your cloud pod and run:

```bash
cd /workspace
git clone https://github.com/xechug/kintsugi-vesuvius.git
cd kintsugi-vesuvius
pip install --upgrade pip
pip install -r cloud/requirements.txt
```

### Step B — Synchronize Preprocessed Zarr Datasets (~4.2 GB without raw layer TIFFs)
The raw dataset contains uncompressed per-slice TIFFs in `layers/` (over 10.4 GB), but the training pipeline streams directly from `volume.zarr`. Syncing only `volume.zarr`, `mask.tif`, `inklabels.tif`, and `split.json` requires only **4.2 GB total**:

From your local terminal:

```bash
rsync -avzP --exclude='layers' --exclude='*.png' \
  -e "ssh -p <SSH_PORT>" \
  /Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge/data/public/ \
  root@<POD_IP>:/workspace/data/public/
```

### Step C — Launch Benchmark (`Gate A`) and Render Comparative Letters
In the cloud SSH session, launch training inside `tmux` so it runs uninterrupted:

```bash
tmux new -s kintsugi
cd /workspace/kintsugi-vesuvius

# 1. Benchmark run (seed 1337)
SEEDS="1337" BASE_CHANNELS=32 GPU_MODEL="NVIDIA RTX 4090 24GB" ./scripts/run_cloud_gate_a.sh

# 2. Render diagnostic letter comparison on held-out Fragment 3:
python scripts/render_fragment3_comparison.py \
  --residual-ckpt /workspace/checkpoints/gate_a/residual_unet_seed1337/best.strict.ckpt \
  --axial-ckpt /workspace/checkpoints/gate_a/kintsugi_axial_seed1337/best.strict.ckpt \
  --physical-ckpt /workspace/checkpoints/gate_a/kintsugi_physical_seed1337/best.strict.ckpt \
  --output-png /workspace/fragment3_gate_a_comparison.png
```
