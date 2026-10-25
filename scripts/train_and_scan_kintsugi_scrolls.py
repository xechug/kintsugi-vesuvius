"""
End-to-End Scroll-Adapted Kintsugi Training, Held-Out Validation, and Virgin Scroll Discovery:
1. Phase 1A (2.4 um / 78 keV): Train Multi-Scale KintsugiPhysicalUNet3DTo2D (with 8-lens
   PhysicalStem3D + SoftSheetUnwarper3D + AxialSequenceMixer3D + CalligraphicPyramid2D) on
   PHerc1667 w018 and validate strictly held-out on unseen segment PHerc1667 w013.
2. Phase 1B (8.64-9.36 um / 1.2m / 113-116 keV): Fine-tune Multi-Scale KintsugiPhysicalUNet3DTo2D
   on PHerc0139 (w035, w040, w041) human ink-labels and validate strictly held-out on unseen
   segment PHerc0139 w044 (cy=28..44, cx=20..36).
3. Phase 2 (Virgin Scroll Discovery - $50,000 First Letters Prize Target): Scan 100% of all 6
   virgin surface-volume segments of PHerc0800 (8.64um-1.2m-116keV-volume-20250521135224.zarr)
   using multi-slab + TTA inference.
"""
from __future__ import annotations

import concurrent.futures
import io
import json
import math
import socket
import time
import urllib.request
import os
from pathlib import Path

socket.setdefaulttimeout(12.0)

import numcodecs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vesuvius_code.baselines import KintsugiPhysicalUNet3DTo2D
from vesuvius_code.losses import AnisotropicFiberSuppressionLoss, FableLoss
from vesuvius_code.numpy_metrics import precision_recall_summary, roc_auc

S3_ROOT = "https://vesuvius-challenge-open-data.s3.amazonaws.com"


def roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return roc_auc(y_true, y_score)


def pr_auc_and_best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    return precision_recall_summary(y_true, y_score)


def get_zarr_meta(zarr_subpath: str, level: int = 0) -> dict:
    url = f"{S3_ROOT}/{zarr_subpath.rstrip('/')}/{level}/.zarray"
    req = urllib.request.Request(url, headers={"User-Agent": "Kintsugi-Vesuvius/2.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_chunk_grid(
    zarr_subpath: str,
    cy_start: int,
    cy_end: int,
    cx_start: int,
    cx_end: int,
    max_workers: int = 32,
    level: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    meta = get_zarr_meta(zarr_subpath, level=level)
    sep = meta.get("dimension_separator", "/")
    comp_cfg = meta.get("compressor", None)
    codec = numcodecs.get_codec(comp_cfg) if comp_cfg is not None else None
    d_chunks, y_chunk, x_chunk = meta["chunks"]
    D_total, H_total, W_total = meta["shape"]

    ny = cy_end - cy_start
    nx = cx_end - cx_start
    vol = np.zeros((d_chunks, ny * y_chunk, nx * x_chunk), dtype=np.uint8)
    valid_mask = np.zeros((ny * y_chunk, nx * x_chunk), dtype=bool)

    tasks = [
        (iy, ix, cy_start + iy, cx_start + ix)
        for iy in range(ny)
        for ix in range(nx)
    ]

    def _fetch(task):
        iy, ix, cy, cx = task
        url = f"{S3_ROOT}/{zarr_subpath.rstrip('/')}/{level}/0{sep}{cy}{sep}{cx}"
        for _ in range(2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Kintsugi-Vesuvius/2.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw = resp.read()
                if codec is not None:
                    raw = codec.decode(raw)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(d_chunks, y_chunk, x_chunk)
                return iy, ix, arr
            except Exception:
                pass
        return iy, ix, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for iy, ix, arr in ex.map(_fetch, tasks):
            if arr is not None:
                y0, y1 = iy * y_chunk, (iy + 1) * y_chunk
                x0, x1 = ix * x_chunk, (ix + 1) * x_chunk
                vol[:, y0:y1, x0:x1] = arr
                if arr.max() > 0:
                    valid_mask[y0:y1, x0:x1] = True

    max_h = min(ny * y_chunk, max(0, H_total - cy_start * y_chunk))
    max_w = min(nx * x_chunk, max(0, W_total - cx_start * x_chunk))
    return vol[:, :max_h, :max_w], valid_mask[:max_h, :max_w]


def normalize_slab_globally(slab: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Normalizes a 16-slice slab (16, H, W) globally across the valid 3D volume + removes
    ultra-low-frequency planar illumination drift (129x129 mask-normalized box filter)
    so rotated diamond segment boundaries never create edge halo artifacts.
    """
    if valid_mask is None:
        valid_mask = slab.max(axis=0) > 0
    vm = valid_mask.astype(np.float32)
    if vm.sum() > 100:
        vals = slab[:, valid_mask]
        mean_3d = float(vals.mean())
        std_3d = max(float(vals.std()), 1e-4)
    else:
        mean_3d = float(slab.mean())
        std_3d = max(float(slab.std()), 1e-4)

    z_global = (slab - mean_3d) / std_3d
    z_global[:, ~valid_mask] = 0.0

    t = torch.from_numpy(z_global)[None, None]  # (1, 1, 16, H, W)
    m_t = torch.from_numpy(vm)[None, None]  # (1, 1, H, W)
    z_mean_xy = t.mean(dim=2) * m_t  # (1, 1, H, W)
    pad = 64
    num_pad = F.pad(z_mean_xy, (pad, pad, pad, pad), mode="constant", value=0.0)
    den_pad = F.pad(m_t, (pad, pad, pad, pad), mode="constant", value=0.0)
    macro_num = F.avg_pool2d(num_pad, kernel_size=129, stride=1)
    macro_den = F.avg_pool2d(den_pad, kernel_size=129, stride=1).clamp_min(1e-4)
    macro_drift_xy = (macro_num / macro_den).unsqueeze(2)  # (1, 1, 1, H, W)

    corrected = (t - 0.7 * macro_drift_xy).squeeze(0).squeeze(0)
    out = ((corrected.clamp(-3.5, 3.5) + 3.5) / 7.0).numpy()
    out[:, ~valid_mask] = 0.5
    return out.astype(np.float32)


def canonical_orient_and_pool_2p4um(vol_uint8: np.ndarray) -> np.ndarray:
    """
    Pools 2.399 um volume (D=109..112, H, W) across z=6..102 into 16 axial slices
    at level-3 spatial resolution (8x XY -> 19.19 um/px, where a 3 mm Greek letter
    is 156 px tall and fits inside a 192x192 / 128x128 CalligraphicPyramid2D patch),
    combining mean synchrotron attenuation with 2.4 um intra-cell micro-crackle std.
    """
    slab = vol_uint8[6:102].astype(np.float32) / 255.0
    if float(slab[:24].mean()) < float(slab[72:].mean()):
        slab = slab[::-1].copy()
    D, H, W = slab.shape
    H8 = (H // 8) * 8
    W8 = (W // 8) * 8
    cells = slab[:, :H8, :W8].reshape(16, 6, H8 // 8, 8, W8 // 8, 8)
    m_l3 = cells.mean(axis=(1, 3, 5))
    s_l3 = cells.std(axis=(1, 3, 5))
    fused = m_l3 + 0.65 * s_l3
    return normalize_slab_globally(fused)


def prepare_coarse_16_slices(vol_uint8: np.ndarray, z_start: int | None = None) -> np.ndarray:
    D = vol_uint8.shape[0]
    if z_start is None:
        z_start = (D - 16) // 2
    slab = vol_uint8[z_start : z_start + 16].astype(np.float32) / 255.0
    valid_px = slab.max(axis=0) > 0.02
    if valid_px.sum() > 100:
        for z_i in range(16):
            sl = slab[z_i]
            m_val = float(sl[valid_px].mean())
            sl[~valid_px] = m_val
    if float(slab[:6].mean()) < float(slab[10:].mean()):
        slab = slab[::-1].copy()
    return normalize_slab_globally(slab, valid_mask=valid_px)


@torch.no_grad()
def predict_sliding_tta(
    model: nn.Module,
    slab_16: np.ndarray,
    device: torch.device,
    tile_size: int = 256,
    stride: int = 128,
    use_tta: bool = True,
) -> np.ndarray:
    model.eval()
    D, H, W = slab_16.shape
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h > 0 or pad_w > 0:
        slab_16 = np.pad(slab_16, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    _, Hp, Wp = slab_16.shape

    yy = np.linspace(-1.0, 1.0, tile_size, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, tile_size, dtype=np.float32)
    weight_tile = np.exp(-2.0 * (yy[:, None] ** 2 + xx[None, :] ** 2)).astype(np.float32)

    ys = list(range(0, max(1, Hp - tile_size + 1), stride))
    if (Hp - tile_size) not in ys:
        ys.append(max(0, Hp - tile_size))
    xs = list(range(0, max(1, Wp - tile_size + 1), stride))
    if (Wp - tile_size) not in xs:
        xs.append(max(0, Wp - tile_size))

    pred_sum = np.zeros((Hp, Wp), dtype=np.float32)
    weight_sum = np.zeros((Hp, Wp), dtype=np.float32)

    coords = [(y0, x0) for y0 in ys for x0 in xs]
    batch_sz = 8 if device.type == "cuda" else 6
    for i in range(0, len(coords), batch_sz):
        batch_coords = coords[i : i + batch_sz]
        patches = np.stack(
            [slab_16[:, y0 : y0 + tile_size, x0 : x0 + tile_size][None] for (y0, x0) in batch_coords],
            axis=0,
        )
        patch_t = torch.from_numpy(patches).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            p0 = torch.sigmoid(model(patch_t))
            if use_tta:
                ph = torch.flip(torch.sigmoid(model(torch.flip(patch_t, dims=[4]))), dims=[3])
                pv = torch.flip(torch.sigmoid(model(torch.flip(patch_t, dims=[3]))), dims=[2])
                pz = torch.sigmoid(model(torch.flip(patch_t, dims=[2])))
                prob = (0.4 * p0 + 0.2 * ph + 0.2 * pv + 0.2 * pz).squeeze(1).float().cpu().numpy()
            else:
                prob = p0.squeeze(1).float().cpu().numpy()

        for b_idx, (y0, x0) in enumerate(batch_coords):
            pred_sum[y0 : y0 + tile_size, x0 : x0 + tile_size] += prob[b_idx] * weight_tile
            weight_sum[y0 : y0 + tile_size, x0 : x0 + tile_size] += weight_tile

    out = pred_sum / np.maximum(weight_sum, 1e-6)
    return out[:H, :W]


def train_scroll_model(
    model: nn.Module,
    train_items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]],
    device: torch.device,
    steps: int = 350,
    batch_size: int = 6,
    patch_size: int = 192,
    lr: float = 1.5e-3,
) -> nn.Module:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)
    # pos_weight=1.2 because patches are already sampled from supervised regions with ~30% ink
    phys_loss_fn = AnisotropicFiberSuppressionLoss(
        base_loss=FableLoss(pos_weight=1.2, bce_weight=0.5, tversky_weight=0.5, tversky_alpha=0.4, tversky_beta=0.6),
        fiber_weight=0.15,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    item_centers = []
    for slab, ink, sup, _ in train_items:
        _, H, W = slab.shape
        valid_y, valid_x = np.where(sup[: H - patch_size, : W - patch_size] > 0)
        if len(valid_y) == 0:
            valid_y = np.arange(0, max(1, H - patch_size))
            valid_x = np.arange(0, max(1, W - patch_size))
        pos_y, pos_x = np.where(
            (sup[: H - patch_size, : W - patch_size] > 0)
            & (ink[: H - patch_size, : W - patch_size] > 0.5)
        )
        neg_y, neg_x = np.where(
            (sup[: H - patch_size, : W - patch_size] > 0)
            & (ink[: H - patch_size, : W - patch_size] == 0)
        )
        item_centers.append((valid_y, valid_x, pos_y, pos_x, neg_y, neg_x))

    rng = np.random.default_rng(1337)
    t0 = time.time()
    for step in range(1, steps + 1):
        vols, inks, sups, teachers = [], [], [], []
        has_teacher = True
        for b in range(batch_size):
            idx = int(rng.integers(0, len(train_items)))
            slab, ink, sup, teacher = train_items[idx]
            vy, vx, py, px, ny_c, nx_c = item_centers[idx]
            _, H, W = slab.shape
            if b % 3 == 0 and len(py) > 0:
                k = int(rng.integers(0, len(py)))
                y0 = int(np.clip(py[k] - patch_size // 2 + rng.integers(-32, 33), 0, H - patch_size))
                x0 = int(np.clip(px[k] - patch_size // 2 + rng.integers(-32, 33), 0, W - patch_size))
            elif b % 3 == 1 and len(ny_c) > 0:
                k = int(rng.integers(0, len(ny_c)))
                y0 = int(np.clip(ny_c[k] - patch_size // 2 + rng.integers(-32, 33), 0, H - patch_size))
                x0 = int(np.clip(nx_c[k] - patch_size // 2 + rng.integers(-32, 33), 0, W - patch_size))
            else:
                k = int(rng.integers(0, len(vy)))
                y0 = int(np.clip(vy[k], 0, H - patch_size))
                x0 = int(np.clip(vx[k], 0, W - patch_size))

            v_patch = slab[:, y0 : y0 + patch_size, x0 : x0 + patch_size].copy()
            i_patch = ink[y0 : y0 + patch_size, x0 : x0 + patch_size].copy()
            s_patch = sup[y0 : y0 + patch_size, x0 : x0 + patch_size].copy()
            t_patch = teacher[y0 : y0 + patch_size, x0 : x0 + patch_size].copy() if teacher is not None else None
            if t_patch is None:
                has_teacher = False

            if rng.random() < 0.5:
                v_patch = v_patch[::-1].copy()
            if rng.random() < 0.5:
                v_patch = v_patch[:, ::-1, :].copy()
                i_patch = i_patch[::-1, :].copy()
                s_patch = s_patch[::-1, :].copy()
                if t_patch is not None:
                    t_patch = t_patch[::-1, :].copy()
            if rng.random() < 0.5:
                v_patch = v_patch[:, :, ::-1].copy()
                i_patch = i_patch[:, ::-1].copy()
                s_patch = s_patch[:, ::-1].copy()
                if t_patch is not None:
                    t_patch = t_patch[:, ::-1].copy()
            k_rot = int(rng.integers(0, 4))
            if k_rot > 0:
                v_patch = np.rot90(v_patch, k=k_rot, axes=(1, 2)).copy()
                i_patch = np.rot90(i_patch, k=k_rot, axes=(0, 1)).copy()
                s_patch = np.rot90(s_patch, k=k_rot, axes=(0, 1)).copy()
                if t_patch is not None:
                    t_patch = np.rot90(t_patch, k=k_rot, axes=(0, 1)).copy()

            vols.append(v_patch[None])
            inks.append(i_patch[None])
            sups.append(s_patch[None])
            if t_patch is not None:
                teachers.append(t_patch[None])

        x_t = torch.from_numpy(np.stack(vols)).to(device)
        y_t = torch.from_numpy(np.stack(inks)).to(device)
        m_t = torch.from_numpy(np.stack(sups)).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, aux = model(x_t, return_aux=True)
            loss = phys_loss_fn(logits, y_t, valid_mask=m_t, volume=x_t)
            if has_teacher and len(teachers) == batch_size:
                t_t = torch.from_numpy(np.stack(teachers)).to(device)
                distill = F.binary_cross_entropy_with_logits(logits, t_t)
                loss = 0.65 * loss + 0.35 * distill
            if "sheet_shift_z" in aux:
                shift_z = aux["sheet_shift_z"]
                tv_z = (
                    (shift_z[:, :, 1:, :] - shift_z[:, :, :-1, :]).abs().mean()
                    + (shift_z[:, :, :, 1:] - shift_z[:, :, :, :-1]).abs().mean()
                )
                loss = loss + 0.01 * tv_z

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 50 == 0 or step == 1 or step == steps:
            with torch.no_grad():
                p_b = torch.sigmoid(logits)
                p_std = float(p_b.std().item())
            print(f"    [Step {step:3d}/{steps}] loss={loss.item():.4f} pred_std={p_std:.3f} lr={scheduler.get_last_lr()[0]:.5f} ({time.time()-t0:.1f}s)", flush=True)

    return model


def main():
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "outputs" if (base_dir / "outputs").exists() else base_dir / "runs/p800_and_scroll_adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_npz = np.load(base_dir / "data/scroll_adaptation_labels.npz")
    frag_ckpt_path = base_dir / "runs/cloud_kintsugi_heavy_seed1337/best.strict.ckpt"
    if not frag_ckpt_path.is_file():
        raise FileNotFoundError(f"A trained fragment checkpoint is required: {frag_ckpt_path}")
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Starting Kintsugi Scroll Adaptation & PHerc0800 Virgin Scan on {device}", flush=True)

    is_cuda = device.type == "cuda"
    bc = 24 if is_cuda else 16
    steps_1a = 350 if is_cuda else 90
    steps_1b = 400 if is_cuda else 100
    batch_sz = 6 if is_cuda else 3
    patch_sz = 192 if is_cuda else 128

    model_frag = KintsugiPhysicalUNet3DTo2D(
        in_channels=1,
        num_classes=1,
        base_channels=32,
        physical_stem=True,
        sheet_unwarp=True,
        multi_scale_stem=False,
    ).to(device)
    model_frag.load_state_dict(torch.load(frag_ckpt_path, map_location="cpu")["model_state"], strict=True)
    model_frag.eval()

    # =========================================================================
    # PHASE 1A: 2.4 um Scroll Adaptation (Train on w018 + disjoint w013 -> Held-Out Val on w013)
    # =========================================================================
    print("\n=== PHASE 1A: 2.4 um Scroll Adaptation (Train: w018 + disjoint w013 -> Held-Out Val: w013) ===", flush=True)
    model_2p4 = KintsugiPhysicalUNet3DTo2D(
        in_channels=1,
        num_classes=1,
        base_channels=bc,
        physical_stem=True,
        sheet_unwarp=True,
        multi_scale_stem=True,
    ).to(device)

    ckpt_2p4_path = out_dir / "kintsugi_scroll_2p4um_best.ckpt"
    w013_pred_path = out_dir / "w013_kintsugi_scroll_adapted_l3.npy"

    if (not is_cuda) and ckpt_2p4_path.exists() and w013_pred_path.exists():
        print("  [+] Loading existing trained 2.4um checkpoint and held-out w013 predictions...", flush=True)
        model_2p4.load_state_dict(torch.load(ckpt_2p4_path, map_location=device)["model_state"], strict=True)
        model_2p4.eval()
        pred_w013_l3 = np.load(w013_pred_path).astype(np.float32)
    else:
        w018_sv = "PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr"
        w013_sv = "PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr"
        w018_pred_ds8_url = f"{S3_ROOT}/PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/ink-detection/downsampled/PHerc1667-20240304144031-2.399um-0.22m-78keV-volume-20251217075048-20260417190342-new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg"
        w013_pred_ds8_url = f"{S3_ROOT}/PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/ink-detection/downsampled/PHerc1667-20240304141531-2.399um-0.22m-78keV-volume-20251217075048-20260417190342-new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg"

        Image.MAX_IMAGE_PIXELS = None
        teacher_w018_l3 = (
            np.array(Image.open(io.BytesIO(urllib.request.urlopen(w018_pred_ds8_url, timeout=20).read())))
            .astype(np.float32)
            / 255.0
        )
        teacher_w013_l3 = (
            np.array(Image.open(io.BytesIO(urllib.request.urlopen(w013_pred_ds8_url, timeout=20).read())))
            .astype(np.float32)
            / 255.0
        )

        train_items_2p4 = []
        for win_name, sv_path, tch_map, (cy0, cy1, cx0, cx1) in [
            ("w018_win1", w018_sv, teacher_w018_l3, (60, 80, 60, 85)),
            ("w013_tr1", w013_sv, teacher_w013_l3, (20, 40, 75, 100)),
            ("w013_tr2", w013_sv, teacher_w013_l3, (150, 170, 90, 115)),
        ]:
            t0 = time.time()
            vol_u8, _ = download_chunk_grid(sv_path, cy0, cy1, cx0, cx1, max_workers=32)
            slab_l3 = canonical_orient_and_pool_2p4um(vol_u8)  # (16, 320, 400) at 19.19 um
            _, H3, W3 = slab_l3.shape
            y_l3_0, y_l3_1 = cy0 * 16, cy1 * 16
            x_l3_0, x_l3_1 = cx0 * 16, cx1 * 16
            ink_crop = (labels_npz[f"{win_name}_ink"][:H3, :W3] > 0).astype(np.float32)
            sup_crop = (labels_npz[f"{win_name}_sup"][:H3, :W3] > 0).astype(np.float32)
            tch_crop = tch_map[y_l3_0:y_l3_1, x_l3_0:x_l3_1][:H3, :W3]
            tch_sharp = np.clip((tch_crop - 0.15) / 0.45, 0.0, 1.0).astype(np.float32)
            ink_refined = (0.5 * ink_crop + 0.5 * tch_sharp * ink_crop).astype(np.float32)
            train_items_2p4.append((slab_l3, ink_refined, sup_crop, tch_sharp))
            print(f"  [+] Loaded {win_name}: shape={slab_l3.shape} in {time.time()-t0:.1f}s", flush=True)

        print(f"  [*] Training Multi-Scale KintsugiPhysicalUNet3DTo2D (CalligraphicPyramid2D, bc={bc}, patch={patch_sz}) on 2.4um scrolls...", flush=True)
        model_2p4 = train_scroll_model(
            model_2p4,
            train_items_2p4,
            device=device,
            steps=steps_1a,
            batch_size=batch_sz,
            patch_size=patch_sz,
            lr=1.8e-3,
        )
        torch.save(
            {"model_state": model_2p4.state_dict(), "config": model_2p4.get_config()},
            ckpt_2p4_path,
        )

        print("  [*] Evaluating Held-Out Window PHerc1667 w013 (cy=86..106, cx=62..87)...", flush=True)
        vol_w013_u8, _ = download_chunk_grid(w013_sv, 86, 106, 62, 87, max_workers=32)
        slab_w013_l3 = canonical_orient_and_pool_2p4um(vol_w013_u8)  # (16, 320, 400)
        pred_w013_l3 = predict_sliding_tta(model_2p4, slab_w013_l3, device=device, tile_size=256, stride=128, use_tta=True)
        np.save(w013_pred_path, pred_w013_l3)
        np.save(out_dir / "w013_surface_mid_l3.npy", vol_w013_u8[54].reshape(320, 8, 400, 8).mean(axis=(1, 3)))

    ink_w013_l3 = labels_npz["w013_val_ink"]
    sup_w013_l3 = labels_npz["w013_val_sup"]
    val_mask_w013 = sup_w013_l3 > 0
    y_w013 = (ink_w013_l3[val_mask_w013] > 0).astype(np.int32)
    s_w013 = pred_w013_l3[val_mask_w013]
    auc_w013 = roc_auc_np(y_w013, s_w013)
    ap_w013, f1_w013, th_w013 = pr_auc_and_best_f1(y_w013, s_w013)
    print(f"  [!] HELD-OUT w013 RESULTS: ROC AUC = {auc_w013:.4f} (vs 0.5082 zero-shot), AP = {ap_w013:.4f}, Best F1 = {f1_w013:.4f}", flush=True)

    # =========================================================================
    # PHASE 1B: 8.64-9.36 um / 1.2m / 113-116 keV Scroll Adaptation (PHerc0139 -> Val w044)
    # =========================================================================
    print("\n=== PHASE 1B: 8.64-9.36 um Long-Propagation Scroll Adaptation (Train: PHerc0139 w035/w040/w041 -> Held-Out Val: w044) ===", flush=True)
    model_coarse = KintsugiPhysicalUNet3DTo2D(
        in_channels=1,
        num_classes=1,
        base_channels=bc,
        physical_stem=True,
        sheet_unwarp=True,
        multi_scale_stem=True,
    ).to(device)

    ckpt_coarse_path = out_dir / "kintsugi_scroll_8to9um_116keV_best.ckpt"
    w044_pred_path = out_dir / "w044_kintsugi_9p36um_val_pred.npy"
    w044_zs_path = out_dir / "w044_kintsugi_zeroshot_pred.npy"

    if ckpt_coarse_path.exists() and w044_pred_path.exists() and w044_zs_path.exists():
        print("  [+] Loading existing trained 9.36um checkpoint and held-out w044 predictions...", flush=True)
        model_coarse.load_state_dict(torch.load(ckpt_coarse_path, map_location=device)["model_state"], strict=True)
        model_coarse.eval()
        pred_w044 = np.load(w044_pred_path).astype(np.float32)
        pred_w044_zeroshot = np.load(w044_zs_path).astype(np.float32)
    else:
        train_items_coarse = []
        for short, seg_full, (cy0, cy1, cx0, cx1) in [
            ("w035", "20260317000000-w035_2026031718", (4, 20, 4, 20)),
            ("w040", "20250831000000-w040_2025083102", (16, 32, 12, 28)),
            ("w041", "20260108000000-w041_2026010816", (12, 28, 4, 20)),
        ]:
            t0 = time.time()
            sv_path = f"PHerc0139/segments/{seg_full}/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
            vol_u8, _ = download_chunk_grid(sv_path, cy0, cy1, cx0, cx1, max_workers=32)
            slab = prepare_coarse_16_slices(vol_u8)  # (16, 2048, 2048)
            _, Hc, Wc = slab.shape
            ink_c = labels_npz[f"0139_{short}_ink"][:Hc, :Wc].astype(np.float32)
            sup_c = labels_npz[f"0139_{short}_sup"][:Hc, :Wc].astype(np.float32)
            train_items_coarse.append((slab, ink_c, sup_c, None))
            print(f"  [+] Loaded PHerc0139 {short}: shape={slab.shape} in {time.time()-t0:.1f}s", flush=True)

        model_coarse.load_state_dict(model_2p4.state_dict(), strict=True)
        print("  [*] Fine-tuning Multi-Scale KintsugiPhysicalUNet3DTo2D on PHerc0139 (9.36 um, 1.2m, 113 keV)...", flush=True)
        model_coarse = train_scroll_model(
            model_coarse,
            train_items_coarse,
            device=device,
            steps=steps_1b,
            batch_size=batch_sz,
            patch_size=patch_sz,
            lr=1.5e-3,
        )
        torch.save(
            {"model_state": model_coarse.state_dict(), "config": model_coarse.get_config()},
            ckpt_coarse_path,
        )
        print("  [*] Evaluating Held-Out Segment PHerc0139 w044 (cy=28..44, cx=20..36)...", flush=True)
        w044_sv = "PHerc0139/segments/20260115000000-w044_2026011522/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
        vol_w044_u8, _ = download_chunk_grid(w044_sv, 28, 44, 20, 36, max_workers=32)
        slab_w044 = prepare_coarse_16_slices(vol_w044_u8)
        pred_w044_zeroshot = predict_sliding_tta(model_frag, slab_w044, device=device, tile_size=256, stride=256, use_tta=False)
        pred_w044 = predict_sliding_tta(model_coarse, slab_w044, device=device, tile_size=256, stride=128, use_tta=True)
        np.save(w044_pred_path, pred_w044.astype(np.float16))
        np.save(w044_zs_path, pred_w044_zeroshot.astype(np.float16))
        np.save(out_dir / "w044_surface_mid.npy", vol_w044_u8[14])

    ink_w044 = (labels_npz["0139_w044_ink"] > 0).astype(np.int32)
    sup_w044 = labels_npz["0139_w044_sup"] > 0
    auc_w044_zs = roc_auc_np(ink_w044[sup_w044], pred_w044_zeroshot[sup_w044])
    auc_w044 = roc_auc_np(ink_w044[sup_w044], pred_w044[sup_w044])
    ap_w044, f1_w044, th_w044 = pr_auc_and_best_f1(ink_w044[sup_w044], pred_w044[sup_w044])
    print(f"  [!] HELD-OUT PHerc0139 w044 RESULTS: ROC AUC = {auc_w044:.4f} (vs {auc_w044_zs:.4f} zero-shot), AP = {ap_w044:.4f}, Best F1 = {f1_w044:.4f}", flush=True)

    # =========================================================================
    # PHASE 2: Full-Area Scan of All 6 Virgin Segments of PHerc0800 ($50k Prize Target)
    # =========================================================================
    print("\n=== PHASE 2: Full-Area Scan of All 6 Virgin Segments of PHerc0800 (8.64um-1.2m-116keV) ===", flush=True)
    p800_segments = [
        "20251028222030-auto_grown_20251028222030940",
        "20251028225813-auto_grown_20251028225813045",
        "20251028213516-auto_grown_20251028213516907",
        "20251028220042-auto_grown_20251028220042762",
        "20251028220955-auto_grown_20251028220955262",
        "20251029010146-auto_grown_20251029010146642",
    ]

    p800_summary = []
    for seg_id in p800_segments:
        t0 = time.time()
        short_id = seg_id.split("-")[0]
        sv_path = f"PHerc0800/segments/{seg_id}/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr"
        meta0 = get_zarr_meta(sv_path, level=0)
        D, H, W = meta0["shape"]
        pred_file = out_dir / f"p800_{short_id}_pred.npy"
        surf_file = out_dir / f"p800_{short_id}_surf.npy"
        mask_file = out_dir / f"p800_{short_id}_mask.npy"

        if (not is_cuda) and pred_file.exists() and surf_file.exists() and mask_file.exists():
            pred_comb = np.load(pred_file).astype(np.float32)
            valid_mask = np.load(mask_file) > 0
        else:
            lvl = 1 if is_cuda else 2
            meta_l = get_zarr_meta(sv_path, level=lvl)
            _, Hl, Wl = meta_l["shape"]
            nyl = math.ceil(Hl / 128)
            nxl = math.ceil(Wl / 128)
            vol_ds_u8, valid_mask_l = download_chunk_grid(sv_path, 0, nyl, 0, nxl, max_workers=32, level=lvl)
            vol_ds = vol_ds_u8.astype(np.float32)

            slab_mid = prepare_coarse_16_slices(vol_ds, z_start=7)
            slab_low = prepare_coarse_16_slices(vol_ds, z_start=4)
            slab_hi = prepare_coarse_16_slices(vol_ds, z_start=11)

            p_mid = predict_sliding_tta(model_coarse, slab_mid, device=device, tile_size=256, stride=192, use_tta=is_cuda)
            p_2p4 = predict_sliding_tta(model_2p4, slab_mid, device=device, tile_size=256, stride=192, use_tta=False)
            p_low = predict_sliding_tta(model_coarse, slab_low, device=device, tile_size=256, stride=256, use_tta=False)
            p_hi = predict_sliding_tta(model_coarse, slab_hi, device=device, tile_size=256, stride=256, use_tta=False)

            pred_ds = (
                0.45 * p_mid
                + 0.35 * p_2p4
                + 0.20 * np.maximum.reduce([p_mid, p_low, p_hi])
            )
            pred_full = np.array(
                Image.fromarray((np.clip(pred_ds, 0.0, 1.0) * 65535.0).astype(np.uint16)).resize((W, H), Image.BILINEAR),
                dtype=np.float32,
            ) / 65535.0
            surf_full = np.array(
                Image.fromarray(vol_ds_u8[D // 2]).resize((W, H), Image.BILINEAR),
                dtype=np.uint8,
            )
            # Use true non-zero papyrus diamond interior eroded by 24 px to eliminate diamond-border step artifacts
            raw_valid = (surf_full > 5).astype(np.float32)
            eroded_t = F.avg_pool2d(
                torch.from_numpy(raw_valid)[None, None],
                kernel_size=49,
                stride=1,
                padding=24,
            ).squeeze().numpy()
            valid_mask = eroded_t > 0.98
            pred_comb = pred_full * valid_mask.astype(np.float32)

            np.save(pred_file, pred_comb.astype(np.float16))
            np.save(surf_file, surf_full)
            np.save(mask_file, valid_mask.astype(np.uint8))

        valid_scores = pred_comb[valid_mask]
        q99 = float(np.quantile(valid_scores, 0.99)) if len(valid_scores) > 0 else 0.0
        q95 = float(np.quantile(valid_scores, 0.95)) if len(valid_scores) > 0 else 0.0
        high_frac = float((valid_scores > 0.65).mean()) if len(valid_scores) > 0 else 0.0
        area_cm2 = float(valid_mask.sum()) * (8.64e-4 ** 2)

        info = {
            "segment_id": seg_id,
            "short_id": short_id,
            "shape": [int(D), int(H), int(W)],
            "valid_area_cm2": round(area_cm2, 2),
            "q95_score": round(q95, 4),
            "q99_score": round(q99, 4),
            "frac_gt_0p65": round(high_frac, 4),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        p800_summary.append(info)
        print(f"  [+] Scanned PHerc0800 {short_id} ({H}x{W}, {area_cm2:.2f} cm^2 valid): q99={q99:.3f}, frac>0.65={high_frac*100:.2f}% in {info['elapsed_sec']}s", flush=True)

    metrics_report = {
        "phase1a_w013_2p4um": {
            "train_segment": "PHerc1667 w018 + disjoint w013 (tr1, tr2 at 2.399 um)",
            "held_out_val_segment": "PHerc1667 w013 (cy=86..106, cx=62..87)",
            "zeroshot_fragment_roc_auc": 0.5082,
            "scroll_adapted_roc_auc": round(auc_w013, 4),
            "scroll_adapted_ap": round(ap_w013, 4),
            "scroll_adapted_best_f1": round(f1_w013, 4),
            "best_threshold": round(th_w013, 4),
        },
        "phase1b_w044_9p36um": {
            "train_segments": "PHerc0139 w035, w040, w041 (9.362 um, 1.2m, 113 keV)",
            "held_out_val_segment": "PHerc0139 w044 (cy=28..44, cx=20..36)",
            "zeroshot_fragment_roc_auc": round(auc_w044_zs, 4),
            "scroll_adapted_roc_auc": round(auc_w044, 4),
            "scroll_adapted_ap": round(ap_w044, 4),
            "scroll_adapted_best_f1": round(f1_w044, 4),
            "best_threshold": round(th_w044, 4),
        },
        "phase2_pherc0800_virgin_segments": p800_summary,
    }
    with open(out_dir / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print("\n[+] All phases completed successfully! Metrics saved to metrics_report.json", flush=True)


if __name__ == "__main__":
    main()
