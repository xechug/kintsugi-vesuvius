import os
import multiprocessing as mp
import math
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr
from zarr.core.group import Group
from torch.utils.data import DataLoader, Dataset

try:
    from .data_contract import DataAuditReport, audit_training_data, open_raster
except ImportError:
    from data_contract import DataAuditReport, audit_training_data, open_raster


class TiffStack:
    def __init__(self, path):
        root = Path(path)
        layer_dir = root / "layers" if (root / "layers").is_dir() else root
        self.files = sorted(layer_dir.glob("*.tif")) + sorted(layer_dir.glob("*.tiff"))
        if not self.files:
            raise FileNotFoundError(f"No TIFF layers found in {layer_dir}")

        first = tifffile.imread(self.files[0])
        if first.ndim != 2:
            raise ValueError(f"Expected 2D TIFF layer, got shape {first.shape}")
        self.shape = (len(self.files), int(first.shape[0]), int(first.shape[1]))
        self.dtype = first.dtype
        self.chunks = (1, self.shape[1], self.shape[2])

    def __getitem__(self, item):
        z_sel, y_sel, x_sel = item
        if isinstance(z_sel, slice):
            indices = range(*z_sel.indices(self.shape[0]))
        else:
            indices = [int(z_sel)]

        layers = [tifffile.imread(self.files[i])[y_sel, x_sel] for i in indices]
        stacked = np.stack(layers, axis=0)
        if not isinstance(z_sel, slice):
            return stacked[0]
        return stacked


def open_volume_array(zarr_path):
    path = Path(zarr_path)
    if path.is_dir() and not path.name.endswith(".zarr") and not (path / ".zarray").exists():
        return TiffStack(path)

    store = zarr.open(zarr_path, mode="r")
    if isinstance(store, Group) and "0" in store:
        return store["0"]
    return store


def read_mask(path):
    return open_raster(path)


def apply_fourth_augment_z(
    vol_patch: np.ndarray,
    rng: np.random.Generator,
    max_shift: int = 2,
    edge_crop_prob: float = 0.5,
) -> np.ndarray:
    """
    Opción 2a Física — Alabeo/desplazamiento en Z y recorte de bordes Z (_fourth_augment,
    inspirado en los ganadores del Grand Prize 2023).
    Obliga a SoftSheetUnwarper3D y al cabezal de profundidad a seguir la epidermis
    aunque la lámina de papiro cambie de cota Z.
    """
    depth = vol_patch.shape[0]
    if depth <= 2 or max_shift <= 0:
        return vol_patch

    out = np.zeros_like(vol_patch)
    shift = int(rng.integers(-int(max_shift), int(max_shift) + 1))
    if shift > 0:
        out[shift:] = vol_patch[: depth - shift]
    elif shift < 0:
        s = -shift
        out[: depth - s] = vol_patch[s:]
    else:
        out[:] = vol_patch

    if rng.random() < float(edge_crop_prob) and depth >= 8:
        max_drop = max(1, min(int(max_shift), depth // 6))
        top_drop = int(rng.integers(0, max_drop + 1))
        bot_drop = int(rng.integers(0, max_drop + 1))
        if top_drop > 0:
            out[:top_drop] = 0.0
        if bot_drop > 0:
            out[depth - bot_drop :] = 0.0
    return out


def inject_synthetic_carbonized_ink(
    vol_patch: np.ndarray,
    ink_patch: np.ndarray,
    valid_patch: np.ndarray,
    rng: np.random.Generator,
    strength: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Opción 2b Física — Simulador de Tinta Carbonizada Sintética sobre papiro desnudo.
    Si el parche tiene poca tinta (<5%), sintetiza un trazo caligráfico con:
    1. Costra laminar delgada (~1.5 capas) anclada a la epidermis local z_0(y, x).
    2. Halo de contraste de fase sincrotrón (segunda derivada Gaussiana en Z).
    3. Patrón de craquelado térmico Voronoi (islas poligonales de goma arábiga a 400 °C).
    """
    depth, height, width = vol_patch.shape
    if float((ink_patch > 0.5).mean()) > 0.05 or depth < 4 or height < 16 or width < 16:
        return vol_patch, ink_patch

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    angle = float(rng.uniform(0.0, np.pi))
    c_ang, s_ang = float(np.cos(angle)), float(np.sin(angle))
    u = xx * c_ang + yy * s_ang - float(rng.uniform(-0.25, 0.25))
    v = -xx * s_ang + yy * c_ang
    curvature = float(rng.uniform(-0.35, 0.35))
    half_width = float(rng.uniform(0.12, 0.24))
    dist_stroke = np.abs(u - curvature * (v**2))
    stroke_mask = (
        (dist_stroke < half_width)
        & (np.abs(v) < 0.78)
        & (valid_patch > 0)
    )
    if int(stroke_mask.sum()) < 16:
        return vol_patch, ink_patch

    # Patrón de craquelado Voronoi (islas térmicas separadas por micro-fisuras)
    num_seeds = max(12, (height * width) // 128)
    seeds_y = rng.uniform(-1.0, 1.0, size=(num_seeds, 1, 1)).astype(np.float32)
    seeds_x = rng.uniform(-1.0, 1.0, size=(num_seeds, 1, 1)).astype(np.float32)
    dist2 = (yy[None, :, :] - seeds_y) ** 2 + (xx[None, :, :] - seeds_x) ** 2
    part = np.partition(dist2, 1, axis=0)
    fissure_gap = np.sqrt(np.maximum(part[1], 0.0)) - np.sqrt(np.maximum(part[0], 0.0))
    crackle_islands = np.clip(fissure_gap / 0.035, 0.15, 1.0).astype(np.float32)

    # Detectar cota epidérmica z_0(y, x) donde el escalón vertical ∂z V es máximo
    dz = np.diff(vol_patch, axis=0, prepend=vol_patch[:1])
    z0_idx = np.argmax(dz, axis=0).astype(np.float32)
    z_grid = np.arange(depth, dtype=np.float32)[:, None, None]
    sigma_z = 1.15
    u_z = (z_grid - z0_idx[None, :, :]) / sigma_z
    # Costra epidérmica + halo de fase sincrotrón
    shell_profile = np.exp(-0.5 * (u_z**2)) * (1.0 + 0.35 * (1.0 - u_z**2))

    local_std = float(np.std(vol_patch) + 1e-4)
    delta_v = (
        float(strength)
        * max(local_std, 0.04)
        * shell_profile
        * crackle_islands[None, :, :]
        * stroke_mask[None, :, :].astype(np.float32)
    )
    new_vol = np.clip(vol_patch + delta_v, 0.0, 1.0).astype(np.float32)
    new_ink = np.maximum(ink_patch, stroke_mask.astype(np.float32))
    return new_vol, new_ink


class PapyriDataset(Dataset):
    def __init__(
        self,
        zarr_path,
        mask_path,
        ink_path,
        patch_size=(64, 64, 64),
        stride=(64, 32, 32),
        downsample_factor=1,
        epoch_size=1000,
        positive_fraction=0.33,
        base_seed=1337,
        z_policy="random",
        coord_scale=32,
        ink_threshold=None,
        normalize_divisor="auto",
        label_kind="ground_truth",
        validate_contract=True,
        data_audit=None,
        min_valid_fraction=0.95,
        min_positive_fraction=0.001,
        max_negative_fraction=0.0,
        allowed_regions=None,
        sampling_mode="balanced",
        augment_xy=False,
        intensity_jitter=0.0,
        noise_std=0.0,
        z_jitter=0,
        synthetic_ink_prob=0.0,
        align_to_chunks=False,
        resample_each_epoch=True,
        quiet=False,
    ):
        self.zarr_path = zarr_path
        self.mask_path = mask_path
        self.ink_path = ink_path
        self.patch_size = tuple(int(x) for x in patch_size)
        self.stride = tuple(int(x) for x in stride)
        self.downsample_factor = int(downsample_factor)
        self.epoch_size = int(epoch_size)
        self.positive_fraction = float(positive_fraction)
        self.base_seed = int(base_seed)
        self.z_policy = z_policy
        self.coord_scale = int(coord_scale)
        self.ink_threshold = ink_threshold
        self.normalize_divisor = normalize_divisor
        self.label_kind = str(label_kind)
        self.validate_contract = bool(validate_contract)
        self.data_audit = data_audit
        self.min_valid_fraction = float(min_valid_fraction)
        self.min_positive_fraction = float(min_positive_fraction)
        self.max_negative_fraction = float(max_negative_fraction)
        self.allowed_regions = tuple(allowed_regions or ())
        self.sampling_mode = str(sampling_mode)
        self.augment_xy = bool(augment_xy)
        self.intensity_jitter = float(intensity_jitter)
        self.noise_std = float(noise_std)
        self.z_jitter = int(z_jitter)
        self.synthetic_ink_prob = float(synthetic_ink_prob)
        self.align_to_chunks = bool(align_to_chunks)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.quiet = quiet
        self._shared_epoch = mp.Value("q", 0)

        if not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be in [0, 1]")
        if not 0.0 <= self.max_negative_fraction <= self.min_positive_fraction <= 1.0:
            raise ValueError(
                "coverage fractions must satisfy 0 <= max_negative <= min_positive <= 1"
            )
        if self.sampling_mode not in {"balanced", "uniform", "grid"}:
            raise ValueError("sampling_mode must be 'balanced', 'uniform', or 'grid'")
        if not 0.0 <= self.intensity_jitter <= 1.0 or not 0.0 <= self.noise_std <= 1.0:
            raise ValueError("intensity_jitter and noise_std must be in [0, 1]")
        if self.z_jitter < 0 or not 0.0 <= self.synthetic_ink_prob <= 1.0:
            raise ValueError("z_jitter must be >= 0 and synthetic_ink_prob in [0, 1]")

        self._volume = None
        self._load_metadata_and_masks()
        self._generate_patches()

    @property
    def volume(self):
        if self._volume is None:
            self._volume = open_volume_array(self.zarr_path)
        return self._volume

    def _load_metadata_and_masks(self):
        if self.validate_contract:
            if self.data_audit is None:
                self.data_audit = audit_training_data(
                    self.zarr_path,
                    self.mask_path,
                    self.ink_path,
                    label_kind=self.label_kind,
                )
            if not isinstance(self.data_audit, DataAuditReport):
                raise TypeError("data_audit must be a DataAuditReport")
            self.data_audit.raise_for_training()

        volume = open_volume_array(self.zarr_path)
        self.volume_shape = tuple(int(v) for v in volume.shape[-3:])
        self.volume_chunks = getattr(volume, "chunks", None)
        self.volume_dtype = getattr(volume, "dtype", None)
        if self.normalize_divisor == "auto":
            if self.volume_dtype is not None and np.issubdtype(self.volume_dtype, np.integer):
                self.normalize_divisor = float(np.iinfo(self.volume_dtype).max)
            else:
                self.normalize_divisor = 1.0
        else:
            self.normalize_divisor = float(self.normalize_divisor)

        self.mask = read_mask(self.mask_path)
        self.ink_mask = read_mask(self.ink_path)
        if self.data_audit is not None and self.data_audit.label.is_binary:
            observed_max = max(self.data_audit.label.observed_values, default=1.0)
            self.label_divisor = max(1.0, float(observed_max))
        elif np.issubdtype(self.ink_mask.dtype, np.integer):
            self.label_divisor = float(np.iinfo(self.ink_mask.dtype).max)
        else:
            self.label_divisor = 1.0

        if not self.quiet:
            print(f"Volume shape: {self.volume_shape}")
            print(f"Volume chunks: {self.volume_chunks}")
            print(f"Volume dtype: {self.volume_dtype}")
            print(f"Papyrus mask shape: {self.mask.shape}")
            print(f"Ink mask shape: {self.ink_mask.shape}")

    def _generate_patches(self):
        depth, height, width = self.volume_shape
        d_z, d_y, d_x = self.patch_size
        scale = max(1, self.coord_scale)

        if d_z > depth or d_y > height or d_x > width:
            raise ValueError(
                f"patch_size={self.patch_size} does not fit volume_shape={self.volume_shape}"
            )
        if self.sampling_mode == "grid" and (d_y % scale or d_x % scale):
            raise ValueError(
                "grid sampling requires XY patch dimensions divisible by coord_scale"
            )

        threshold = self.ink_threshold
        if threshold is None:
            threshold = self.label_divisor * 0.5

        valid_fraction = self._coarse_fraction(self.mask, scale, lambda values: values > 0)
        ink_fraction = self._coarse_fraction(
            self.ink_mask, scale, lambda values: values > float(threshold)
        )
        window_y = max(1, math.ceil(d_y / scale))
        window_x = max(1, math.ceil(d_x / scale))
        valid_coverage = self._window_mean(valid_fraction, window_y, window_x)
        ink_coverage = self._window_mean(ink_fraction, window_y, window_x)

        coarse_y, coarse_x = np.indices(valid_coverage.shape, dtype=np.int64)
        origin_y = coarse_y * scale
        origin_x = coarse_x * scale
        fits = (origin_y <= height - d_y) & (origin_x <= width - d_x)
        fits &= self._allowed_origin_mask(origin_y, origin_x, d_y, d_x)
        if self.sampling_mode == "grid":
            fits &= self._evaluation_grid_mask(origin_y, origin_x, d_y, d_x)
        if self.align_to_chunks:
            if self.volume_chunks is None or len(self.volume_chunks) < 2:
                raise ValueError("chunk alignment requires volume chunk metadata")
            chunk_y, chunk_x = (int(value) for value in self.volume_chunks[-2:])
            fits &= (origin_y % chunk_y == 0) & (origin_x % chunk_x == 0)
        valid = valid_coverage >= self.min_valid_fraction
        positive = fits & valid & (ink_coverage >= self.min_positive_fraction)
        negative = fits & valid & (ink_coverage <= self.max_negative_fraction)

        self.valid_coords = np.column_stack(
            (origin_y[fits & valid], origin_x[fits & valid])
        ).astype(np.int64)
        uniform_rng = np.random.default_rng(
            np.random.SeedSequence([self.base_seed, 0x554E4946])
        )
        self._uniform_order = uniform_rng.permutation(len(self.valid_coords))

        self.ink_coords = np.column_stack((origin_y[positive], origin_x[positive])).astype(
            np.int64
        )
        self.empty_coords = np.column_stack((origin_y[negative], origin_x[negative])).astype(
            np.int64
        )

        if len(self.valid_coords) == 0:
            raise ValueError("No valid coordinates found inside the papyrus mask")
        if self.sampling_mode == "grid":
            self.epoch_size = min(self.epoch_size, len(self.valid_coords))
        if (
            self.sampling_mode == "balanced"
            and len(self.ink_coords) == 0
            and len(self.empty_coords) == 0
        ):
            raise ValueError("No valid training coordinates found inside the papyrus mask")

        if not self.quiet:
            print(f"Ink coordinates: {len(self.ink_coords)}")
            print(f"Empty coordinates: {len(self.empty_coords)}")

    @staticmethod
    def _filter_valid_xy(coords, height, width, d_y, d_x):
        if len(coords) == 0:
            return coords.astype(np.int64)
        valid = (coords[:, 0] <= height - d_y) & (coords[:, 1] <= width - d_x)
        return coords[valid].astype(np.int64)

    @staticmethod
    def _window_mean(array, window_y, window_x):
        if window_y > array.shape[0] or window_x > array.shape[1]:
            return np.empty((0, 0), dtype=np.float32)
        integral = np.pad(
            np.cumsum(np.cumsum(array, axis=0, dtype=np.float64), axis=1, dtype=np.float64),
            ((1, 0), (1, 0)),
        )
        sums = (
            integral[window_y:, window_x:]
            - integral[:-window_y, window_x:]
            - integral[window_y:, :-window_x]
            + integral[:-window_y, :-window_x]
        )
        return (sums / float(window_y * window_x)).astype(np.float32)

    @staticmethod
    def _coarse_fraction(raster, scale, predicate):
        height, width = raster.shape
        out_height = math.ceil(height / scale)
        out_width = math.ceil(width / scale)
        output = np.zeros((out_height, out_width), dtype=np.float32)
        block_rows = max(scale, math.ceil(raster.chunks[0] / scale) * scale)

        for y0 in range(0, height, block_rows):
            y1 = min(y0 + block_rows, height)
            values = np.asarray(raster[y0:y1, :])
            binary = np.asarray(predicate(values), dtype=np.float32)
            padded_height = math.ceil(binary.shape[0] / scale) * scale
            padded_width = out_width * scale
            padded = np.zeros((padded_height, padded_width), dtype=np.float32)
            counts = np.zeros_like(padded)
            padded[: binary.shape[0], : binary.shape[1]] = binary
            counts[: binary.shape[0], : binary.shape[1]] = 1.0
            sums = padded.reshape(-1, scale, out_width, scale).sum(axis=(1, 3))
            denominators = counts.reshape(-1, scale, out_width, scale).sum(axis=(1, 3))
            start = y0 // scale
            rows = sums.shape[0]
            output[start : start + rows] = sums / np.maximum(denominators, 1.0)
        return output

    def _allowed_origin_mask(self, origin_y, origin_x, patch_height, patch_width):
        if not self.allowed_regions:
            return np.ones(origin_y.shape, dtype=bool)
        allowed = np.zeros(origin_y.shape, dtype=bool)
        for region in self.allowed_regions:
            if hasattr(region, "contains_patch"):
                allowed |= region.contains_patch(origin_y, origin_x, patch_height, patch_width)
            else:
                y0, x0, y1, x1 = (int(value) for value in region)
                allowed |= (
                    (origin_y >= y0)
                    & (origin_x >= x0)
                    & (origin_y + patch_height <= y1)
                    & (origin_x + patch_width <= x1)
                )
        return allowed

    def _evaluation_grid_mask(self, origin_y, origin_x, patch_height, patch_width):
        if not self.allowed_regions:
            return (origin_y % patch_height == 0) & (origin_x % patch_width == 0)

        selected = np.zeros(origin_y.shape, dtype=bool)
        for region in self.allowed_regions:
            if hasattr(region, "contains_patch"):
                y0, x0 = int(region.y0), int(region.x0)
                contained = region.contains_patch(
                    origin_y, origin_x, patch_height, patch_width
                )
            else:
                y0, x0, y1, x1 = (int(value) for value in region)
                contained = (
                    (origin_y >= y0)
                    & (origin_x >= x0)
                    & (origin_y + patch_height <= y1)
                    & (origin_x + patch_width <= x1)
                )
            selected |= (
                contained
                & ((origin_y - y0) % patch_height == 0)
                & ((origin_x - x0) % patch_width == 0)
            )
        return selected

    def __len__(self):
        return self.epoch_size

    @property
    def epoch(self):
        return int(self._shared_epoch.value)

    def set_epoch(self, epoch):
        if self.resample_each_epoch:
            with self._shared_epoch.get_lock():
                self._shared_epoch.value = int(epoch)

    def _rng_for_idx(self, idx):
        epoch = self.epoch if self.resample_each_epoch else 0
        return np.random.default_rng(np.random.SeedSequence([self.base_seed, epoch, int(idx)]))

    def _choose_xy(self, rng, index):
        if self.sampling_mode == "grid":
            ordered_index = self._uniform_order[int(index)]
            return self.valid_coords[int(ordered_index)]
        if self.sampling_mode == "uniform":
            if not self.resample_each_epoch:
                ordered_index = self._uniform_order[int(index) % len(self._uniform_order)]
                return self.valid_coords[int(ordered_index)]
            return self.valid_coords[int(rng.integers(0, len(self.valid_coords)))]

        wants_positive = rng.random() < self.positive_fraction
        if wants_positive and len(self.ink_coords) > 0:
            coords = self.ink_coords
        elif len(self.empty_coords) > 0:
            coords = self.empty_coords
        else:
            coords = self.ink_coords

        return coords[int(rng.integers(0, len(coords)))]

    def _choose_z(self, rng):
        depth = self.volume_shape[0]
        d_z = self.patch_size[0]
        max_start = depth - d_z
        if max_start <= 0:
            return 0
        if self.z_policy == "center":
            return max(0, (depth // 2) - (d_z // 2))
        if self.z_policy == "random":
            return int(rng.integers(0, max_start + 1))
        raise ValueError(f"Unknown z_policy: {self.z_policy}")

    def __getitem__(self, idx):
        rng = self._rng_for_idx(idx)
        y, x = self._choose_xy(rng, idx)
        z = self._choose_z(rng)
        d_z, d_y, d_x = self.patch_size

        vol_patch = np.asarray(self.volume[z : z + d_z, y : y + d_y, x : x + d_x])
        ink_patch = np.asarray(self.ink_mask[y : y + d_y, x : x + d_x])
        valid_patch = np.asarray(self.mask[y : y + d_y, x : x + d_x])

        if self.augment_xy:
            if d_y == d_x:
                rotations = int(rng.integers(0, 4))
            else:
                rotations = int(rng.integers(0, 2)) * 2
            vol_patch = np.rot90(vol_patch, rotations, axes=(-2, -1))
            ink_patch = np.rot90(ink_patch, rotations, axes=(-2, -1))
            valid_patch = np.rot90(valid_patch, rotations, axes=(-2, -1))
            if rng.random() < 0.5:
                vol_patch = np.flip(vol_patch, axis=-2)
                ink_patch = np.flip(ink_patch, axis=-2)
                valid_patch = np.flip(valid_patch, axis=-2)
            if rng.random() < 0.5:
                vol_patch = np.flip(vol_patch, axis=-1)
                ink_patch = np.flip(ink_patch, axis=-1)
                valid_patch = np.flip(valid_patch, axis=-1)

        if self.downsample_factor > 1:
            df = self.downsample_factor
            vol_patch = vol_patch[:, ::df, ::df]
            ink_patch = ink_patch[::df, ::df]
            valid_patch = valid_patch[::df, ::df]

        vol_patch = vol_patch.astype(np.float32) / self.normalize_divisor
        ink_patch = np.clip(ink_patch.astype(np.float32) / self.label_divisor, 0.0, 1.0)
        if self.synthetic_ink_prob > 0.0 and rng.random() < self.synthetic_ink_prob:
            vol_patch, ink_patch = inject_synthetic_carbonized_ink(
                vol_patch, ink_patch, valid_patch, rng
            )
        if self.z_jitter > 0:
            vol_patch = apply_fourth_augment_z(
                vol_patch, rng, max_shift=self.z_jitter
            )
        if self.intensity_jitter > 0.0:
            gain = rng.uniform(1.0 - self.intensity_jitter, 1.0 + self.intensity_jitter)
            offset = rng.uniform(-self.intensity_jitter / 2.0, self.intensity_jitter / 2.0)
            vol_patch = vol_patch * gain + offset
        if self.noise_std > 0.0:
            vol_patch = vol_patch + rng.normal(0.0, self.noise_std, size=vol_patch.shape)
        vol_patch = np.ascontiguousarray(np.clip(vol_patch, 0.0, 1.0), dtype=np.float32)
        ink_patch = np.ascontiguousarray(ink_patch, dtype=np.float32)

        vol_tensor = torch.from_numpy(vol_patch).unsqueeze(0)
        target_tensor = torch.from_numpy(ink_patch).unsqueeze(0)
        valid_tensor = torch.from_numpy(np.ascontiguousarray(valid_patch > 0)).unsqueeze(0)
        coords_tensor = torch.tensor((z, int(y), int(x)), dtype=torch.int64)

        return {
            "image": vol_tensor,
            "target": target_tensor,
            "valid_mask": valid_tensor,
            "coords": coords_tensor,
        }


class BalancedMultiSourceDataset(Dataset):
    """Round-robin wrapper that gives each physical source equal representation."""

    def __init__(self, datasets, *, epoch_size):
        self.datasets = tuple(datasets)
        self.epoch_size = int(epoch_size)
        if not self.datasets:
            raise ValueError("at least one source dataset is required")
        if self.epoch_size <= 0:
            raise ValueError("epoch_size must be positive")

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        source_index = int(index) % len(self.datasets)
        local_index = int(index) // len(self.datasets)
        sample = dict(self.datasets[source_index][local_index])
        sample["source_index"] = torch.tensor(source_index, dtype=torch.int64)
        return sample

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)


def make_dataloader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=None,
    pin_memory=None,
    prefetch_factor=2,
    persistent_workers=None,
    drop_last=False,
):
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers

    return DataLoader(dataset, **kwargs)
