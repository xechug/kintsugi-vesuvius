from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile
import zarr
from PIL import Image
from zarr.core.group import Group


class DataContractError(ValueError):
    pass


def sha256_file(path: str | os.PathLike[str], block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


class LazyRaster:
    """A picklable 2D raster that opens pixel storage only on first indexing."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(Path(path))
        source = Path(self.path)
        if not source.is_file():
            raise FileNotFoundError(source)

        suffix = source.suffix.lower()
        self._backend = "tiff" if suffix in {".tif", ".tiff"} else "image"
        if self._backend == "tiff":
            with tifffile.TiffFile(source) as tif:
                if len(tif.pages) != 1:
                    raise DataContractError(f"Expected one 2D page in {source}")
                page = tif.pages[0]
                self._shape = tuple(int(value) for value in page.shape)
                self._dtype = np.dtype(page.dtype)
                tile_y = int(page.tilelength or page.rowsperstrip or self._shape[0])
                tile_x = int(page.tilewidth or self._shape[1])
                self._chunks = (min(tile_y, self._shape[0]), min(tile_x, self._shape[1]))
        else:
            with Image.open(source) as image:
                width, height = image.size
                self._shape = (int(height), int(width))
                probe = np.asarray(image)
                self._dtype = probe.dtype
                self._chunks = self._shape

        if len(self._shape) != 2:
            raise DataContractError(f"Expected a 2D raster, got {self._shape} from {source}")

        self._array = None
        self._store = None
        self._pid = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunks(self) -> tuple[int, int]:
        return self._chunks

    @property
    def is_open(self) -> bool:
        return self._array is not None and self._pid == os.getpid()

    def _ensure_open(self):
        if self.is_open:
            return self._array
        self.close()
        if self._backend == "tiff":
            self._store = tifffile.imread(self.path, aszarr=True)
            self._array = zarr.open(self._store, mode="r")
        else:
            with Image.open(self.path) as image:
                array = np.asarray(image)
            if array.ndim == 3:
                array = array[..., 0]
            self._array = array
        self._pid = os.getpid()
        return self._array

    def __getitem__(self, item):
        return self._ensure_open()[item]

    def iter_chunks(self) -> Iterator[np.ndarray]:
        chunk_y, chunk_x = self.chunks
        height, width = self.shape
        for y in range(0, height, chunk_y):
            for x in range(0, width, chunk_x):
                yield np.asarray(self[y : min(y + chunk_y, height), x : min(x + chunk_x, width)])

    def close(self) -> None:
        store = self._store
        self._array = None
        self._store = None
        self._pid = None
        if store is not None and hasattr(store, "close"):
            store.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_array"] = None
        state["_store"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def open_raster(path: str | os.PathLike[str]) -> LazyRaster:
    return LazyRaster(path)


def _is_binary_raster(raster: LazyRaster) -> tuple[bool, tuple[float, ...]]:
    observed: set[float] = set()
    dtype = raster.dtype
    if np.issubdtype(dtype, np.integer):
        allowed = {0.0, 1.0, float(np.iinfo(dtype).max)}
    else:
        allowed = {0.0, 1.0}

    for chunk in raster.iter_chunks():
        observed.update(float(value) for value in np.unique(chunk))
        if not observed.issubset(allowed):
            return False, tuple(sorted(observed)[:32])
    return True, tuple(sorted(observed))


def _open_volume(path: str | os.PathLike[str]):
    source = Path(path)
    if source.is_dir() and (source / "layers").is_dir():
        files = sorted((source / "layers").glob("*.tif*"))
        if not files:
            raise FileNotFoundError(f"No TIFF layers found in {source / 'layers'}")
        with tifffile.TiffFile(files[0]) as tif:
            page = tif.pages[0]
            return (len(files), *page.shape), np.dtype(page.dtype), (1, *page.shape)

    array = zarr.open(str(source), mode="r")
    if isinstance(array, Group):
        if "0" not in array:
            raise DataContractError(f"Zarr group {source} has no level '0'")
        array = array["0"]
    return tuple(int(v) for v in array.shape), np.dtype(array.dtype), tuple(int(v) for v in array.chunks)


def _volume_metadata_sha256(path: str | os.PathLike[str]) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    if source.is_file():
        return sha256_file(source)

    metadata_names = {".zarray", ".zattrs", ".zgroup", "zarr.json"}
    metadata_files = sorted(
        item for item in source.rglob("*") if item.is_file() and item.name in metadata_names
    )
    if not metadata_files:
        for item in sorted(source.rglob("*.tif*")):
            relative = item.relative_to(source).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(str(item.stat().st_size).encode("ascii"))
        return digest.hexdigest()

    for item in metadata_files:
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        with item.open("rb") as stream:
            digest.update(stream.read())
    return digest.hexdigest()


@dataclass(frozen=True)
class VolumeAudit:
    path: str
    metadata_sha256: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class RasterAudit:
    path: str
    sha256: str
    shape: tuple[int, int]
    dtype: str
    kind: str
    is_binary: bool
    observed_values: tuple[float, ...]


@dataclass(frozen=True)
class DataAuditReport:
    volume: VolumeAudit
    valid_mask: RasterAudit
    label: RasterAudit
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_training(self) -> None:
        if self.errors:
            raise DataContractError("; ".join(self.errors))

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    def save(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return destination


def audit_training_data(
    volume_path: str | os.PathLike[str],
    valid_mask_path: str | os.PathLike[str],
    label_path: str | os.PathLike[str],
    *,
    label_kind: str = "ground_truth",
) -> DataAuditReport:
    if label_kind not in {"ground_truth", "teacher"}:
        raise DataContractError("label_kind must be 'ground_truth' or 'teacher'")

    volume_shape, volume_dtype, volume_chunks = _open_volume(volume_path)
    if len(volume_shape) != 3:
        raise DataContractError(f"Expected ZYX volume, got shape {volume_shape}")

    valid = open_raster(valid_mask_path)
    label = open_raster(label_path)
    valid_binary, valid_values = _is_binary_raster(valid)
    label_binary, label_values = _is_binary_raster(label)
    valid_hash = sha256_file(valid_mask_path)
    label_hash = sha256_file(label_path)

    errors: list[str] = []
    warnings: list[str] = []
    expected_shape = tuple(volume_shape[-2:])
    if valid.shape != expected_shape:
        errors.append(f"valid mask shape {valid.shape} does not match volume YX {expected_shape}")
    if label.shape != expected_shape:
        errors.append(f"label shape {label.shape} does not match volume YX {expected_shape}")
    if valid_hash == label_hash:
        errors.append("valid mask and label files are identical")
    if not valid_binary:
        errors.append("valid mask must be binary")
    if label_kind == "ground_truth" and not label_binary:
        errors.append("ground_truth labels must be binary")
    if label_kind == "teacher":
        warnings.append("teacher scores are allowed for training only, never validation or test")

    valid.close()
    label.close()
    return DataAuditReport(
        volume=VolumeAudit(
            path=str(Path(volume_path)),
            metadata_sha256=_volume_metadata_sha256(volume_path),
            shape=volume_shape,
            chunks=volume_chunks,
            dtype=str(volume_dtype),
        ),
        valid_mask=RasterAudit(
            path=str(Path(valid_mask_path)),
            sha256=valid_hash,
            shape=valid.shape,
            dtype=str(valid.dtype),
            kind="valid_mask",
            is_binary=valid_binary,
            observed_values=valid_values,
        ),
        label=RasterAudit(
            path=str(Path(label_path)),
            sha256=label_hash,
            shape=label.shape,
            dtype=str(label.dtype),
            kind=label_kind,
            is_binary=label_binary,
            observed_values=label_values,
        ),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
