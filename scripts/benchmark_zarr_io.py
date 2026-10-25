from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vesuvius_code.dataset import open_volume_array


def chunk_bytes_for_patch(*, shape, chunks, dtype, origin, patch_size):
    shape = tuple(int(value) for value in shape)
    chunks = tuple(int(value) for value in chunks)
    origin = tuple(int(value) for value in origin)
    patch_size = tuple(int(value) for value in patch_size)
    if not all(
        start >= 0 and size > 0 and start + size <= extent
        for start, size, extent in zip(origin, patch_size, shape, strict=True)
    ):
        raise ValueError("patch lies outside the array")

    axis_indices = []
    for start, size, chunk in zip(origin, patch_size, chunks, strict=True):
        first = start // chunk
        last = (start + size - 1) // chunk
        axis_indices.append(range(first, last + 1))

    itemsize = np.dtype(dtype).itemsize
    storage_bytes = 0
    chunks_touched = 0
    for indices in itertools.product(*axis_indices):
        chunk_shape = [
            min(chunk, extent - index * chunk)
            for index, chunk, extent in zip(indices, chunks, shape, strict=True)
        ]
        storage_bytes += math.prod(chunk_shape) * itemsize
        chunks_touched += 1
    logical_bytes = math.prod(patch_size) * itemsize
    return {
        "chunks_touched": chunks_touched,
        "logical_bytes": logical_bytes,
        "storage_bytes": storage_bytes,
        "amplification": storage_bytes / logical_bytes,
    }


class RandomPatchDataset(Dataset):
    def __init__(self, zarr_path, patch_size, origins):
        self.zarr_path = str(zarr_path)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.origins = np.asarray(origins, dtype=np.int64)
        self._array = None

    @property
    def array(self):
        if self._array is None:
            self._array = open_volume_array(self.zarr_path)
        return self._array

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        z, y, x = (int(value) for value in self.origins[index])
        depth, height, width = self.patch_size
        patch = np.asarray(
            self.array[z : z + depth, y : y + height, x : x + width]
        )
        if np.issubdtype(patch.dtype, np.integer):
            divisor = float(np.iinfo(patch.dtype).max)
        else:
            divisor = 1.0
        patch = np.ascontiguousarray(patch.astype(np.float32) / divisor)
        return torch.from_numpy(patch)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_array"] = None
        return state


def random_origins(shape, patch_size, samples, *, seed=1337, align_to_chunks=None):
    rng = np.random.default_rng(int(seed))
    columns = []
    for extent, patch, chunk in zip(
        shape,
        patch_size,
        align_to_chunks or (None, None, None),
        strict=True,
    ):
        maximum = int(extent) - int(patch)
        if maximum < 0:
            raise ValueError("patch does not fit the volume")
        if chunk is None:
            columns.append(rng.integers(0, maximum + 1, size=int(samples)))
        else:
            choices = np.arange(0, maximum + 1, int(chunk), dtype=np.int64)
            columns.append(rng.choice(choices, size=int(samples), replace=True))
    return np.column_stack(columns)


def main():
    parser = argparse.ArgumentParser(description="Benchmark out-of-core random Zarr patches")
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--patch-depth", type=int, default=16)
    parser.add_argument("--patch-height", type=int, default=128)
    parser.add_argument("--patch-width", type=int, default=128)
    parser.add_argument("--align-to-chunks", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if (
        args.samples <= 0
        or args.warmup_samples < 0
        or args.batch_size <= 0
        or args.workers < 0
    ):
        raise ValueError(
            "samples and batch size must be positive; warmup and workers cannot be negative"
        )

    array = open_volume_array(args.zarr)
    shape = tuple(int(value) for value in array.shape)
    chunks = tuple(int(value) for value in array.chunks)
    patch_size = (args.patch_depth, args.patch_height, args.patch_width)
    warmup_samples = math.ceil(args.warmup_samples / args.batch_size) * args.batch_size
    origins = random_origins(
        shape,
        patch_size,
        warmup_samples + args.samples,
        seed=args.seed,
        align_to_chunks=chunks if args.align_to_chunks else None,
    )
    estimates = [
        chunk_bytes_for_patch(
            shape=shape,
            chunks=chunks,
            dtype=array.dtype,
            origin=origin,
            patch_size=patch_size,
        )
        for origin in origins[warmup_samples:]
    ]
    dataset = RandomPatchDataset(args.zarr, patch_size, origins)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": False,
    }
    if args.workers > 0:
        loader_kwargs.update(
            {
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": True,
            }
        )
    loader = DataLoader(dataset, **loader_kwargs)

    checksum = 0.0
    consumed = 0
    observed = 0
    started = None
    for batch in loader:
        if consumed == warmup_samples:
            started = time.perf_counter()
        checksum += float(batch[:, :, :1, :1].sum())
        batch_count = int(batch.shape[0])
        if consumed >= warmup_samples:
            observed += batch_count
        consumed += batch_count
    if started is None:
        raise RuntimeError("benchmark did not reach the timed samples")
    elapsed = time.perf_counter() - started
    logical_bytes = sum(row["logical_bytes"] for row in estimates)
    storage_bytes = sum(row["storage_bytes"] for row in estimates)
    report = {
        "zarr": str(args.zarr),
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": str(array.dtype),
        "patch_size": list(patch_size),
        "samples": observed,
        "warmup_samples": warmup_samples,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "prefetch_factor": args.prefetch_factor if args.workers > 0 else None,
        "aligned_to_chunks": args.align_to_chunks,
        "elapsed_seconds": elapsed,
        "patches_per_second": observed / elapsed,
        "logical_mib_per_second": logical_bytes / elapsed / 1024**2,
        "estimated_storage_mib_per_second": storage_bytes / elapsed / 1024**2,
        "mean_read_amplification": storage_bytes / logical_bytes,
        "mean_chunks_per_patch": sum(row["chunks_touched"] for row in estimates)
        / len(estimates),
        "checksum": checksum,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
