from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr
from numcodecs import Blosc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vesuvius_code.data_contract import sha256_file


def convert_tiff_stack(
    source: str | Path,
    destination: str | Path,
    *,
    chunk_shape=(16, 128, 128),
    overwrite=False,
    hash_sources=False,
):
    source = Path(source)
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    layer_dir = source / "layers" if (source / "layers").is_dir() else source
    files = sorted(layer_dir.glob("*.tif")) + sorted(layer_dir.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No TIFF layers found in {layer_dir}")

    memory_maps = []
    for file in files:
        try:
            memory_maps.append(tifffile.memmap(file, mode="r"))
        except ValueError as error:
            raise ValueError(
                f"TIFF {file} is not memory-mappable; decompress it before conversion"
            ) from error
    first = memory_maps[0]
    if first.ndim != 2:
        raise ValueError(f"Expected 2D layers, got {first.shape}")
    if any(layer.shape != first.shape or layer.dtype != first.dtype for layer in memory_maps):
        raise ValueError("All TIFF layers must have identical shape and dtype")

    shape = (len(memory_maps), int(first.shape[0]), int(first.shape[1]))
    chunks = tuple(min(int(requested), size) for requested, size in zip(chunk_shape, shape, strict=True))
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    if destination.exists():
        shutil.rmtree(destination)
    partial.parent.mkdir(parents=True, exist_ok=True)

    try:
        output = zarr.open(
            str(partial),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=first.dtype,
            zarr_format=2,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        )
        output.attrs.update(
            {
                "axis_order": "ZYX",
                "source_directory": str(source),
                "source_files": [file.name for file in files],
                "source_sizes": [file.stat().st_size for file in files],
            }
        )
        if hash_sources:
            output.attrs["source_sha256"] = [sha256_file(file) for file in files]

        _, chunk_y, chunk_x = chunks
        for y0 in range(0, shape[1], chunk_y):
            y1 = min(y0 + chunk_y, shape[1])
            for x0 in range(0, shape[2], chunk_x):
                x1 = min(x0 + chunk_x, shape[2])
                block = np.stack(
                    [np.asarray(layer[y0:y1, x0:x1]) for layer in memory_maps],
                    axis=0,
                )
                output[:, y0:y1, x0:x1] = block
        os.replace(partial, destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description="Convert a TIFF layer stack to patchable Zarr")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--chunk-depth", type=int, default=16)
    parser.add_argument("--chunk-height", type=int, default=128)
    parser.add_argument("--chunk-width", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-sources", action="store_true")
    args = parser.parse_args()
    result = convert_tiff_stack(
        args.source,
        args.destination,
        chunk_shape=(args.chunk_depth, args.chunk_height, args.chunk_width),
        overwrite=args.overwrite,
        hash_sources=args.hash_sources,
    )
    array = zarr.open(str(result), mode="r")
    print(f"saved={result} shape={array.shape} chunks={array.chunks} dtype={array.dtype}")


if __name__ == "__main__":
    main()
