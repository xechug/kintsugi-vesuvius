from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vesuvius_code.data_contract import sha256_file


def convert_raster_to_tiled_tiff(
    source: str | Path,
    destination: str | Path,
    *,
    tile_shape=(256, 256),
    overwrite=False,
):
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    tile_shape = tuple(int(value) for value in tile_shape)
    if len(tile_shape) != 2 or any(value <= 0 or value % 16 for value in tile_shape):
        raise ValueError("TIFF tile dimensions must be positive multiples of 16")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(source) as image:
            array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected a 2D raster, got {array.shape}")
    observed = set(float(value) for value in np.unique(array))
    if not observed.issubset({0.0, 1.0, 255.0}):
        raise ValueError(f"expected a binary raster, observed {sorted(observed)[:16]}")
    array = np.ascontiguousarray(array)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(
        f".{destination.stem}.partial{destination.suffix or '.tif'}"
    )
    if partial.exists():
        partial.unlink()
    description = json.dumps(
        {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "conversion": "binary raster to tiled TIFF",
        },
        sort_keys=True,
    )
    try:
        tifffile.imwrite(
            partial,
            array,
            tile=tile_shape,
            compression="deflate",
            metadata=None,
            description=description,
        )
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def main():
    parser = argparse.ArgumentParser(description="Convert a binary raster to tiled TIFF")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--tile-height", type=int, default=256)
    parser.add_argument("--tile-width", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = convert_raster_to_tiled_tiff(
        args.source,
        args.destination,
        tile_shape=(args.tile_height, args.tile_width),
        overwrite=args.overwrite,
    )
    with tifffile.TiffFile(result) as tif:
        page = tif.pages[0]
        print(
            f"saved={result} shape={page.shape} dtype={page.dtype} "
            f"tile=({page.tilelength},{page.tilewidth})"
        )


if __name__ == "__main__":
    main()
