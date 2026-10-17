import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import tifffile
import zarr


class TiffConversionTests(unittest.TestCase):
    def test_conversion_cli_help_runs_from_repository_root(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/convert_tiff_stack_to_zarr.py", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Convert a TIFF layer stack", result.stdout)

    def test_conversion_preserves_zyx_values_and_provenance(self):
        from scripts.convert_tiff_stack_to_zarr import convert_tiff_stack

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = root / "layers"
            layers.mkdir()
            expected = []
            for index in range(3):
                layer = np.arange(12 * 10, dtype=np.uint16).reshape(12, 10) + index * 1000
                expected.append(layer)
                tifffile.imwrite(layers / f"{index:02d}.tif", layer)
            destination = root / "volume.zarr"

            convert_tiff_stack(
                layers,
                destination,
                chunk_shape=(3, 6, 5),
                overwrite=False,
            )
            restored = zarr.open(str(destination), mode="r")

            np.testing.assert_array_equal(restored[:], np.stack(expected))
            self.assertEqual(tuple(restored.chunks), (3, 6, 5))
            self.assertEqual(restored.attrs["axis_order"], "ZYX")
            self.assertEqual(len(restored.attrs["source_files"]), 3)

    def test_conversion_refuses_to_overwrite_existing_destination(self):
        from scripts.convert_tiff_stack_to_zarr import convert_tiff_stack

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = root / "layers"
            layers.mkdir()
            tifffile.imwrite(layers / "00.tif", np.zeros((4, 4), dtype=np.uint8))
            destination = root / "volume.zarr"
            destination.mkdir()

            with self.assertRaises(FileExistsError):
                convert_tiff_stack(layers, destination, chunk_shape=(1, 4, 4))


if __name__ == "__main__":
    unittest.main()
