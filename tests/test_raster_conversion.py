import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from scripts.convert_raster_to_tiled_tiff import convert_raster_to_tiled_tiff


class RasterConversionTests(unittest.TestCase):
    def test_conversion_preserves_binary_pixels_and_creates_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mask.png"
            destination = root / "mask.tif"
            expected = np.zeros((40, 48), dtype=np.uint8)
            expected[5:20, 7:30] = 1
            Image.fromarray(expected).save(source)

            convert_raster_to_tiled_tiff(source, destination, tile_shape=(16, 16))

            np.testing.assert_array_equal(tifffile.imread(destination), expected)
            with tifffile.TiffFile(destination) as tif:
                self.assertTrue(tif.pages[0].is_tiled)
                self.assertEqual(tif.pages[0].tilelength, 16)
                self.assertEqual(tif.pages[0].tilewidth, 16)

    def test_conversion_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mask.png"
            destination = root / "mask.tif"
            Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(source)
            destination.touch()

            with self.assertRaises(FileExistsError):
                convert_raster_to_tiled_tiff(source, destination)


if __name__ == "__main__":
    unittest.main()
