import unittest

import numpy as np

from scripts.benchmark_zarr_io import chunk_bytes_for_patch


class ZarrIoBenchmarkTests(unittest.TestCase):
    def test_aligned_patch_reads_one_chunk_without_amplification(self):
        result = chunk_bytes_for_patch(
            shape=(16, 256, 256),
            chunks=(8, 64, 64),
            dtype=np.dtype("uint16"),
            origin=(8, 64, 128),
            patch_size=(8, 64, 64),
        )

        self.assertEqual(result["chunks_touched"], 1)
        self.assertEqual(result["storage_bytes"], result["logical_bytes"])
        self.assertEqual(result["amplification"], 1.0)

    def test_unaligned_patch_spans_eight_chunks(self):
        result = chunk_bytes_for_patch(
            shape=(16, 256, 256),
            chunks=(8, 64, 64),
            dtype=np.dtype("uint16"),
            origin=(1, 1, 1),
            patch_size=(8, 64, 64),
        )

        self.assertEqual(result["chunks_touched"], 8)
        self.assertEqual(result["amplification"], 8.0)


if __name__ == "__main__":
    unittest.main()
