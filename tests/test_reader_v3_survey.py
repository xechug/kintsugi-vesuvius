import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.reader_v3_survey import NumpyAccumulator, download_native


class ReaderSurveyContracts(unittest.TestCase):
    def test_overlap_normalizes_constant_scores_including_boundaries(self):
        accumulator = NumpyAccumulator((5, 5))
        weight = np.outer(np.hanning(4), np.hanning(4)).astype(np.float32)
        weight = np.maximum(weight, 1e-3)
        for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
            accumulator.add_tile(y0=y, x0=x, tile=np.full((4, 4), 0.75), tile_weights=weight)
        np.testing.assert_allclose(accumulator.result(), 0.75, atol=1e-7)

    def test_sparse_native_download_preserves_fill_and_clips_partial_edges(self):
        metadata = dict(shape=[3, 3, 5], chunks=[3, 2, 2], dtype="|u1", order="C",
                        fill_value=7, dimension_separator="/", compressor=None, filters=None)

        def response(url):
            if url.endswith(".zarray"):
                return json.dumps(metadata).encode()
            if url.endswith(".zattrs"):
                return b"{}"
            cy, cx = map(int, url.split("/")[-2:])
            if (cy, cx) == (0, 1):
                raise urllib.error.HTTPError(url, 404, "No such chunk", {}, None)
            return np.full((3, 2, 2), 10 + cy * 3 + cx, np.uint8).tobytes()

        with tempfile.TemporaryDirectory() as folder, patch("scripts.reader_v3_survey.read_url", side_effect=response):
            destination = Path(folder) / "native.npy"
            source = download_native("https://example.test/volume", destination)
            volume = np.load(destination)
            self.assertEqual(volume.shape, (3, 3, 5))
            np.testing.assert_array_equal(volume[:, :2, 2:4], 7)
            np.testing.assert_array_equal(volume[:, 2:, 4:], 15)
            self.assertEqual(source["absent_fill_chunks"], [[0, 1]])
            self.assertEqual(source["chunks_total"], 6)
            # Cache reads must detect damaged CT rather than reuse it silently.
            destination.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                download_native("https://example.test/volume", destination)

    def test_network_failure_cannot_be_treated_as_empty_papyrus(self):
        metadata = dict(shape=[3, 2, 2], chunks=[3, 2, 2], dtype="|u1", order="C",
                        fill_value=0, compressor=None, filters=None)

        def response(url):
            if url.endswith(".zarray"):
                return json.dumps(metadata).encode()
            raise urllib.error.HTTPError(url, 503, "Unavailable", {}, None)

        with tempfile.TemporaryDirectory() as folder, patch("scripts.reader_v3_survey.read_url", side_effect=response):
            destination = Path(folder) / "native.npy"
            with self.assertRaises(urllib.error.HTTPError):
                download_native("https://example.test/volume", destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
