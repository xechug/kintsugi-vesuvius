import copy
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.controlled_ink_benchmark import box_sum, prepare_slab, validate_protocol


class ControlledInkProtocolTests(unittest.TestCase):
    def test_segment_overlap_is_rejected(self):
        path = Path(__file__).resolve().parents[1] / "configs/controlled_ink_20261025.json"
        config = json.loads(path.read_text())
        validate_protocol(config)
        config["test"] = config["train"][0]
        with self.assertRaises(ValueError):
            validate_protocol(config)

    def test_fast_masked_box_filter_matches_direct_sum_at_edges(self):
        values = np.arange(20, dtype=np.float32).reshape(4, 5)
        result = box_sum(values, 1)
        expected = np.empty_like(values)
        for y in range(4):
            for x in range(5):
                expected[y, x] = values[max(0,y-1):y+2, max(0,x-1):x+2].sum()
        np.testing.assert_array_equal(result, expected)

    def test_normalization_handles_empty_padding_and_depth_bounds(self):
        volume = np.zeros((28, 16, 16), np.uint8)
        volume[:, 3:13, 3:13] = np.arange(40, 68, dtype=np.uint8)[:, None, None]
        slab, valid = prepare_slab(volume)
        self.assertEqual(slab.shape, (16,16,16))
        self.assertTrue(np.isfinite(slab).all())
        np.testing.assert_array_equal(slab[:, ~valid], 0.5)
        with self.assertRaises(ValueError):
            prepare_slab(volume, offset=30)


if __name__ == "__main__":
    unittest.main()
