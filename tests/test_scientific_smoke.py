import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr

from vesuvius_code.splits import build_spatial_split


class ScientificSmokeTests(unittest.TestCase):
    def test_smoke_selects_a_mixed_validation_patch(self):
        from vesuvius_code.smoke_test import select_informative_validation_sample

        class FakeValidation:
            valid_coords = [0, 1]

            def __getitem__(self, index):
                target = torch.zeros(1, 4, 4)
                if index == 1:
                    target[:, 1:3, 1:3] = 1
                return {
                    "target": target,
                    "valid_mask": torch.ones_like(target, dtype=torch.bool),
                    "coords": torch.tensor([0, index, index]),
                }

        index, sample = select_informative_validation_sample(FakeValidation())

        self.assertEqual(index, 1)
        self.assertEqual(sample["coords"].tolist(), [0, 1, 1])

    def test_end_to_end_smoke_has_disjoint_validation_and_strict_round_trip(self):
        from vesuvius_code.smoke_test import SmokeTestConfig, run_smoke_test

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume_path = root / "volume.zarr"
            valid_path = root / "valid.tif"
            labels_path = root / "labels.tif"
            split_path = root / "split.json"
            output_dir = root / "output"

            volume = zarr.open(
                str(volume_path),
                mode="w",
                shape=(4, 64, 64),
                chunks=(4, 16, 16),
                dtype="uint8",
            )
            rng = np.random.default_rng(7)
            volume[:] = rng.integers(0, 256, size=volume.shape, dtype=np.uint8)
            valid = np.ones((64, 64), dtype=np.uint8)
            labels = np.zeros((64, 64), dtype=np.uint8)
            for y in range(4, 64, 16):
                for x in range(4, 64, 16):
                    labels[y : y + 6, x : x + 6] = 1
            tifffile.imwrite(valid_path, valid, tile=(16, 16))
            tifffile.imwrite(labels_path, labels, tile=(16, 16))
            split = build_spatial_split(
                height=64,
                width=64,
                block_size=32,
                guard_band=4,
                fractions=(0.5, 0.25, 0.25),
                seed=5,
            )
            split.save(split_path)

            report = run_smoke_test(
                SmokeTestConfig(
                    volume=str(volume_path),
                    valid_mask=str(valid_path),
                    labels=str(labels_path),
                    split_manifest=str(split_path),
                    output_dir=str(output_dir),
                    model_name="residual_unet",
                    backend="fallback",
                    base_channels=2,
                    patch_size=(4, 8, 8),
                    coord_scale=2,
                    train_steps=2,
                    validation_samples=2,
                    seed=11,
                )
            )

            self.assertTrue(report["data_audit_ok"])
            self.assertTrue(report["checkpoint_round_trip"])
            self.assertFalse(report["split_overlap"])
            self.assertEqual(report["train_steps"], 2)
            self.assertTrue(Path(report["checkpoint"]).is_file())
            self.assertTrue(np.isfinite(report["validation"]["brier"]))
            self.assertTrue(np.isfinite(report["validation"]["average_precision"]))
            self.assertGreaterEqual(report["validation"]["selected_threshold"], 0.0)
            self.assertLessEqual(report["validation"]["selected_threshold"], 1.0)
            self.assertIn("f1_at_selected_threshold", report["validation"])
            json.dumps(report, allow_nan=False)
            restored = json.loads((output_dir / "smoke_report.json").read_text())
            self.assertEqual(restored["checkpoint_schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
