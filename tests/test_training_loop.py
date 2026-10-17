import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import zarr

from vesuvius_code.checkpoints import read_checkpoint_metadata
from vesuvius_code.train import train_model


class TrainingLoopTests(unittest.TestCase):
    def test_training_requires_explicit_audited_data_paths(self):
        with self.assertRaisesRegex(ValueError, "explicitly provided"):
            train_model()

    def test_tiny_training_run_writes_versioned_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume_path = root / "volume.zarr"
            valid_path = root / "valid.tif"
            labels_path = root / "labels.tif"
            checkpoint_path = root / "model.ckpt"
            volume = zarr.open(
                str(volume_path),
                mode="w",
                shape=(6, 24, 24),
                chunks=(3, 12, 12),
                dtype="uint8",
            )
            rng = np.random.default_rng(2)
            volume[:] = rng.integers(0, 256, size=volume.shape, dtype=np.uint8)
            valid = np.full((24, 24), 255, dtype=np.uint8)
            labels = np.zeros((24, 24), dtype=np.uint8)
            labels[6:12, 6:12] = 255
            tifffile.imwrite(valid_path, valid, tile=(16, 16))
            tifffile.imwrite(labels_path, labels, tile=(16, 16))

            result = train_model(
                zarr_dir=str(volume_path),
                mask_file=str(valid_path),
                ink_file=str(labels_path),
                save_path=str(checkpoint_path),
                epochs=1,
                batch_size=1,
                num_workers=0,
                base_channels=2,
                patch_size=(4, 12, 12),
                coord_scale=2,
                epoch_size=2,
                device_override="cpu",
            )
            metadata = read_checkpoint_metadata(result)

            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["epoch"], 0)
            self.assertEqual(metadata["global_step"], 2)
            self.assertEqual(metadata["run_config"]["model"]["base_channels"], 2)
            self.assertIn("valid_mask", metadata["data_audit"])


if __name__ == "__main__":
    unittest.main()
