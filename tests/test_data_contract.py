import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import zarr


def load_api(testcase):
    spec = importlib.util.find_spec("vesuvius_code.data_contract")
    testcase.assertIsNotNone(spec, "vesuvius_code.data_contract must exist")
    return importlib.import_module("vesuvius_code.data_contract")


class DataContractTests(unittest.TestCase):
    def make_volume(self, root):
        path = root / "volume.zarr"
        volume = zarr.open(
            str(path),
            mode="w",
            shape=(4, 16, 16),
            chunks=(2, 8, 8),
            dtype="uint8",
        )
        volume[:] = 7
        return path

    def write_raster(self, path, values):
        tifffile.imwrite(path, np.asarray(values, dtype=np.uint8), tile=(16, 16))

    def test_identical_valid_mask_and_labels_are_rejected(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = self.make_volume(root)
            valid = root / "valid.tif"
            labels = root / "labels.tif"
            pixels = np.zeros((16, 16), dtype=np.uint8)
            pixels[4:12, 4:12] = 255
            self.write_raster(valid, pixels)
            self.write_raster(labels, pixels)

            report = api.audit_training_data(volume, valid, labels, label_kind="ground_truth")

            with self.assertRaisesRegex(api.DataContractError, "identical"):
                report.raise_for_training()

    def test_non_binary_validity_mask_is_rejected(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = self.make_volume(root)
            valid = root / "valid.tif"
            labels = root / "labels.tif"
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[2:14, 2:14] = 127
            labels_array = np.zeros((16, 16), dtype=np.uint8)
            labels_array[7:9, 7:9] = 255
            self.write_raster(valid, mask)
            self.write_raster(labels, labels_array)

            report = api.audit_training_data(volume, valid, labels, label_kind="ground_truth")

            with self.assertRaisesRegex(api.DataContractError, "valid.*binary"):
                report.raise_for_training()

    def test_soft_labels_require_teacher_role(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = self.make_volume(root)
            valid = root / "valid.tif"
            labels = root / "labels.tif"
            self.write_raster(valid, np.full((16, 16), 255, dtype=np.uint8))
            scores = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
            self.write_raster(labels, scores)

            ground_truth = api.audit_training_data(
                volume, valid, labels, label_kind="ground_truth"
            )
            teacher = api.audit_training_data(volume, valid, labels, label_kind="teacher")

            with self.assertRaisesRegex(api.DataContractError, "ground_truth.*binary"):
                ground_truth.raise_for_training()
            teacher.raise_for_training()
            self.assertEqual(teacher.label.kind, "teacher")
            self.assertFalse(teacher.label.is_binary)

    def test_tiff_raster_opens_lazily_and_reads_slices(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raster.tif"
            pixels = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
            self.write_raster(path, pixels)

            raster = api.open_raster(path)

            self.assertFalse(raster.is_open)
            self.assertEqual(raster.shape, (16, 16))
            self.assertFalse(raster.is_open)
            np.testing.assert_array_equal(raster[3:6, 4:8], pixels[3:6, 4:8])
            self.assertTrue(raster.is_open)

    def test_report_is_json_serializable_and_contains_fingerprints(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = self.make_volume(root)
            valid = root / "valid.tif"
            labels = root / "labels.tif"
            valid_pixels = np.full((16, 16), 255, dtype=np.uint8)
            label_pixels = np.zeros((16, 16), dtype=np.uint8)
            label_pixels[3:5, 3:5] = 255
            self.write_raster(valid, valid_pixels)
            self.write_raster(labels, label_pixels)

            report = api.audit_training_data(volume, valid, labels, label_kind="ground_truth")
            payload = report.to_dict()

            report.raise_for_training()
            self.assertEqual(len(payload["valid_mask"]["sha256"]), 64)
            self.assertEqual(len(payload["label"]["sha256"]), 64)
            self.assertNotEqual(payload["valid_mask"]["sha256"], payload["label"]["sha256"])
            self.assertEqual(payload["volume"]["shape"], [4, 16, 16])

            output = root / "audit.json"
            report.save(output)
            self.assertEqual(json.loads(output.read_text()), payload)

    def test_dataset_keeps_large_rasters_lazy(self):
        api = load_api(self)
        from vesuvius_code.dataset import PapyriDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = self.make_volume(root)
            valid = root / "valid.tif"
            labels = root / "labels.tif"
            valid_pixels = np.full((16, 16), 255, dtype=np.uint8)
            label_pixels = np.zeros((16, 16), dtype=np.uint8)
            label_pixels[4:8, 4:8] = 255
            self.write_raster(valid, valid_pixels)
            self.write_raster(labels, label_pixels)

            dataset = PapyriDataset(
                str(volume),
                str(valid),
                str(labels),
                patch_size=(2, 8, 8),
                epoch_size=2,
                coord_scale=2,
                quiet=True,
            )

            self.assertIsInstance(dataset.mask, api.LazyRaster)
            self.assertIsInstance(dataset.ink_mask, api.LazyRaster)

    def test_dataset_rejects_duplicate_mask_and_labels_by_default(self):
        api = load_api(self)
        from vesuvius_code.dataset import PapyriDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = self.make_volume(root)
            valid = root / "valid.tif"
            labels = root / "labels.tif"
            pixels = np.full((16, 16), 255, dtype=np.uint8)
            self.write_raster(valid, pixels)
            self.write_raster(labels, pixels)

            with self.assertRaisesRegex(api.DataContractError, "identical"):
                PapyriDataset(
                    str(volume),
                    str(valid),
                    str(labels),
                    patch_size=(2, 8, 8),
                    epoch_size=1,
                    quiet=True,
                )

    def test_dataset_preserves_binary_zero_one_labels(self):
        from PIL import Image
        from vesuvius_code.dataset import PapyriDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = self.make_volume(root)
            valid = root / "valid.png"
            labels = root / "labels.png"
            Image.fromarray(np.ones((16, 16), dtype=np.uint8)).save(valid)
            label_pixels = np.zeros((16, 16), dtype=np.uint8)
            label_pixels[4:12, 4:12] = 1
            Image.fromarray(label_pixels).save(labels)
            dataset = PapyriDataset(
                str(volume),
                str(valid),
                str(labels),
                patch_size=(2, 16, 16),
                epoch_size=1,
                positive_fraction=1.0,
                coord_scale=1,
                quiet=True,
            )

            target = dataset[0]["target"]

            self.assertEqual(float(target.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
