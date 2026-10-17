import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr

from vesuvius_code.dataset import BalancedMultiSourceDataset, PapyriDataset


def load_splits(testcase):
    spec = importlib.util.find_spec("vesuvius_code.splits")
    testcase.assertIsNotNone(spec, "vesuvius_code.splits must exist")
    return importlib.import_module("vesuvius_code.splits")


class SpatialSamplingTests(unittest.TestCase):
    def make_data(self, root, shape=(8, 64, 64), ink=None):
        volume_path = root / "volume.zarr"
        valid_path = root / "valid.tif"
        ink_path = root / "ink.tif"
        volume = zarr.open(
            str(volume_path),
            mode="w",
            shape=shape,
            chunks=(4, 32, 32),
            dtype="uint8",
        )
        volume[:] = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        valid = np.full(shape[-2:], 255, dtype=np.uint8)
        if ink is None:
            ink = np.zeros(shape[-2:], dtype=np.uint8)
            ink[12:20, 12:20] = 255
            ink[42:48, 42:50] = 255
        tifffile.imwrite(valid_path, valid, tile=(16, 16))
        tifffile.imwrite(ink_path, ink, tile=(16, 16))
        return volume_path, valid_path, ink_path

    @staticmethod
    def rectangles_overlap(a, b):
        return not (a.y1 <= b.y0 or b.y1 <= a.y0 or a.x1 <= b.x0 or b.x1 <= a.x0)

    def test_spatial_split_has_disjoint_guarded_regions(self):
        api = load_splits(self)

        manifest = api.build_spatial_split(
            height=128,
            width=128,
            block_size=64,
            guard_band=8,
            fractions=(0.5, 0.5, 0.0),
            seed=4,
        )

        train = manifest.regions("train")
        validation = manifest.regions("validation")
        self.assertTrue(train)
        self.assertTrue(validation)
        for train_region in train:
            for validation_region in validation:
                self.assertFalse(self.rectangles_overlap(train_region, validation_region))
                expanded_train = train_region.expand(8, height=128, width=128)
                self.assertFalse(self.rectangles_overlap(expanded_train, validation_region))

    def test_patch_is_positive_from_coverage_not_top_left_pixel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ink = np.zeros((32, 32), dtype=np.uint8)
            ink[6:8, 6:8] = 255
            volume, valid, labels = self.make_data(root, shape=(4, 32, 32), ink=ink)

            dataset = PapyriDataset(
                str(volume),
                str(valid),
                str(labels),
                patch_size=(2, 8, 8),
                epoch_size=4,
                coord_scale=1,
                min_positive_fraction=0.01,
                max_negative_fraction=0.0,
                quiet=True,
            )

            positive_origins = {tuple(coord) for coord in dataset.ink_coords.tolist()}
            self.assertIn((0, 0), positive_origins)
            self.assertEqual(int(ink[0, 0]), 0)

    def test_training_resamples_each_epoch_but_validation_is_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume, valid, labels = self.make_data(root)
            common = dict(
                zarr_path=str(volume),
                mask_path=str(valid),
                ink_path=str(labels),
                patch_size=(4, 16, 16),
                epoch_size=12,
                positive_fraction=0.5,
                coord_scale=2,
                quiet=True,
            )
            training = PapyriDataset(**common, resample_each_epoch=True)
            validation = PapyriDataset(**common, resample_each_epoch=False, base_seed=9001)

            train_epoch_0 = [tuple(training[idx]["coords"].tolist()) for idx in range(12)]
            val_epoch_0 = [tuple(validation[idx]["coords"].tolist()) for idx in range(12)]
            training.set_epoch(1)
            validation.set_epoch(1)
            train_epoch_1 = [tuple(training[idx]["coords"].tolist()) for idx in range(12)]
            val_epoch_1 = [tuple(validation[idx]["coords"].tolist()) for idx in range(12)]

            self.assertNotEqual(train_epoch_0, train_epoch_1)
            self.assertEqual(val_epoch_0, val_epoch_1)

    def test_dataset_returns_valid_mask_and_respects_allowed_regions(self):
        splits = load_splits(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume, valid, labels = self.make_data(root)
            region = splits.SpatialRegion("train", 8, 8, 40, 40, block_id="chosen")
            dataset = PapyriDataset(
                str(volume),
                str(valid),
                str(labels),
                patch_size=(4, 16, 16),
                epoch_size=8,
                coord_scale=2,
                allowed_regions=[region],
                quiet=True,
            )

            sample = dataset[0]

            self.assertEqual(set(sample), {"image", "target", "valid_mask", "coords"})
            self.assertEqual(tuple(sample["image"].shape), (1, 4, 16, 16))
            self.assertEqual(tuple(sample["target"].shape), (1, 16, 16))
            self.assertTrue(torch.all(sample["valid_mask"] == 1))
            _, y, x = sample["coords"].tolist()
            self.assertGreaterEqual(y, region.y0)
            self.assertGreaterEqual(x, region.x0)
            self.assertLessEqual(y + 16, region.y1)
            self.assertLessEqual(x + 16, region.x1)

    def test_uniform_sampling_coordinates_do_not_depend_on_ink_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume, valid, labels_a = self.make_data(root)
            labels_b = root / "ink_b.tif"
            alternate = np.zeros((64, 64), dtype=np.uint8)
            alternate[24:40, 24:40] = 255
            tifffile.imwrite(labels_b, alternate, tile=(16, 16))
            common = dict(
                zarr_path=str(volume),
                mask_path=str(valid),
                patch_size=(4, 16, 16),
                epoch_size=24,
                coord_scale=2,
                sampling_mode="uniform",
                resample_each_epoch=False,
                base_seed=81,
                quiet=True,
            )

            first = PapyriDataset(ink_path=str(labels_a), **common)
            second = PapyriDataset(ink_path=str(labels_b), **common)

            first_coords = [tuple(first[index]["coords"].tolist()) for index in range(24)]
            second_coords = [tuple(second[index]["coords"].tolist()) for index in range(24)]
            self.assertEqual(first_coords, second_coords)
            unique_coords = {
                tuple(first[index]["coords"].tolist()) for index in range(100)
            }
            self.assertEqual(len(unique_coords), 100)

    def test_grid_sampling_is_finite_and_non_overlapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume, valid, labels = self.make_data(root)
            dataset = PapyriDataset(
                zarr_path=str(volume),
                mask_path=str(valid),
                ink_path=str(labels),
                patch_size=(4, 16, 16),
                epoch_size=100,
                coord_scale=2,
                sampling_mode="grid",
                resample_each_epoch=False,
                quiet=True,
            )

            self.assertEqual(len(dataset), 16)
            footprints = []
            for index in range(len(dataset)):
                _, y, x = dataset[index]["coords"].tolist()
                footprints.append((y, x, y + 16, x + 16))

            self.assertEqual(len(set(footprints)), len(footprints))
            for index, first in enumerate(footprints):
                for second in footprints[index + 1 :]:
                    separated = (
                        first[2] <= second[0]
                        or second[2] <= first[0]
                        or first[3] <= second[1]
                        or second[3] <= first[1]
                    )
                    self.assertTrue(separated)

    def test_multisource_dataset_balances_sources_and_propagates_epoch(self):
        class TinyDataset:
            def __init__(self, value):
                self.value = value
                self.epochs = []

            def __getitem__(self, index):
                return {"value": torch.tensor(self.value * 100 + index)}

            def set_epoch(self, epoch):
                self.epochs.append(epoch)

        first = TinyDataset(1)
        second = TinyDataset(2)
        mixed = BalancedMultiSourceDataset([first, second], epoch_size=5)

        rows = [mixed[index] for index in range(len(mixed))]
        self.assertEqual([row["source_index"].item() for row in rows], [0, 1, 0, 1, 0])
        self.assertEqual([row["value"].item() for row in rows], [100, 200, 101, 201, 102])
        mixed.set_epoch(3)
        self.assertEqual(first.epochs, [3])
        self.assertEqual(second.epochs, [3])

    def test_xy_augmentation_keeps_volume_labels_and_validity_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume_path = root / "volume.zarr"
            valid_path = root / "valid.tif"
            labels_path = root / "labels.tif"
            labels = np.zeros((16, 16), dtype=np.uint8)
            labels[2:7, 4:11] = 255
            valid = np.ones((16, 16), dtype=np.uint8) * 255
            volume = zarr.open(
                str(volume_path),
                mode="w",
                shape=(4, 16, 16),
                chunks=(4, 16, 16),
                dtype="uint8",
            )
            volume[:] = labels[None, :, :]
            tifffile.imwrite(valid_path, valid, tile=(16, 16))
            tifffile.imwrite(labels_path, labels, tile=(16, 16))
            dataset = PapyriDataset(
                volume_path,
                valid_path,
                labels_path,
                patch_size=(4, 16, 16),
                epoch_size=2,
                coord_scale=1,
                positive_fraction=1.0,
                augment_xy=True,
                quiet=True,
            )

            sample = dataset[0]

            self.assertTrue(
                torch.equal(sample["image"][0, 0] > 0.5, sample["target"][0] > 0.5)
            )
            self.assertTrue(torch.all(sample["valid_mask"]))

    def test_chunk_aligned_sampling_uses_chunk_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume, valid, labels = self.make_data(root)
            dataset = PapyriDataset(
                volume,
                valid,
                labels,
                patch_size=(4, 16, 16),
                epoch_size=16,
                coord_scale=2,
                align_to_chunks=True,
                quiet=True,
            )

            for index in range(16):
                _, y, x = dataset[index]["coords"].tolist()
                self.assertEqual(y % 32, 0)
                self.assertEqual(x % 32, 0)


if __name__ == "__main__":
    unittest.main()
