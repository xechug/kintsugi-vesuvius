import unittest
from contextlib import redirect_stderr
from io import StringIO

from vesuvius_code import cloud_lightning_train


class CloudEntrypointTests(unittest.TestCase):
    def test_parser_requires_spatial_split_and_builds_training_config(self):
        parser = cloud_lightning_train.build_parser()

        incomplete = parser.parse_args(["--zarr", "v", "--mask", "m", "--ink", "i"])
        with self.assertRaisesRegex(ValueError, "split-manifest"):
            cloud_lightning_train.validate_data_args(incomplete)

        args = parser.parse_args(
            [
                "--zarr",
                "volume.zarr",
                "--mask",
                "valid.tif",
                "--ink",
                "labels.tif",
                "--split-manifest",
                "split.json",
                "--gpu-model",
                "NVIDIA A100 80GB",
                "--devices",
                "2",
                "--batch-size",
                "1",
                "--accumulate",
                "4",
            ]
        )
        cloud_lightning_train.validate_data_args(args)
        config = cloud_lightning_train.training_config_from_args(
            args, validation_regions=3
        )

        self.assertEqual(config.effective_global_batch_size, 8)
        self.assertEqual(args.model, "kintsugi_axial")
        self.assertEqual(args.backend, "mamba")
        self.assertTrue(args.align_to_chunks)

    def test_parser_accepts_manifest_instead_of_single_source_paths(self):
        parser = cloud_lightning_train.build_parser()
        args = parser.parse_args(["--data-manifest", "sources.json"])

        cloud_lightning_train.validate_data_args(args)

        conflicting = parser.parse_args(
            ["--data-manifest", "sources.json", "--zarr", "volume.zarr"]
        )
        with self.assertRaisesRegex(ValueError, "either"):
            cloud_lightning_train.validate_data_args(conflicting)


if __name__ == "__main__":
    unittest.main()
