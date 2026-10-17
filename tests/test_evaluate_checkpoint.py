import unittest

from vesuvius_code.checkpoints import CheckpointCompatibilityError
from vesuvius_code import evaluate_checkpoint


class EvaluateCheckpointTests(unittest.TestCase):
    def test_validation_requires_split_but_can_select_threshold(self):
        parser = evaluate_checkpoint.build_parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                "model.ckpt",
                "--zarr",
                "volume.zarr",
                "--mask",
                "mask.png",
                "--ink",
                "labels.png",
                "--split-manifest",
                "split.json",
                "--split",
                "validation",
            ]
        )

        evaluate_checkpoint.validate_args(args)

    def test_test_split_requires_a_frozen_threshold(self):
        parser = evaluate_checkpoint.build_parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                "model.ckpt",
                "--zarr",
                "volume.zarr",
                "--mask",
                "mask.png",
                "--ink",
                "labels.png",
                "--split-manifest",
                "split.json",
                "--split",
                "test",
            ]
        )

        with self.assertRaisesRegex(ValueError, "frozen threshold"):
            evaluate_checkpoint.validate_args(args)

    def test_external_ground_truth_is_allowed_unless_identity_is_required(self):
        checkpoint_audit = {
            "valid_mask": {"sha256": "old-mask"},
            "label": {"sha256": "old-label"},
        }
        current_audit = {
            "valid_mask": {"sha256": "new-mask"},
            "label": {"sha256": "new-label"},
        }

        self.assertFalse(
            evaluate_checkpoint.enforce_data_identity(
                checkpoint_audit,
                current_audit,
                require_match=False,
            )
        )
        with self.assertRaises(CheckpointCompatibilityError):
            evaluate_checkpoint.enforce_data_identity(
                checkpoint_audit,
                current_audit,
                require_match=True,
            )

    def test_identity_match_accepts_any_training_source_in_multisource_audit(self):
        current = {
            "volume": {"metadata_sha256": "volume-a"},
            "valid_mask": {"sha256": "mask-a"},
            "label": {"sha256": "label-a"},
        }
        checkpoint_audit = {
            "sources": {
                "fragment-a": current,
                "fragment-b": {
                    "volume": {"metadata_sha256": "volume-b"},
                    "valid_mask": {"sha256": "mask-b"},
                    "label": {"sha256": "label-b"},
                },
            }
        }

        self.assertTrue(
            evaluate_checkpoint.enforce_data_identity(
                checkpoint_audit,
                current,
                require_match=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
