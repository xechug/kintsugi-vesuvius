import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


def load_api(testcase):
    spec = importlib.util.find_spec("vesuvius_code.checkpoints")
    testcase.assertIsNotNone(spec, "vesuvius_code.checkpoints must exist")
    return importlib.import_module("vesuvius_code.checkpoints")


class CheckpointTests(unittest.TestCase):
    def make_training_objects(self, out_features=2):
        model = torch.nn.Linear(3, out_features)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        return model, optimizer, scheduler

    def test_round_trip_restores_full_training_metadata(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ckpt"
            model, optimizer, scheduler = self.make_training_objects()
            run_config = {"model": {"in_features": 3, "out_features": 2}, "seed": 17}

            api.save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                run_config=run_config,
                data_audit={"label": {"sha256": "abc"}},
                epoch=3,
                global_step=41,
            )
            restored, restored_optimizer, restored_scheduler = self.make_training_objects()
            payload = api.load_checkpoint(
                path,
                model=restored,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                expected_config=run_config,
                strict=True,
            )

            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["global_step"], 41)
            self.assertIn("torch", payload["rng_state"])
            self.assertIn("numpy", payload["rng_state"])
            for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
                self.assertTrue(torch.equal(expected, actual))

    def test_architecture_mismatch_is_rejected_even_when_strict_requested(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ckpt"
            model, optimizer, scheduler = self.make_training_objects(out_features=2)
            api.save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                run_config={"model": {"out_features": 2}},
                data_audit={},
                epoch=0,
                global_step=0,
            )
            incompatible = torch.nn.Linear(3, 4)

            with self.assertRaisesRegex(api.CheckpointCompatibilityError, "model state"):
                api.load_checkpoint(path, model=incompatible, strict=True)

    def test_expected_configuration_mismatch_is_rejected(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ckpt"
            model, optimizer, scheduler = self.make_training_objects()
            api.save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                run_config={"model": {"backend": "fallback"}},
                data_audit={},
                epoch=0,
                global_step=0,
            )

            with self.assertRaisesRegex(api.CheckpointCompatibilityError, "configuration"):
                api.load_checkpoint(
                    path,
                    model=model,
                    expected_config={"model": {"backend": "mamba"}},
                )

    def test_legacy_or_partial_checkpoint_is_rejected(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pth"
            model, _, _ = self.make_training_objects()
            torch.save({"model": model.state_dict()}, path)

            with self.assertRaisesRegex(api.CheckpointCompatibilityError, "schema"):
                api.load_checkpoint(path, model=model)

    def test_save_is_atomic_and_replaces_previous_checkpoint(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "model.ckpt"
            model, optimizer, scheduler = self.make_training_objects()
            for epoch in (1, 2):
                api.save_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    run_config={"model": {"out_features": 2}},
                    data_audit={},
                    epoch=epoch,
                    global_step=epoch * 10,
                )

            payload = api.load_checkpoint(path, model=model)

            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_read_metadata_does_not_expose_model_tensors(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ckpt"
            model, optimizer, scheduler = self.make_training_objects()
            api.save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                run_config={"model": {"out_features": 2}},
                data_audit={"volume": {"shape": [2, 3, 4]}},
                epoch=2,
                global_step=9,
            )

            metadata = api.read_checkpoint_metadata(path)

            self.assertEqual(metadata["run_config"]["model"]["out_features"], 2)
            self.assertEqual(metadata["epoch"], 2)
            self.assertNotIn("model_state", metadata)
            self.assertNotIn("optimizer_state", metadata)

    def test_lightning_checkpoint_exports_to_strict_inference_schema(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lightning_path = root / "best-lightning.ckpt"
            strict_path = root / "best.strict.ckpt"
            source, _, _ = self.make_training_objects()
            torch.save(
                {
                    "state_dict": {
                        f"model.{name}": tensor.clone()
                        for name, tensor in source.state_dict().items()
                    },
                    "epoch": 7,
                    "global_step": 123,
                },
                lightning_path,
            )
            export_model, _, _ = self.make_training_objects()
            run_config = {"model": {"in_features": 3, "out_features": 2}}

            api.export_lightning_checkpoint(
                lightning_path,
                strict_path,
                model=export_model,
                run_config=run_config,
                data_audit={"sources": {"fragment": {"label": "verified"}}},
            )
            restored, _, _ = self.make_training_objects()
            payload = api.load_checkpoint(
                strict_path,
                model=restored,
                expected_config=run_config,
            )

            self.assertEqual(payload["epoch"], 7)
            self.assertEqual(payload["global_step"], 123)
            for expected, actual in zip(source.parameters(), restored.parameters(), strict=True):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
