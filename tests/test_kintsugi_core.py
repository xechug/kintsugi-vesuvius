import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr
from PIL import Image

from vesuvius_code.dataset import PapyriDataset
from vesuvius_code.losses import FableLoss
from vesuvius_code.metrics import aggregate_binary_segmentation_metrics, binary_segmentation_metrics
from vesuvius_code.model import DualStreamVMambaForense, KintsugiAttention


class KintsugiCoreTests(unittest.TestCase):
    def test_model_returns_logits_and_predicts_probabilities_separately(self):
        model = DualStreamVMambaForense(
            in_channels=1,
            base_channels=4,
            depth_topk=2,
            use_real_mamba=False,
        )
        x = torch.randn(2, 1, 8, 16, 16)

        logits = model(x)
        probs = model.predict_proba(x)

        self.assertEqual(tuple(logits.shape), (2, 1, 16, 16))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.all((probs >= 0) & (probs <= 1)))
        self.assertFalse(any(isinstance(m, torch.nn.Sigmoid) for m in model.segmentation_head.modules()))

    def test_kintsugi_gate_can_suppress_noise(self):
        attention = KintsugiAttention(in_channels=4, attention_strength=0.5)
        forensic = torch.zeros(1, 4, 3, 4, 4)
        mamba = torch.ones(1, 4, 3, 4, 4)

        with torch.no_grad():
            for parameter in attention.crack_detector.parameters():
                parameter.zero_()
            attention.crack_detector[-2].bias.fill_(-20.0)

        attended, mask = attention(forensic, mamba)

        self.assertLess(attended.mean().item(), mamba.mean().item())
        self.assertTrue(torch.all((mask >= 0) & (mask <= 1)))

    def test_fable_loss_accepts_logits_and_backpropagates(self):
        loss_fn = FableLoss(pos_weight=3.0, tversky_beta=0.7)
        logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = torch.zeros(2, 1, 8, 8)
        target[:, :, 2:4, 2:5] = 1.0

        loss = loss_fn(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_dataset_is_deterministic_per_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zarr_path = root / "volume.zarr"
            mask_path = root / "mask.tif"
            ink_path = root / "ink.tif"

            volume = zarr.open(
                str(zarr_path),
                mode="w",
                shape=(12, 24, 24),
                chunks=(4, 12, 12),
                dtype="uint16",
            )
            volume[:] = np.arange(12 * 24 * 24, dtype=np.uint16).reshape(12, 24, 24)

            mask = np.ones((24, 24), dtype=np.uint8) * 255
            ink = np.zeros((24, 24), dtype=np.uint8)
            ink[8:12, 8:12] = 255
            tifffile.imwrite(mask_path, mask)
            tifffile.imwrite(ink_path, ink)

            dataset = PapyriDataset(
                str(zarr_path),
                str(mask_path),
                str(ink_path),
                patch_size=(4, 8, 8),
                epoch_size=8,
                base_seed=123,
                z_policy="random",
                positive_fraction=0.5,
                coord_scale=4,
                quiet=True,
            )

            sample_a = dataset[3]
            sample_b = dataset[3]

            self.assertTrue(torch.equal(sample_a["image"], sample_b["image"]))
            self.assertTrue(torch.equal(sample_a["target"], sample_b["target"]))

    def test_dataset_auto_normalizes_uint8_zarr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zarr_path = root / "volume.zarr"
            mask_path = root / "mask.tif"
            ink_path = root / "ink.tif"

            volume = zarr.open(
                str(zarr_path),
                mode="w",
                shape=(4, 8, 8),
                chunks=(4, 8, 8),
                dtype="uint8",
            )
            volume[:] = np.full((4, 8, 8), 255, dtype=np.uint8)

            mask = np.ones((8, 8), dtype=np.uint8) * 255
            ink = np.zeros((8, 8), dtype=np.uint8)
            tifffile.imwrite(mask_path, mask)
            tifffile.imwrite(ink_path, ink)

            dataset = PapyriDataset(
                str(zarr_path),
                str(mask_path),
                str(ink_path),
                patch_size=(4, 8, 8),
                epoch_size=1,
                normalize_divisor="auto",
                quiet=True,
            )

            vol = dataset[0]["image"]
            self.assertAlmostEqual(float(vol.max()), 1.0, places=6)

    def test_dataset_reads_tiff_layer_folder_with_png_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = root / "layers"
            layers.mkdir()
            mask_path = root / "mask.png"
            ink_path = root / "ink.png"

            for z_idx in range(4):
                tifffile.imwrite(layers / f"{z_idx:02}.tif", np.full((8, 8), z_idx, dtype=np.uint8))

            Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(mask_path)
            ink = np.zeros((8, 8), dtype=np.uint8)
            ink[2:4, 2:4] = 255
            Image.fromarray(ink).save(ink_path)

            dataset = PapyriDataset(
                str(root),
                str(mask_path),
                str(ink_path),
                patch_size=(4, 8, 8),
                epoch_size=1,
                normalize_divisor="auto",
                quiet=True,
            )

            sample = dataset[0]
            vol = sample["image"]
            mask = sample["target"]
            self.assertEqual(tuple(vol.shape), (1, 4, 8, 8))
            self.assertEqual(tuple(mask.shape), (1, 8, 8))
            self.assertAlmostEqual(float(vol.max()), 3.0 / 255.0, places=6)

    def test_binary_segmentation_metrics_counts_precision_recall_and_fp_density(self):
        probs = torch.tensor([[[[0.9, 0.8, 0.1], [0.7, 0.2, 0.6]]]])
        target = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]])

        metrics = binary_segmentation_metrics(probs, target, threshold=0.5)

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 2)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["fp_per_mpix"], 2 / 6 * 1_000_000)

    def test_aggregate_binary_segmentation_metrics_sums_counts(self):
        metric_rows = [
            {"tp": 2, "fp": 1, "fn": 1, "tn": 6},
            {"tp": 1, "fp": 3, "fn": 0, "tn": 7},
        ]

        metrics = aggregate_binary_segmentation_metrics(metric_rows)

        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 4)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 13)
        self.assertAlmostEqual(metrics["precision"], 3 / 7)
        self.assertAlmostEqual(metrics["recall"], 3 / 4)
        self.assertAlmostEqual(metrics["fp_per_mpix"], 4 / 21 * 1_000_000)


if __name__ == "__main__":
    unittest.main()
