import unittest

import torch

from vesuvius_code.losses import FableLoss
from vesuvius_code.metrics import (
    average_precision,
    binary_segmentation_metrics,
    probability_calibration_metrics,
    threshold_sweep_metrics,
)


class ValidAwareLossMetricTests(unittest.TestCase):
    def test_invalid_pixels_have_zero_loss_gradient(self):
        logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]])
        valid = torch.tensor([[[[1, 1, 0], [0, 0, 0]]]], dtype=torch.bool)
        loss_fn = FableLoss(pos_weight=1.0)

        loss = loss_fn(logits, target, valid_mask=valid)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(logits.grad[~valid] == 0))
        self.assertTrue(torch.any(logits.grad[valid] != 0))

    def test_tversky_skips_empty_target_samples(self):
        logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
        target = torch.zeros_like(logits)
        valid = torch.ones_like(logits, dtype=torch.bool)
        loss_fn = FableLoss(bce_weight=0.0, tversky_weight=1.0)

        loss = loss_fn(logits, target, valid_mask=valid)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.all(logits.grad == 0))

    def test_metrics_ignore_invalid_pixels_and_use_valid_denominator(self):
        probs = torch.tensor([[[[0.9, 0.9, 0.8], [0.1, 0.7, 0.6]]]])
        target = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
        valid = torch.tensor([[[[1, 0, 0], [1, 1, 0]]]], dtype=torch.bool)

        metrics = binary_segmentation_metrics(
            probs, target, threshold=0.5, valid_mask=valid
        )

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["valid_pixels"], 3)
        self.assertEqual(metrics["fp_per_mpix"], 0.0)

    def test_brier_and_ece_use_only_valid_pixels(self):
        probs = torch.tensor([0.0, 0.5, 1.0, 1.0])
        target = torch.tensor([0.0, 1.0, 1.0, 0.0])
        valid = torch.tensor([1, 1, 1, 0], dtype=torch.bool)

        metrics = probability_calibration_metrics(
            probs, target, valid_mask=valid, num_bins=2
        )

        self.assertEqual(metrics["valid_pixels"], 3)
        self.assertAlmostEqual(metrics["brier"], (0.0 + 0.25 + 0.0) / 3.0)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)

    def test_all_invalid_loss_remains_differentiable_and_zero(self):
        logits = torch.randn(1, 1, 3, 3, requires_grad=True)
        target = torch.ones_like(logits)
        valid = torch.zeros_like(logits, dtype=torch.bool)

        loss = FableLoss()(logits, target, valid_mask=valid)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.all(logits.grad == 0))

    def test_average_precision_is_exact_for_perfect_valid_ranking(self):
        probs = torch.tensor([0.95, 0.90, 0.20, 0.10, 0.99])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        valid = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)

        result = average_precision(probs, target, valid_mask=valid)

        self.assertEqual(result["valid_pixels"], 4)
        self.assertEqual(result["positive_pixels"], 2)
        self.assertAlmostEqual(result["average_precision"], 1.0)

    def test_average_precision_groups_tied_scores(self):
        probs = torch.full((5,), 0.5)
        target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])

        result = average_precision(probs, target)

        self.assertAlmostEqual(result["average_precision"], 2.0 / 5.0)

    def test_threshold_sweep_selects_f1_and_reports_fp_density(self):
        probs = torch.tensor([0.90, 0.70, 0.60, 0.20])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])

        result = threshold_sweep_metrics(
            probs,
            target,
            thresholds=[0.5, 0.8],
        )

        self.assertEqual(result["threshold"], 0.5)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 1)
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertEqual(result["fp_per_mpix"], 250_000.0)


if __name__ == "__main__":
    unittest.main()
