import unittest

import numpy as np

from vesuvius_code.numpy_metrics import binary_report, precision_recall_summary, roc_auc


class NumpyRankingMetricTests(unittest.TestCase):
    def test_constant_predictions_are_chance_for_either_label_order(self):
        for target in ([0, 0, 1, 1], [1, 1, 0, 0]):
            self.assertEqual(roc_auc(target, [0.5] * 4), 0.5)
            ap, f1, threshold = precision_recall_summary(target, [0.5] * 4)
            self.assertEqual(ap, 0.5)
            self.assertAlmostEqual(f1, 2 / 3)
            self.assertEqual(threshold, 0.5)

    def test_tied_scores_match_pairwise_auc_and_are_permutation_invariant(self):
        y = np.array([1, 0, 1, 0, 0, 1])
        s = np.array([0.8, 0.8, 0.4, 0.1, 0.4, 0.8])
        pos, neg = s[y == 1], s[y == 0]
        reference = ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / 9
        self.assertAlmostEqual(roc_auc(y, s), reference)
        before = precision_recall_summary(y, s)
        for seed in range(5):
            order = np.random.default_rng(seed).permutation(len(y))
            self.assertEqual(precision_recall_summary(y[order], s[order]), before)

    def test_frozen_threshold_is_distinct_from_test_best_threshold(self):
        result = binary_report([0, 1, 1], [0.1, 0.3, 0.4], threshold=0.5)
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["best_f1_diagnostic"], 1.0)
        self.assertEqual(result["fp"], 0)

    def test_invalid_inputs_fail(self):
        for y, s in [([], []), ([1], [0.1, 0.2]), ([0, 1], [0.0, np.nan])]:
            with self.assertRaises(ValueError):
                roc_auc(y, s)

    def test_auc_is_undefined_for_one_class(self):
        self.assertTrue(np.isnan(roc_auc([1, 1], [0.1, 0.2])))


if __name__ == "__main__":
    unittest.main()
