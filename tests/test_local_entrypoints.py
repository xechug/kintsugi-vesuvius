import unittest
from pathlib import Path

import torch

from vesuvius_code import local_mps_train, train_overfit


class LocalEntrypointTests(unittest.TestCase):
    def test_local_training_defaults_to_audited_fragment_config(self):
        args = local_mps_train.build_parser().parse_args([])

        self.assertNotIn("sample_data", args.config)
        self.assertTrue(Path(args.config).is_file())
        self.assertGreater(args.steps, 0)

    def test_overfit_diagnostic_defaults_to_audited_data(self):
        args = train_overfit.build_parser().parse_args([])

        self.assertNotIn("sample_data", args.config)
        self.assertTrue(Path(args.config).is_file())
        self.assertGreater(args.iterations, 0)

    def test_overfit_diagnostic_selects_an_informative_mixed_patch(self):
        class FakeDataset:
            def __init__(self):
                self.rates = [0.9, 0.22, 0.08, 0.0]

            def __getitem__(self, index):
                rate = self.rates[index]
                positives = int(rate * 100)
                target = torch.zeros(1, 10, 10)
                target.view(-1)[:positives] = 1
                return {
                    "target": target,
                    "valid_mask": torch.ones_like(target, dtype=torch.bool),
                    "coords": torch.tensor([0, index, index]),
                }

        sample, rate = train_overfit.choose_informative_sample(
            FakeDataset(),
            candidates=4,
            target_positive_rate=0.1,
        )

        self.assertEqual(sample["coords"].tolist(), [0, 2, 2])
        self.assertAlmostEqual(rate, 0.08, places=5)


if __name__ == "__main__":
    unittest.main()
