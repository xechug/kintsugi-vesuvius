import unittest

import torch
from torch import nn

from vesuvius_code.reader_correction import ReaderCorrection


class TinyRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(1, 1, 1)

    def forward(self, volume):
        return self.layer(volume.mean(2))


class ReaderCorrectionTests(unittest.TestCase):
    def test_initial_model_preserves_reader_and_never_backpropagates_into_it(self):
        model = ReaderCorrection(TinyRefiner())
        volume = torch.rand(2, 1, 17, 8, 8)
        scores = torch.rand(2, 1, 8, 8, requires_grad=True) * 0.5 + 0.25
        scores.retain_grad()
        output = model(volume, scores)
        torch.testing.assert_close(output.sigmoid(), scores)
        output.sum().backward()
        self.assertIsNone(scores.grad)
        self.assertGreater(float(model.correction_head.weight.grad.abs().sum()), 0)

    def test_learned_correction_is_bounded_and_reaches_physical_branch(self):
        model = ReaderCorrection(TinyRefiner(), max_delta=4)
        with torch.no_grad():
            model.correction_head.weight.fill_(0.1)
        scores = torch.full((1, 1, 8, 8), 0.3)
        volume = torch.rand(1, 1, 17, 8, 8)
        output = model(volume, scores)
        self.assertLessEqual(float((output.detach() - torch.logit(scores)).abs().max()), 4)
        output.sum().backward()
        self.assertGreater(float(model.refiner.layer.weight.grad.abs().sum()), 0)

    def test_grid_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pixel grid"):
            ReaderCorrection(TinyRefiner())(torch.zeros(1, 1, 17, 8, 8), torch.zeros(1, 1, 7, 8))


if __name__ == "__main__":
    unittest.main()
