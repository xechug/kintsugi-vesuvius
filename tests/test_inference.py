import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch


def load_api(testcase):
    spec = importlib.util.find_spec("vesuvius_code.inference")
    testcase.assertIsNotNone(spec, "vesuvius_code.inference must exist")
    return importlib.import_module("vesuvius_code.inference")


class ConstantProbabilityModel(torch.nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = float(probability)

    def predict_proba(self, batch):
        batch_size, _, _, height, width = batch.shape
        return torch.full(
            (batch_size, 1, height, width),
            self.probability,
            device=batch.device,
            dtype=batch.dtype,
        )


class MeanDepthModel(torch.nn.Module):
    def predict_proba(self, batch):
        return batch.mean(dim=2)


class InferenceTests(unittest.TestCase):
    def test_normalization_divisor_uses_actual_dtype(self):
        api = load_api(self)

        self.assertEqual(api.normalization_divisor(np.dtype("uint8")), 255.0)
        self.assertEqual(api.normalization_divisor(np.dtype("uint16")), 65535.0)
        self.assertEqual(api.normalization_divisor(np.dtype("float32")), 1.0)

    def test_sliding_origins_include_final_border(self):
        api = load_api(self)

        origins = api.sliding_origins(length=100, patch=32, stride=20)

        self.assertEqual(origins, [0, 20, 40, 60, 68])

    def test_hann_blending_reconstructs_constant_prediction_without_seams(self):
        api = load_api(self)
        volume = np.zeros((4, 20, 20), dtype=np.uint8)
        model = ConstantProbabilityModel(0.25)

        result = api.predict_region(
            model,
            volume,
            region=(0, 0, 20, 20),
            patch_size=(4, 8, 8),
            stride=(4, 4),
            device="cpu",
            batch_size=3,
        )

        np.testing.assert_allclose(result, 0.25, atol=1e-6)

    def test_predict_region_normalizes_uint8_before_model(self):
        api = load_api(self)
        volume = np.full((2, 8, 8), 255, dtype=np.uint8)

        result = api.predict_region(
            MeanDepthModel(),
            volume,
            region=(0, 0, 8, 8),
            patch_size=(2, 8, 8),
            stride=(8, 8),
            device="cpu",
        )

        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_probability_tiff_remains_float_without_contrast_stretch(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.tif"
            probabilities = np.array([[0.02, 0.04], [0.06, 0.08]], dtype=np.float32)

            api.save_probability_tiff(path, probabilities, metadata={"checkpoint": "demo"})
            restored = tifffile.imread(path)

            self.assertEqual(restored.dtype, np.float32)
            np.testing.assert_array_equal(restored, probabilities)

    def test_legacy_checkpoint_is_rejected_before_inference(self):
        api = load_api(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pth"
            torch.save({"model": torch.nn.Linear(1, 1).state_dict()}, path)

            with self.assertRaisesRegex(Exception, "schema"):
                api.load_kintsugi_model(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
