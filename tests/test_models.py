import importlib
import importlib.util
import unittest

import torch

from vesuvius_code.model import DepthSequenceMixer, DualStreamVMambaForense, init_weights


def load_baselines(testcase):
    spec = importlib.util.find_spec("vesuvius_code.baselines")
    testcase.assertIsNotNone(spec, "vesuvius_code.baselines must exist")
    return importlib.import_module("vesuvius_code.baselines")


class ModelContractTests(unittest.TestCase):
    def test_explicit_mamba_backend_never_silently_falls_back(self):
        with self.assertRaisesRegex((ImportError, RuntimeError, ModuleNotFoundError), "mamba"):
            DepthSequenceMixer(d_model=4, backend="mamba")

    def test_initializer_preserves_parameters_marked_no_reinit(self):
        linear = torch.nn.Linear(4, 4)
        with torch.no_grad():
            linear.bias.fill_(7.0)
        linear.bias._no_reinit = True

        init_weights(linear)

        self.assertTrue(torch.all(linear.bias == 7.0))

    def test_bidirectional_fallback_is_reversal_equivariant(self):
        mixer = DepthSequenceMixer(d_model=4, backend="fallback", bidirectional=True)
        sequence = torch.randn(2, 7, 4)

        forward = mixer(sequence)
        reversed_output = mixer(sequence.flip(1)).flip(1)

        torch.testing.assert_close(forward, reversed_output, atol=1e-5, rtol=1e-5)

    def test_model_configuration_records_actual_backend(self):
        model = DualStreamVMambaForense(
            in_channels=1,
            base_channels=4,
            depth_topk=2,
            mamba_backend="fallback",
            bidirectional=True,
        )

        config = model.get_config()

        self.assertEqual(config["backend"], "fallback")
        self.assertTrue(config["bidirectional"])
        self.assertEqual(config["base_channels"], 4)

    def test_residual_unet_baseline_returns_2d_logits_and_backpropagates(self):
        api = load_baselines(self)
        model = api.ResidualUNet3DTo2D(in_channels=1, base_channels=4)
        image = torch.randn(2, 1, 6, 17, 19)

        logits = model(image)
        logits.mean().backward()

        self.assertEqual(tuple(logits.shape), (2, 1, 17, 19))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_axial_kintsugi_mixes_all_axes_at_bottleneck(self):
        api = load_baselines(self)
        model = api.KintsugiAxialUNet3DTo2D(
            in_channels=1,
            base_channels=4,
            backend="fallback",
        )
        image = torch.randn(1, 1, 5, 16, 16)

        logits, auxiliary = model(image, return_aux=True)
        logits.square().mean().backward()

        self.assertEqual(tuple(logits.shape), (1, 1, 16, 16))
        self.assertEqual(auxiliary["mixed_axes"], ("z", "y", "x"))
        self.assertEqual(auxiliary["backend"], "fallback")
        self.assertTrue(torch.isfinite(logits).all())

    def test_model_factory_round_trips_each_supported_configuration(self):
        from vesuvius_code.model_factory import build_model_from_config, create_model

        for name in (
            "residual_unet",
            "residual_physical",
            "kintsugi_axial",
            "kintsugi_physical",
            "legacy_kintsugi",
        ):
            with self.subTest(name=name):
                original = create_model(
                    name,
                    in_channels=1,
                    num_classes=1,
                    base_channels=4,
                    backend="fallback",
                )
                restored = build_model_from_config(original.get_config())
                logits = restored(torch.randn(1, 1, 4, 8, 8))

                self.assertIs(type(restored), type(original))
                self.assertEqual(tuple(logits.shape), (1, 1, 8, 8))

    def test_physical_stem_is_invariant_to_affine_intensity_shifts(self):
        api = load_baselines(self)
        stem = api.PhysicalStem3D(in_channels=1).eval()
        volume = torch.randn(2, 1, 8, 16, 16)
        shifted_and_scaled = volume * 3.7 + 42.0

        out_original = stem(volume)
        out_transformed = stem(shifted_and_scaled)

        self.assertEqual(tuple(out_original.shape), (2, 4, 8, 16, 16))
        torch.testing.assert_close(out_original, out_transformed, atol=1e-4, rtol=1e-4)

    def test_soft_sheet_unwarper_starts_as_identity_and_backpropagates(self):
        api = load_baselines(self)
        unwarper = api.SoftSheetUnwarper3D(channels=4, max_shift=2.0)
        features = torch.randn(1, 4, 6, 12, 12, requires_grad=True)

        warped, shift = unwarper(features)
        torch.testing.assert_close(warped, features, atol=1e-6, rtol=1e-6)
        self.assertEqual(tuple(shift.shape), (1, 1, 12, 12))
        self.assertTrue(torch.all(shift == 0))

        target = torch.randn_like(warped)
        loss = (warped - target).square().mean()
        loss.backward()

        self.assertIsNotNone(unwarper.offset_head.weight.grad)
        self.assertGreater(
            float(unwarper.offset_head.weight.grad.abs().sum()),
            0.0,
        )

    def test_kintsugi_physical_exposes_sheet_shift_in_auxiliary_output(self):
        api = load_baselines(self)
        model = api.KintsugiPhysicalUNet3DTo2D(
            in_channels=1,
            base_channels=4,
            backend="fallback",
        )
        image = torch.randn(1, 1, 6, 16, 16)

        logits, aux = model(image, return_aux=True)
        logits.mean().backward()

        self.assertEqual(tuple(logits.shape), (1, 1, 16, 16))
        self.assertIn("sheet_shift_z", aux)
        self.assertEqual(tuple(aux["sheet_shift_z"].shape), (1, 1, 16, 16))

    def test_synthetic_carbonized_ink_and_fourth_augment_preserve_bounds(self):
        import numpy as np
        from vesuvius_code.dataset import (
            apply_fourth_augment_z,
            inject_synthetic_carbonized_ink,
        )

        rng = np.random.default_rng(1337)
        vol = np.full((12, 32, 32), 0.2, dtype=np.float32)
        vol[5:] = 0.6
        ink = np.zeros((32, 32), dtype=np.float32)
        valid = np.ones((32, 32), dtype=np.uint8)

        new_vol, new_ink = inject_synthetic_carbonized_ink(
            vol, ink, valid, rng, strength=0.10
        )
        self.assertEqual(new_vol.shape, vol.shape)
        self.assertGreater(float(new_ink.mean()), 0.02)
        self.assertGreater(float((new_vol - vol).max()), 0.0)

        warped_vol = apply_fourth_augment_z(new_vol, rng, max_shift=2, edge_crop_prob=1.0)
        self.assertEqual(warped_vol.shape, vol.shape)
        self.assertTrue(np.all((warped_vol >= 0.0) & (warped_vol <= 1.0)))

    def test_anisotropic_fiber_suppression_loss_penalizes_fiber_aligned_false_positives(self):
        from vesuvius_code.losses import AnisotropicFiberSuppressionLoss

        criterion = AnisotropicFiberSuppressionLoss(fiber_weight=0.5)
        y = torch.linspace(-1.0, 1.0, 32).view(1, 1, 1, 32, 1).expand(1, 1, 8, 32, 32)
        striated_vol = torch.sin(y * 12.0) * 0.5 + 0.5
        target = torch.zeros((1, 1, 32, 32))
        valid = torch.ones((1, 1, 32, 32))
        logits = torch.ones((1, 1, 32, 32), requires_grad=True)

        loss_with_fiber = criterion(logits, target, valid_mask=valid, volume=striated_vol)
        loss_without_fiber = criterion(logits, target, valid_mask=valid, volume=None)
        loss_with_fiber.backward()

        self.assertGreater(float(loss_with_fiber), float(loss_without_fiber))
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()


