import importlib
import importlib.util
import unittest


def load_api(testcase):
    spec = importlib.util.find_spec("vesuvius_code.training_config")
    testcase.assertIsNotNone(spec, "vesuvius_code.training_config must exist")
    return importlib.import_module("vesuvius_code.training_config")


class TrainingConfigTests(unittest.TestCase):
    def test_t4_rejects_bfloat16(self):
        api = load_api(self)
        config = api.TrainingConfig(
            gpu_model="NVIDIA T4",
            precision="bf16-mixed",
            validation_regions=1,
        )

        with self.assertRaisesRegex(ValueError, "BF16.*T4"):
            config.validate()

    def test_effective_global_batch_includes_devices_and_accumulation(self):
        api = load_api(self)
        config = api.TrainingConfig(
            devices=4,
            batch_size_per_device=2,
            accumulate_grad_batches=8,
            validation_regions=2,
        )

        self.assertEqual(config.effective_global_batch_size, 64)

    def test_prefetch_memory_estimate_is_bounded_by_cpu_budget(self):
        api = load_api(self)
        config = api.TrainingConfig(
            devices=4,
            batch_size_per_device=4,
            workers_per_device=16,
            prefetch_factor=8,
            patch_size=(256, 256, 256),
            cpu_ram_budget_gib=8.0,
            validation_regions=1,
        )

        self.assertGreater(config.estimated_prefetch_gib, 8.0)
        with self.assertRaisesRegex(ValueError, "prefetch.*RAM"):
            config.validate()

    def test_validation_regions_are_mandatory(self):
        api = load_api(self)
        config = api.TrainingConfig(validation_regions=0)

        with self.assertRaisesRegex(ValueError, "validation"):
            config.validate()

    def test_safe_single_gpu_configuration_validates(self):
        api = load_api(self)
        config = api.TrainingConfig(
            gpu_model="NVIDIA A100 80GB",
            devices=1,
            precision="bf16-mixed",
            batch_size_per_device=1,
            workers_per_device=2,
            prefetch_factor=2,
            patch_size=(64, 128, 128),
            cpu_ram_budget_gib=32.0,
            validation_regions=3,
        )

        config.validate()
        self.assertLess(config.estimated_prefetch_gib, 1.0)


if __name__ == "__main__":
    unittest.main()
