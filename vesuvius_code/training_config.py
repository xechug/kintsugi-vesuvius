from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    gpu_model: str = "unknown"
    devices: int = 1
    precision: str = "32-true"
    batch_size_per_device: int = 1
    accumulate_grad_batches: int = 1
    workers_per_device: int = 2
    prefetch_factor: int = 2
    patch_size: tuple[int, int, int] = (64, 128, 128)
    cpu_ram_budget_gib: float = 32.0
    validation_regions: int = 0

    @property
    def effective_global_batch_size(self) -> int:
        return (
            int(self.devices)
            * int(self.batch_size_per_device)
            * int(self.accumulate_grad_batches)
        )

    @property
    def estimated_prefetch_gib(self) -> float:
        depth, height, width = (int(value) for value in self.patch_size)
        image_bytes = depth * height * width * 4
        target_and_valid_bytes = height * width * 5
        per_sample = image_bytes + target_and_valid_bytes
        queued_batches = (
            int(self.devices)
            * int(self.workers_per_device)
            * int(self.prefetch_factor)
        )
        estimated_bytes = per_sample * int(self.batch_size_per_device) * queued_batches
        return estimated_bytes / float(1024**3)

    def validate(self) -> None:
        positive_integer_fields = {
            "devices": self.devices,
            "batch_size_per_device": self.batch_size_per_device,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "prefetch_factor": self.prefetch_factor,
        }
        for name, value in positive_integer_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.workers_per_device) < 0:
            raise ValueError("workers_per_device cannot be negative")
        if len(self.patch_size) != 3 or any(int(value) <= 0 for value in self.patch_size):
            raise ValueError("patch_size must contain three positive values")
        if int(self.validation_regions) <= 0:
            raise ValueError("at least one independent validation region is required")
        if self.precision not in {"32-true", "16-mixed", "bf16-mixed"}:
            raise ValueError(f"unsupported precision: {self.precision}")

        gpu_name = self.gpu_model.upper()
        unsupported_bf16 = ("T4", "P100", "V100")
        if self.precision == "bf16-mixed" and any(name in gpu_name for name in unsupported_bf16):
            matched = next(name for name in unsupported_bf16 if name in gpu_name)
            raise ValueError(f"BF16 is not supported efficiently by {matched}; use 16-mixed")
        if not math.isfinite(float(self.cpu_ram_budget_gib)) or self.cpu_ram_budget_gib <= 0:
            raise ValueError("cpu_ram_budget_gib must be positive and finite")
        if self.estimated_prefetch_gib > float(self.cpu_ram_budget_gib):
            raise ValueError(
                f"prefetch estimate {self.estimated_prefetch_gib:.2f} GiB exceeds "
                f"CPU RAM budget {self.cpu_ram_budget_gib:.2f} GiB"
            )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["patch_size"] = list(self.patch_size)
        payload["effective_global_batch_size"] = self.effective_global_batch_size
        payload["estimated_prefetch_gib"] = self.estimated_prefetch_gib
        return payload
