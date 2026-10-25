"""Train Kintsugi beside a frozen Reader score map, without changing Reader weights."""
from __future__ import annotations

import torch
from torch import nn

from .baselines import KintsugiPhysicalUNet3DTo2D, ResidualUNet3DTo2D


class ReaderCorrection(nn.Module):
    def __init__(self, refiner: nn.Module, max_delta: float = 4.0):
        super().__init__()
        self.refiner = refiner
        self.max_delta = float(max_delta)
        self.correction_head = nn.Conv2d(2, 1, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, volume, reader_scores):
        if reader_scores.ndim != 4 or reader_scores.shape[1] != 1:
            raise ValueError("Reader scores must have shape B,1,H,W")
        if volume.shape[0] != reader_scores.shape[0] or volume.shape[-2:] != reader_scores.shape[-2:]:
            raise ValueError("CT and Reader scores must share the same pixel grid")
        # Cached Reader scores are an input, never a target or a trainable teacher.
        base = torch.logit(reader_scores.detach().float().clamp(1e-5, 1 - 1e-5))
        features = self.refiner(volume)
        delta = self.max_delta * torch.tanh(self.correction_head(torch.cat((features, base), dim=1)))
        return base + delta


def build_correction_model(name, base_channels=16):
    physical = name != "reader_residual"
    if physical:
        branch = KintsugiPhysicalUNet3DTo2D(base_channels=base_channels, backend="fallback",
            physical_stem=True, sheet_unwarp=True, multi_scale_stem=True)
    else:
        branch = ResidualUNet3DTo2D(base_channels=base_channels)
    if name == "kintsugi_alone":
        return branch
    if name not in ("reader_residual", "reader_kintsugi", "reader_kintsugi_fiber"):
        raise ValueError(f"Unknown model: {name}")
    return ReaderCorrection(branch)
