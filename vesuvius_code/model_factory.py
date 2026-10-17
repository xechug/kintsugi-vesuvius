from __future__ import annotations

from .baselines import (
    KintsugiAxialUNet3DTo2D,
    KintsugiPhysicalUNet3DTo2D,
    ResidualPhysicalUNet3DTo2D,
    ResidualUNet3DTo2D,
)
from .model import DualStreamVMambaForense


MODEL_ALIASES = {
    "residual_unet": "ResidualUNet3DTo2D",
    "residual_physical": "ResidualPhysicalUNet3DTo2D",
    "kintsugi_axial": "KintsugiAxialUNet3DTo2D",
    "kintsugi_physical": "KintsugiPhysicalUNet3DTo2D",
    "legacy_kintsugi": "DualStreamVMambaForense",
}


def create_model(
    name,
    *,
    in_channels=1,
    num_classes=1,
    base_channels=8,
    backend="auto",
    bidirectional=True,
    depth_topk=4,
    depth_aggregation="topk_mean",
    attention_strength=0.5,
    physical_stem=False,
    sheet_unwarp=False,
):
    canonical = MODEL_ALIASES.get(name, name)
    common = {
        "in_channels": int(in_channels),
        "num_classes": int(num_classes),
        "base_channels": int(base_channels),
    }
    if canonical == "ResidualUNet3DTo2D":
        return ResidualUNet3DTo2D(
            **common,
            physical_stem=physical_stem,
            sheet_unwarp=sheet_unwarp,
        )
    if canonical == "ResidualPhysicalUNet3DTo2D":
        return ResidualPhysicalUNet3DTo2D(
            **common,
            physical_stem=True,
            sheet_unwarp=True,
        )
    if canonical == "KintsugiAxialUNet3DTo2D":
        return KintsugiAxialUNet3DTo2D(
            **common,
            backend=backend,
            bidirectional=bidirectional,
            physical_stem=physical_stem,
            sheet_unwarp=sheet_unwarp,
        )
    if canonical == "KintsugiPhysicalUNet3DTo2D":
        return KintsugiPhysicalUNet3DTo2D(
            **common,
            backend=backend,
            bidirectional=bidirectional,
            physical_stem=True,
            sheet_unwarp=True,
        )
    if canonical == "DualStreamVMambaForense":
        return DualStreamVMambaForense(
            **common,
            depth_topk=depth_topk,
            depth_aggregation=depth_aggregation,
            attention_strength=attention_strength,
            mamba_backend=backend,
            bidirectional=bidirectional,
        )
    raise ValueError(f"Unknown model name: {name}")


def build_model_from_config(config):
    return create_model(
        config["name"],
        in_channels=config.get("in_channels", 1),
        num_classes=config.get("num_classes", 1),
        base_channels=config["base_channels"],
        backend=config.get("backend", "fallback"),
        bidirectional=config.get("bidirectional", True),
        depth_topk=config.get("depth_topk", 4),
        depth_aggregation=config.get("depth_aggregation", "topk_mean"),
        attention_strength=config.get("attention_strength", 0.5),
        physical_stem=config.get("physical_stem", False),
        sheet_unwarp=config.get("sheet_unwarp", False),
    )

