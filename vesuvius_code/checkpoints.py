from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 2


class CheckpointCompatibilityError(RuntimeError):
    pass


def _json_payload(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.loads(json.dumps(value, sort_keys=True))


def _config_sha256(run_config: dict) -> str:
    encoded = json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    run_config: dict,
    data_audit,
    epoch: int,
    global_step: int,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized_config = _json_payload(run_config)
    normalized_audit = _json_payload(data_audit)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "model_state": model.state_dict(),
        "run_config": normalized_config,
        "run_config_sha256": _config_sha256(normalized_config),
        "data_audit": normalized_audit,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "torch_version": str(torch.__version__),
    }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _read_payload(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint schema; expected version {SCHEMA_VERSION}"
        )
    return payload


def read_checkpoint_metadata(path: str | os.PathLike[str], map_location="cpu") -> dict:
    payload = _read_payload(path, map_location=map_location)
    state_fields = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
    }
    return {key: value for key, value in payload.items() if key not in state_fields}


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    expected_config: dict | None = None,
    strict: bool = True,
    restore_rng: bool = False,
    map_location="cpu",
) -> dict:
    if not strict:
        raise CheckpointCompatibilityError("partial model loading is disabled")
    payload = _read_payload(path, map_location=map_location)

    required = {"model_state", "model_class", "run_config", "rng_state", "epoch", "global_step"}
    missing = sorted(required.difference(payload))
    if missing:
        raise CheckpointCompatibilityError(f"checkpoint schema is missing fields: {missing}")

    if expected_config is not None:
        normalized_expected = _json_payload(expected_config)
        if normalized_expected != payload["run_config"]:
            raise CheckpointCompatibilityError("checkpoint configuration does not match expected configuration")

    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"incompatible model state: {error}") from error

    if optimizer is not None:
        if payload.get("optimizer_state") is None:
            raise CheckpointCompatibilityError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload.get("scheduler_state") is None:
            raise CheckpointCompatibilityError("checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None:
        if payload.get("scaler_state") is None:
            raise CheckpointCompatibilityError("checkpoint has no scaler state")
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


def export_lightning_checkpoint(
    lightning_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    run_config: dict,
    data_audit,
    model_prefix: str = "model.",
) -> Path:
    """Export the model portion of a Lightning checkpoint to strict schema v2."""

    payload = torch.load(lightning_path, map_location="cpu", weights_only=True)
    lightning_state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(lightning_state, dict):
        raise CheckpointCompatibilityError("Lightning checkpoint has no state_dict")
    model_state = {
        name[len(model_prefix) :]: tensor
        for name, tensor in lightning_state.items()
        if name.startswith(model_prefix)
    }
    expected_keys = set(model.state_dict())
    actual_keys = set(model_state)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        unexpected = sorted(actual_keys.difference(expected_keys))
        raise CheckpointCompatibilityError(
            f"Lightning model state mismatch; missing={missing}, unexpected={unexpected}"
        )
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(
            f"incompatible Lightning model state: {error}"
        ) from error
    return save_checkpoint(
        destination,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        run_config=run_config,
        data_audit=data_audit,
        epoch=int(payload.get("epoch", 0)),
        global_step=int(payload.get("global_step", 0)),
    )
