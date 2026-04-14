from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a normal dependency, but keep fallback simple.
    np = None


def resume_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def module_state_for_resume(module: torch.nn.Module) -> dict[str, Any]:
    inner = getattr(module, "_orig_mod", module)
    return inner.state_dict()


def _normalize_state_dict(state) -> dict[str, Any]:
    if isinstance(state, dict):
        if isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        elif isinstance(state.get("model"), dict):
            state = state["model"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint state_dict payload must be a mapping")
    out: dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("_orig_mod.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        out[name] = value
    return out


def load_module_state_for_resume(module: torch.nn.Module, state, *, strict: bool = True):
    inner = getattr(module, "_orig_mod", module)
    return inner.load_state_dict(_normalize_state_dict(state), strict=strict)


def capture_rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if np is not None:
        payload["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        try:
            payload["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return payload


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_resume_payload(
    path: str | Path | None,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    if not path:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    if expected_fingerprint is not None and str(payload.get("fingerprint", "")) != str(expected_fingerprint):
        return None
    return payload


def save_resume_payload(
    path: str | Path,
    *,
    fingerprint: str,
    status: str,
    epoch_completed: int,
    model_states: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    scaler_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    epoch_history: list[dict[str, Any]] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    payload = {
        "fingerprint": str(fingerprint),
        "status": str(status),
        "epoch_completed": int(epoch_completed),
        "model_states": dict(model_states or {}),
        "optimizer_state": optimizer_state,
        "scaler_state": scaler_state,
        "metadata": dict(metadata or {}),
        "result": dict(result or {}) if result is not None else None,
        "epoch_history": list(epoch_history or []),
        "rng_state": dict(rng_state or {}),
    }
    _atomic_torch_save(checkpoint_path, payload)
    return checkpoint_path
