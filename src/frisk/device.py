from __future__ import annotations

import torch


def _mps_built() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def resolve_device(device: str | None = "auto") -> torch.device:
    requested = str(device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if requested == "mps":
        if not _mps_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {device}. Expected one of auto,cpu,cuda,mps")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and _mps_available():
        torch.mps.synchronize()


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and _mps_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def collect_device_diagnostics() -> dict[str, str]:
    info = {
        "torch": str(torch.__version__),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "mps_built": str(_mps_built()),
        "mps_available": str(_mps_available()),
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device_name"] = str(torch.cuda.get_device_name(0))
        except Exception as exc:
            info["cuda_device_name"] = f"error: {exc}"
    return info
