#!/usr/bin/env python3
from __future__ import annotations

import torch


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"mps built: {torch.backends.mps.is_built()}")
    print(f"mps available: {torch.backends.mps.is_available()}")

    device_count = "n/a"
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "device_count"):
            device_count = torch.mps.device_count()
    except Exception as exc:  # pragma: no cover
        device_count = f"error: {exc}"

    print(f"mps device_count: {device_count}")

    if torch.backends.mps.is_available():
        try:
            x = torch.randn(1, device="mps")
            print(f"tensor device: {x.device}")
        except Exception as exc:
            print(f"failed to create mps tensor: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
