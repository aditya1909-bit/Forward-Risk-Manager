from __future__ import annotations

import contextlib

import torch
from torch.optim import Adam

from frisk.device import sync_device


def parse_amp_dtype(value) -> torch.dtype:
    name = str(value).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def build_optimizer(params, lr: float, device: torch.device, use_fused: bool):
    params = tuple(params)
    if not params:
        raise ValueError("optimizer got an empty parameter list")
    kwargs = {}
    if device.type == "cuda":
        kwargs["foreach"] = True
        if use_fused:
            kwargs["fused"] = True
    try:
        return Adam(params, lr=lr, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
    try:
        return Adam(params, lr=lr, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("foreach", None)
        return Adam(params, lr=lr, **kwargs)


def make_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_if_needed(enabled: bool, dtype: torch.dtype):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def state_dict_for_save(module: torch.nn.Module):
    inner = getattr(module, "_orig_mod", module)
    return inner.state_dict()


def forward_encoder(model, *args, **kwargs):
    compiler_ns = getattr(torch, "compiler", None)
    mark_step = (
        getattr(compiler_ns, "cudagraph_mark_step_begin", None) if compiler_ns is not None else None
    )
    if callable(mark_step):
        mark_step()
    return model(*args, **kwargs)


def compile_mode_candidates(requested_mode: str, device: torch.device) -> list[str]:
    requested = str(requested_mode).strip() or "default"
    candidates: list[str] = []
    if device.type == "cuda" and requested == "reduce-overhead":
        candidates.append("max-autotune-no-cudagraphs")
    candidates.append(requested)
    if "default" not in candidates:
        candidates.append("default")
    seen: set[str] = set()
    return [mode for mode in candidates if not (mode in seen or seen.add(mode))]


def maybe_compile_encoder(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    context: str,
) -> torch.nn.Module:
    if not bool(config.get("torch_compile", False)):
        return model
    if not hasattr(torch, "compile"):
        config["torch_compile"] = False
        print(f"warning: {context} torch.compile requested but unavailable; disabled.")
        return model
    requested_mode = str(config.get("torch_compile_mode", "max-autotune-no-cudagraphs"))
    for mode in compile_mode_candidates(requested_mode, device):
        try:
            model = torch.compile(model, mode=mode)
            config["torch_compile_mode"] = mode
            print(f"{context} torch_compile active (mode={mode})")
            return model
        except Exception as exc:
            print(f"warning: {context} torch.compile failed (mode={mode}): {exc}")
    config["torch_compile"] = False
    print(f"warning: {context} torch.compile disabled after fallback attempts.")
    return model


def optimizer_step(
    optim: torch.optim.Optimizer,
    loss: torch.Tensor,
    grad_clip: float,
    clip_params,
    scaler,
) -> None:
    optim.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
        scaler.step(optim)
        scaler.update()
        return
    loss.backward()
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
    optim.step()


def sync(device: torch.device) -> None:
    sync_device(device)


def reset_peak_cuda_memory(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def peak_cuda_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    except Exception:
        return float("nan")
