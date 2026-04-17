from __future__ import annotations

import contextlib
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from frisk.device import resolve_device


@dataclass
class SignalCheckpointController:
    enabled: bool
    _requested: bool = False

    def mark_requested(self, *_args) -> None:
        self._requested = True

    @property
    def requested(self) -> bool:
        return bool(self._requested)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def init_distributed_context(
    *,
    requested_device: str | None,
    distributed: bool = False,
    backend: str = "nccl",
    local_rank: int | None = None,
) -> DistributedContext:
    env_world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    enabled = bool(distributed or env_world > 1)
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    env_rank = int(os.environ.get("RANK", "0") or "0")
    use_local_rank = env_local_rank if local_rank is None else int(local_rank)
    backend_name = str(backend or "nccl")

    if enabled:
        if requested_device in {None, "", "auto"}:
            device = torch.device("cuda", use_local_rank) if torch.cuda.is_available() else torch.device("cpu")
        elif str(requested_device).strip().lower() == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda", use_local_rank)
        else:
            device = resolve_device(requested_device)
        if device.type == "cuda":
            torch.cuda.set_device(use_local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend_name)
        return DistributedContext(
            enabled=True,
            rank=env_rank,
            world_size=env_world,
            local_rank=use_local_rank,
            backend=backend_name,
            device=device,
        )

    return DistributedContext(
        enabled=False,
        rank=0,
        world_size=1,
        local_rank=0,
        backend=backend_name,
        device=resolve_device(requested_device or "auto"),
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_wrap_ddp(module: torch.nn.Module, ctx: DistributedContext) -> torch.nn.Module:
    if not ctx.enabled:
        return module
    kwargs: dict[str, Any] = {}
    if ctx.device.type == "cuda":
        kwargs["device_ids"] = [ctx.local_rank]
        kwargs["output_device"] = ctx.local_rank
    return DistributedDataParallel(module, **kwargs)


def unwrap_module(module: torch.nn.Module | None) -> torch.nn.Module | None:
    if module is None:
        return None
    if isinstance(module, DistributedDataParallel):
        return module.module
    return getattr(module, "module", module)


def maybe_make_sampler(
    dataset,
    ctx: DistributedContext,
    *,
    shuffle: bool,
    drop_last: bool = False,
) -> DistributedSampler | None:
    if not ctx.enabled:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def dataloader_kwargs_with_sampler(
    kwargs: dict[str, Any],
    sampler: DistributedSampler | None,
) -> dict[str, Any]:
    out = dict(kwargs)
    if sampler is None:
        return out
    out.pop("shuffle", None)
    out["sampler"] = sampler
    return out


def set_sampler_epoch(sampler: DistributedSampler | None, epoch: int) -> None:
    if sampler is not None:
        sampler.set_epoch(int(epoch))


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float | int, ctx: DistributedContext) -> float:
    tensor = torch.tensor(float(value), device=ctx.device)
    if ctx.enabled and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(ctx.world_size)
    return float(tensor.item())


def reduce_sum(value: float | int, ctx: DistributedContext) -> float:
    tensor = torch.tensor(float(value), device=ctx.device)
    if ctx.enabled and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def gather_objects(value: Any, ctx: DistributedContext) -> list[Any]:
    if not ctx.enabled or not dist.is_initialized():
        return [value]
    gathered: list[Any] = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def broadcast_object(value: Any, ctx: DistributedContext, *, src: int = 0) -> Any:
    if not ctx.enabled or not dist.is_initialized():
        return value
    payload = [value if ctx.rank == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def install_signal_checkpoint_controller(
    *,
    enabled: bool,
    signals: Iterable[int] = (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT),
) -> SignalCheckpointController:
    controller = SignalCheckpointController(enabled=bool(enabled))
    if not controller.enabled:
        return controller
    for sig in signals:
        with contextlib.suppress(ValueError, OSError, RuntimeError):
            signal.signal(sig, controller.mark_requested)
    return controller


def checkpoint_due(
    *,
    last_saved_at: float | None,
    every_minutes: float | int | None,
    signal_controller: SignalCheckpointController | None = None,
) -> bool:
    if signal_controller is not None and signal_controller.requested:
        return True
    if every_minutes is None:
        return False
    interval_s = float(every_minutes) * 60.0
    if interval_s <= 0:
        return False
    if last_saved_at is None:
        return True
    return (time.time() - float(last_saved_at)) >= interval_s


def rank_zero_write_text(path: str | Path, text: str, ctx: DistributedContext) -> None:
    if not ctx.is_primary:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
