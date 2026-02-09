from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def _segment_logsumexp(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    max_vals = torch.full((num_segments,), -torch.inf, device=values.device, dtype=values.dtype)
    if hasattr(max_vals, "scatter_reduce_"):
        max_vals.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    else:
        for gid in range(num_segments):
            mask = segment_ids == gid
            if mask.any():
                max_vals[gid] = values[mask].max()
    shifted = torch.exp(values - max_vals.index_select(0, segment_ids))
    sum_vals = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    sum_vals.index_add_(0, segment_ids, shifted)
    return max_vals + torch.log(sum_vals.clamp_min(1e-12))


def goodness(h: torch.Tensor, batch: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if h.numel() == 0:
        return torch.empty(0, device=h.device, dtype=h.dtype)

    node_energy = (h * h).mean(dim=1)
    _, segment_ids = torch.unique(batch, sorted=True, return_inverse=True)
    scaled = node_energy / temperature
    lse = _segment_logsumexp(scaled, segment_ids, int(segment_ids.max().item()) + 1)
    return temperature * lse


def make_negative(
    x: torch.Tensor,
    batch: torch.Tensor,
    mode: Literal[
        "shuffle",
        "noise",
        "shuffle+noise",
        "time_flip",
        "shuffle+time_flip",
        "time_flip+noise",
    ] = "shuffle",
    noise_std: float = 0.05,
    window_len: int | None = None,
    summary_dim: int = 0,
) -> torch.Tensor:
    out = x.clone()
    if out.numel() == 0:
        return out
    batch_idx = batch if batch.device == out.device else batch.to(out.device)

    def _time_flip(tensor: torch.Tensor) -> torch.Tensor:
        if window_len is None:
            return torch.flip(tensor, dims=[1])
        w = tensor[:, :window_len]
        flipped = torch.flip(w, dims=[1])
        if summary_dim > 0:
            s = tensor[:, window_len : window_len + summary_dim]
            rest = tensor[:, window_len + summary_dim :]
            if rest.numel() == 0:
                return torch.cat([flipped, s], dim=1)
            return torch.cat([flipped, s, rest], dim=1)
        return torch.cat([flipped, tensor[:, window_len:]], dim=1)

    if mode in ("time_flip", "shuffle+time_flip", "time_flip+noise"):
        out = _time_flip(out)

    if mode in ("shuffle", "shuffle+noise", "shuffle+time_flip"):
        # Group-preserving shuffle: each node is reassigned to another node in the same graph.
        rand = torch.rand(batch_idx.numel(), device=out.device, dtype=torch.float64)
        try:
            order = torch.argsort(batch_idx, stable=True)
            shuffled = torch.argsort(batch_idx.to(torch.float64) + rand, stable=True)
        except TypeError:
            order = torch.argsort(batch_idx)
            shuffled = torch.argsort(batch_idx.to(torch.float64) + rand)

        source_for_dest = torch.empty_like(order)
        source_for_dest[order] = shuffled
        out = out.index_select(0, source_for_dest)

    if mode in ("noise", "shuffle+noise", "time_flip+noise") and noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return out


def ff_loss(
    g_pos: torch.Tensor,
    g_neg: torch.Tensor,
    target: float = 1.0,
) -> torch.Tensor:
    # Encourage g_pos > target and g_neg < target
    loss_pos = F.softplus(target - g_pos)
    loss_neg = F.softplus(g_neg - target)
    return (loss_pos + loss_neg).mean()
