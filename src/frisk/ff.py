from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

def goodness(h: torch.Tensor, batch: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    h2 = h * h
    # per-node energy
    node_energy = h2.mean(dim=1)
    # log-sum-exp per graph (smooth max)
    g_list = []
    for gid in batch.unique():
        idx = (batch == gid).nonzero(as_tuple=False).view(-1)
        e = node_energy[idx]
        g = temperature * torch.logsumexp(e / temperature, dim=0)
        g_list.append(g)
    return torch.stack(g_list, dim=0)


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
    graph_ids = batch.unique()

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

    for gid in graph_ids:
        idx = (batch == gid).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        if mode in ("time_flip", "shuffle+time_flip", "time_flip+noise"):
            out[idx] = _time_flip(out[idx])
        if mode in ("shuffle", "shuffle+noise", "shuffle+time_flip"):
            perm = torch.randperm(idx.numel(), device=x.device)
            out[idx] = out[idx][perm]
        if mode in ("noise", "shuffle+noise", "time_flip+noise") and noise_std > 0:
            out[idx] = out[idx] + noise_std * torch.randn_like(out[idx])
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
