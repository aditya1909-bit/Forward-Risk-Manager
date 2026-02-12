from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .ff import goodness


@dataclass
class HallucinationConfig:
    steps: int = 10
    lr: float = 0.1
    l2_weight: float = 0.1
    mean_weight: float = 0.05
    std_weight: float = 0.05
    corr_weight: float = 1.0
    clamp_std: Optional[float] = 3.0
    goodness_temp: float = 1.0
    node_fraction: float = 1.0
    node_min: int = 1
    init_noise: float = 0.0
    return_slice_len: int = 0
    penalty_scope: str = "returns"  # "returns" or "all"
    corr_scope: str = "returns"  # "returns" or "all"
    freeze_non_return_features: bool = True
    corr_every_n_steps: int = 1
    corr_edge_fraction: float = 1.0
    corr_edge_min: int = 1


def _edge_corr_loss(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    return_slice_len: int = 0,
    edge_sample_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if edge_attr is None:
        return torch.tensor(0.0, device=x.device)
    if return_slice_len and return_slice_len > 0 and x.size(1) >= return_slice_len:
        x = x[:, :return_slice_len]
    if edge_attr.ndim == 2 and edge_attr.shape[1] == 1:
        w = edge_attr.squeeze(1)
    else:
        w = edge_attr

    src = edge_index[0]
    dst = edge_index[1]
    if edge_sample_idx is not None and edge_sample_idx.numel() > 0:
        src = src.index_select(0, edge_sample_idx)
        dst = dst.index_select(0, edge_sample_idx)
        w = w.index_select(0, edge_sample_idx)
    xi = x[src]
    xj = x[dst]

    xi = xi - xi.mean(dim=1, keepdim=True)
    xj = xj - xj.mean(dim=1, keepdim=True)
    xi = xi / (xi.std(dim=1, keepdim=True) + 1e-6)
    xj = xj / (xj.std(dim=1, keepdim=True) + 1e-6)

    corr = (xi * xj).mean(dim=1)
    return ((corr - w) ** 2).mean()


def hallucinate_negative(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    batch: torch.Tensor,
    config: HallucinationConfig,
    edge_weight: Optional[torch.Tensor] = None,
    forward_fn=None,
    constraint_fn=None,
    force_indices: Optional[list[int]] = None,
) -> torch.Tensor:
    # Freeze model params during hallucination steps
    req_grad = [p.requires_grad for p in model.parameters()]
    train_state = model.training
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        x0 = x.detach()
        return_slice_len = int(config.return_slice_len)
        use_return_scope = (
            return_slice_len > 0
            and x0.size(1) >= return_slice_len
            and str(config.penalty_scope).strip().lower() == "returns"
        )
        use_corr_return_scope = (
            return_slice_len > 0
            and x0.size(1) >= return_slice_len
            and str(config.corr_scope).strip().lower() == "returns"
        )

        if use_return_scope:
            x0_scope = x0[:, :return_slice_len]
        else:
            x0_scope = x0

        mean0 = x0_scope.mean()
        std0 = x0_scope.std() + 1e-6
        x_var = x0.clone()
        if config.init_noise and config.init_noise > 0:
            x_var = x_var + torch.randn_like(x0) * (config.init_noise * std0)
        x_var = x_var.detach().requires_grad_(True)
        opt = torch.optim.Adam([x_var], lr=config.lr)
        if config.clamp_std is not None:
            clamp_min = mean0 - config.clamp_std * std0
            clamp_max = mean0 + config.clamp_std * std0

        if config.node_fraction < 1.0:
            mask = torch.zeros(x0.size(0), device=x0.device, dtype=torch.bool)
            for gid in batch.unique():
                idx = (batch == gid).nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                k = max(config.node_min, int(idx.numel() * config.node_fraction))
                perm = torch.randperm(idx.numel(), device=x0.device)[:k]
                mask[idx[perm]] = True
        else:
            mask = torch.ones(x0.size(0), device=x0.device, dtype=torch.bool)
        if force_indices:
            mask[torch.tensor(force_indices, device=x0.device, dtype=torch.long)] = True
        mask = mask[:, None]

        corr_every = max(1, int(config.corr_every_n_steps))
        corr_edge_fraction = float(max(0.0, min(1.0, config.corr_edge_fraction)))
        corr_edge_min = max(1, int(config.corr_edge_min))
        num_edges = int(edge_index.size(1))

        for step_i in range(config.steps):
            if forward_fn is not None:
                h = forward_fn(x_var)
            else:
                h = model(x_var, edge_index, edge_weight=edge_weight)
            g = goodness(h, batch, temperature=config.goodness_temp).mean()

            if use_return_scope:
                x_var_scope = x_var[:, :return_slice_len]
            else:
                x_var_scope = x_var

            l2 = (x_var_scope - x0_scope).pow(2).mean()
            mean_pen = (x_var_scope.mean() - mean0).pow(2)
            std_pen = (x_var_scope.std() - std0).pow(2)
            if config.corr_weight > 0 and (step_i % corr_every == 0):
                edge_sample_idx = None
                if 0 < corr_edge_fraction < 1.0 and num_edges > 0:
                    sample_k = max(corr_edge_min, int(num_edges * corr_edge_fraction))
                    sample_k = min(sample_k, num_edges)
                    edge_sample_idx = torch.randperm(num_edges, device=x_var.device)[:sample_k]
                corr_pen = _edge_corr_loss(
                    x_var,
                    edge_index,
                    edge_attr,
                    return_slice_len=return_slice_len if use_corr_return_scope else 0,
                    edge_sample_idx=edge_sample_idx,
                )
            else:
                corr_pen = torch.zeros((), device=x_var.device, dtype=x_var.dtype)

            loss = (
                -g
                + config.l2_weight * l2
                + config.mean_weight * mean_pen
                + config.std_weight * std_pen
                + config.corr_weight * corr_pen
            )
            if constraint_fn is not None:
                loss = loss + constraint_fn(x_var)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if mask is not None:
                with torch.no_grad():
                    x_var.data = torch.where(mask, x_var.data, x0)

            if use_return_scope and config.freeze_non_return_features and x_var.size(1) > return_slice_len:
                with torch.no_grad():
                    x_var.data[:, return_slice_len:] = x0[:, return_slice_len:]

            if config.clamp_std is not None:
                if use_return_scope:
                    x_var.data[:, :return_slice_len].clamp_(clamp_min, clamp_max)
                else:
                    x_var.data.clamp_(clamp_min, clamp_max)

        return x_var.detach()
    finally:
        for p, rg in zip(model.parameters(), req_grad):
            p.requires_grad_(rg)
        model.train(train_state)
