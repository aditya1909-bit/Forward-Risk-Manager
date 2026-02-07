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


def _edge_corr_loss(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
) -> torch.Tensor:
    if edge_attr is None:
        return torch.tensor(0.0, device=x.device)
    if edge_attr.ndim == 2 and edge_attr.shape[1] == 1:
        w = edge_attr.squeeze(1)
    else:
        w = edge_attr

    src = edge_index[0]
    dst = edge_index[1]
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
        x_var = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x_var], lr=config.lr)

        mean0 = x0.mean()
        std0 = x0.std() + 1e-6
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

        for _ in range(config.steps):
            if forward_fn is not None:
                h = forward_fn(x_var)
            else:
                h = model(x_var, edge_index, edge_weight=edge_weight)
            g = goodness(h, batch, temperature=config.goodness_temp).mean()

            l2 = (x_var - x0).pow(2).mean()
            mean_pen = (x_var.mean() - mean0).pow(2)
            std_pen = (x_var.std() - std0).pow(2)
            corr_pen = _edge_corr_loss(x_var, edge_index, edge_attr)

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

            if config.clamp_std is not None:
                x_var.data.clamp_(clamp_min, clamp_max)

        return x_var.detach()
    finally:
        for p, rg in zip(model.parameters(), req_grad):
            p.requires_grad_(rg)
        model.train(train_state)
