from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def graph_target_tensor(
    graph_idx,
    targets: list[float | None] | None,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not targets:
        return None, None
    if torch.is_tensor(graph_idx):
        idx_list = graph_idx.detach().cpu().tolist()
    elif isinstance(graph_idx, (list, tuple)):
        idx_list = list(graph_idx)
    else:
        idx_list = [int(graph_idx)]
    values = []
    for graph_id in idx_list:
        if 0 <= int(graph_id) < len(targets):
            target_value = targets[int(graph_id)]
        else:
            target_value = None
        values.append(float(target_value) if target_value is not None else float("nan"))
    target = torch.tensor(values, dtype=torch.float32, device=device)
    mask = torch.isfinite(target)
    return target, mask


def _weighted_mean_loss(loss: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return loss.mean()
    weight = sample_weight.reshape(-1).to(device=loss.device, dtype=loss.dtype)
    denom = weight.sum().clamp_min(1e-12)
    return (loss.reshape(-1) * weight).sum() / denom


def compute_supervised_return_loss(
    return_head: torch.nn.Module,
    embeddings: torch.Tensor,
    graph_idx,
    portfolio_targets: list[float | None] | None,
    device: torch.device,
    loss_type: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    target, mask = graph_target_tensor(graph_idx, portfolio_targets, device=device)
    if target is None or mask is None or not mask.any():
        return None, None, None
    pred = return_head(embeddings)
    if pred.ndim == 2 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if pred.ndim != 1:
        raise RuntimeError(f"supervised return head output shape mismatch: {tuple(pred.shape)}")
    loss_mode = str(loss_type).strip().lower()
    if loss_mode == "mse":
        loss = F.mse_loss(pred[mask], target[mask])
    else:
        loss = F.smooth_l1_loss(pred[mask], target[mask])
    return loss, pred, target


def compute_portfolio_head_loss(
    portfolio_head: torch.nn.Module,
    embeddings: torch.Tensor,
    graph_idx,
    portfolio_targets: list[float | None] | None,
    device: torch.device,
    loss_type: str,
    sample_weight: torch.Tensor | None = None,
    cara_risk_aversion: float = 1.0,
    baseline_exposure: float = 0.0,
    delta_scale: float = 1.0,
    max_abs_exposure: float = 1.0,
) -> torch.Tensor | None:
    target, mask = graph_target_tensor(graph_idx, portfolio_targets, device=device)
    if target is None or mask is None or not mask.any():
        return None

    pred_raw = portfolio_head(embeddings)
    if pred_raw.ndim == 2 and pred_raw.size(1) == 1:
        pred_raw = pred_raw.squeeze(1)
    if pred_raw.ndim != 1:
        raise RuntimeError(f"portfolio head output shape mismatch: {tuple(pred_raw.shape)}")
    pred = torch.tanh(pred_raw)
    weight = None
    if sample_weight is not None:
        weight = sample_weight.to(device=device, dtype=pred.dtype).reshape(-1)

    loss_mode = str(loss_type).strip().lower()
    if loss_mode == "mse":
        err = (pred[mask] - target[mask]).pow(2)
        if weight is None:
            return err.mean()
        return _weighted_mean_loss(err, weight[mask])

    if loss_mode in {"cara", "delta_cara", "baseline_cara"}:
        risk_aversion = max(1e-6, float(cara_risk_aversion))
        exposure = pred
        if loss_mode != "cara":
            exposure = float(baseline_exposure) + float(delta_scale) * exposure
        exposure = exposure.clamp(
            min=-abs(float(max_abs_exposure)),
            max=abs(float(max_abs_exposure)),
        )
        pnl = exposure[mask] * target[mask]
        utility_loss = torch.exp(-risk_aversion * pnl)
        if weight is None:
            return utility_loss.mean()
        return _weighted_mean_loss(utility_loss, weight[mask])

    pnl = pred[mask] * target[mask]
    if pnl.numel() == 0:
        return None
    if pnl.numel() == 1:
        return -pnl.mean()
    if weight is None:
        mean = pnl.mean()
        std = pnl.std(unbiased=False) + 1e-6
    else:
        w = weight[mask]
        w = w / w.sum().clamp_min(1e-12)
        mean = (pnl * w).sum()
        std = torch.sqrt(((pnl - mean).pow(2) * w).sum() + 1e-6)
    return -(mean / std)


def bootstrap_graph_latent_loss(
    predictor: torch.nn.Module,
    online_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
) -> torch.Tensor | None:
    if online_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise ValueError("Expected graph embeddings with shape [num_graphs, dim]")
    if online_embeddings.shape != target_embeddings.shape:
        raise ValueError(
            "online_embeddings and target_embeddings must have the same shape"
        )
    if online_embeddings.numel() == 0:
        return None
    pred = predictor(online_embeddings)
    if pred.shape != target_embeddings.shape:
        raise ValueError(
            f"bootstrap predictor output mismatch: pred={tuple(pred.shape)} target={tuple(target_embeddings.shape)}"
        )
    pred_n = F.normalize(pred, p=2, dim=1)
    target_n = F.normalize(target_embeddings.detach(), p=2, dim=1)
    return 2.0 - 2.0 * (pred_n * target_n).sum(dim=1).mean()


def vicreg_graph_loss(
    embeddings: torch.Tensor,
    *,
    view_embeddings: torch.Tensor | None = None,
    invariance_weight: float = 0.0,
    variance_weight: float = 1.0,
    covariance_weight: float = 1.0,
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor | None:
    if embeddings.ndim != 2:
        raise ValueError("Expected embeddings with shape [num_graphs, dim]")
    if embeddings.numel() == 0:
        return None

    loss = embeddings.new_zeros(())
    used = False

    if view_embeddings is not None and float(invariance_weight) > 0:
        if view_embeddings.shape != embeddings.shape:
            raise ValueError(
                f"view_embeddings shape mismatch: {tuple(view_embeddings.shape)} vs {tuple(embeddings.shape)}"
            )
        loss = loss + float(invariance_weight) * F.mse_loss(embeddings, view_embeddings)
        used = True

    num_graphs = int(embeddings.size(0))
    if num_graphs < 2:
        return loss if used else None

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    if float(variance_weight) > 0:
        std = torch.sqrt(centered.pow(2).mean(dim=0) + float(eps))
        var_penalty = F.relu(float(variance_target) - std).mean()
        loss = loss + float(variance_weight) * var_penalty
        used = True

    if float(covariance_weight) > 0 and embeddings.size(1) > 1:
        cov = (centered.T @ centered) / max(1, num_graphs - 1)
        cov = cov - torch.diag(torch.diag(cov))
        cov_penalty = cov.pow(2).sum() / float(embeddings.size(1))
        loss = loss + float(covariance_weight) * cov_penalty
        used = True

    return loss if used else None


def compute_multi_horizon_risk_loss(
    risk_head: torch.nn.Module,
    embeddings: torch.Tensor,
    graph_idx,
    risk_targets_by_horizon: list[list[float | None]],
    device: torch.device,
    risk_loss_type: str,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not risk_targets_by_horizon:
        return None
    if torch.is_tensor(graph_idx):
        idx_list = graph_idx.detach().cpu().tolist()
    elif isinstance(graph_idx, (list, tuple)):
        idx_list = list(graph_idx)
    else:
        idx_list = [int(graph_idx)]

    target_rows = []
    for graph_id in idx_list:
        row = []
        for horizon_targets in risk_targets_by_horizon:
            if 0 <= int(graph_id) < len(horizon_targets):
                target_value = horizon_targets[int(graph_id)]
            else:
                target_value = None
            row.append(float(target_value) if target_value is not None else float("nan"))
        target_rows.append(row)

    target = torch.tensor(target_rows, dtype=torch.float32, device=device)
    mask = torch.isfinite(target)
    if not mask.any():
        return None

    pred = risk_head(embeddings)
    if pred.ndim == 1:
        pred = pred.unsqueeze(-1)
    if pred.shape != target.shape:
        raise RuntimeError(
            f"risk head output shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    row_weight = None
    if sample_weight is not None:
        row_weight = sample_weight.to(device=device, dtype=pred.dtype).reshape(-1, 1).expand_as(pred)
    if str(risk_loss_type).strip().lower() == "mse":
        err = (pred - target).pow(2)
    else:
        err = F.smooth_l1_loss(pred, target, reduction="none")
    if row_weight is None:
        return err[mask].mean()
    return _weighted_mean_loss(err[mask], row_weight[mask])


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
    if x.size < 2:
        return float("nan")
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def regression_eval_metrics(pred: list[float], target: list[float]) -> dict[str, float]:
    pred_np = np.asarray(pred, dtype=float)
    target_np = np.asarray(target, dtype=float)
    mask = np.isfinite(pred_np) & np.isfinite(target_np)
    pred_np = pred_np[mask]
    target_np = target_np[mask]
    if pred_np.size == 0:
        return {
            "eval_return_mse": float("nan"),
            "eval_return_mae": float("nan"),
            "eval_return_corr": float("nan"),
            "eval_return_rank_corr": float("nan"),
        }
    pred_rank = pd.Series(pred_np).rank(method="average").to_numpy(dtype=float)
    target_rank = pd.Series(target_np).rank(method="average").to_numpy(dtype=float)
    return {
        "eval_return_mse": float(np.mean((pred_np - target_np) ** 2)),
        "eval_return_mae": float(np.mean(np.abs(pred_np - target_np))),
        "eval_return_corr": safe_corrcoef(pred_np, target_np),
        "eval_return_rank_corr": safe_corrcoef(pred_rank, target_rank),
    }
