#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import random
import time
from pathlib import Path
import sys
import tomllib

import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import GCNEncoder
from frisk.ff import (
    ff_loss,
    goodness,
    make_negative,
    pairwise_distance_forward_loss,
    permute_graph_embeddings,
    self_contrastive_loss,
    self_contrastive_retrieval_accuracy,
)
from frisk.hallucinate import HallucinationConfig, hallucinate_negative
from frisk.device import resolve_device, sync_device
from frisk.eval_metrics import (
    binary_auprc,
    binary_auroc,
    binary_brier_and_ece,
    ff_binary_metrics,
)
from frisk.splits import (
    is_walk_forward_mode,
    simple_split_indices,
    simple_train_eval_split,
    walk_forward_splits,
)
from frisk.econ_eval import (
    evaluate_goodness_strategy,
    infer_graph_goodness,
    load_forward_returns_from_prices,
    resolve_price_ticker,
)

_NEG_AUG_MODES = {
    "shuffle",
    "noise",
    "shuffle+noise",
    "time_flip",
    "shuffle+time_flip",
    "time_flip+noise",
    "block_bootstrap",
    "cross_asset_mix",
    "phase_randomize",
}


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _choose_device(device: str) -> torch.device:
    return resolve_device(device)


def _parse_amp_dtype(value) -> torch.dtype:
    name = str(value).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def _build_optimizer(params, lr: float, device: torch.device, use_fused: bool):
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


def _make_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_if_needed(enabled: bool, dtype: torch.dtype):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _optimizer_step(optim, loss: torch.Tensor, grad_clip: float, clip_params, scaler) -> None:
    optim.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
        scaler.step(optim)
        scaler.update()
        return
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
    optim.step()


def _split_graphs(
    graphs,
    eval_frac: float = 0.2,
    seed: int = 7,
    split_mode: str = "chronological",
):
    return simple_train_eval_split(
        graphs,
        eval_frac=eval_frac,
        seed=seed,
        split_mode=split_mode,
    )


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _aggregate_fold_results(fold_rows: list[dict]) -> dict:
    if not fold_rows:
        return {}
    out: dict[str, object] = {}
    key_union = sorted({k for r in fold_rows for k in r.keys()})
    skip_keys = {
        "mode",
        "row_type",
        "walk_forward_num_folds",
    }
    for key in key_union:
        if key in skip_keys or key.startswith("fold_"):
            continue
        vals: list[float] = []
        for row in fold_rows:
            value = row.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.number)):
                fv = float(value)
                if np.isfinite(fv):
                    vals.append(fv)
        if vals:
            mean_val, std_val = _mean_std(vals)
            out[key] = mean_val
            out[f"{key}_std"] = std_val
    first = fold_rows[0]
    for key in (
        "eval_objective",
        "neg_mode_effective",
        "eval_neg_mode_effective",
    ):
        if key in first:
            out[key] = first[key]
    out["walk_forward_num_folds"] = len(fold_rows)
    out["split_mode_effective"] = "walk_forward"
    return out


def _compute_econ_metrics_for_eval(
    model,
    eval_graphs,
    eval_dates,
    config: dict,
):
    meta = {
        "econ_ticker_requested": str(config.get("econ_ticker", "")),
        "econ_ticker_effective": str(config.get("econ_ticker_effective", "")),
        "econ_ticker_source": str(config.get("econ_ticker_source", "")),
        "econ_ticker_rows": float(config.get("econ_ticker_rows", 0) or 0),
    }
    if not bool(config.get("econ_enabled", False)):
        return meta
    if not eval_graphs or not eval_dates:
        return meta
    fwd_ret_1 = config.get("econ_fwd_ret_1")
    if fwd_ret_1 is None:
        return meta
    g = infer_graph_goodness(
        model,
        eval_graphs,
        goodness_temp=float(config.get("goodness_temp", 1.0)),
        batch_size=int(config.get("econ_loader_batch_size", config.get("batch_size", 64))),
    )
    if g.size == 0:
        return meta
    out = evaluate_goodness_strategy(
        eval_dates,
        g,
        fwd_ret_1=fwd_ret_1,
        signal_window=int(config.get("econ_signal_window", 126)),
        signal_quantile=float(config.get("econ_signal_quantile", 0.5)),
        turnover_cost_bps=float(config.get("econ_turnover_cost_bps", 0.0)),
        trading_days=int(config.get("econ_trading_days", 252)),
    )
    out.update(meta)
    return out


def _get_use_mode(epoch: int, neg_mode: str, warmup: int, mix_start: float, mix_end: float, ramp: int):
    if neg_mode == "schedule":
        return "shuffle" if epoch <= warmup else "hallucinate"
    if neg_mode == "mix":
        if epoch <= warmup:
            return "shuffle"
        ramp = max(1, ramp)
        progress = min(1.0, (epoch - warmup) / ramp)
        p_hall = mix_start + progress * (mix_end - mix_start)
        return "hallucinate" if random.random() < p_hall else "shuffle"
    return neg_mode


def _resolve_mode(eval_mode: str, train_mode: str) -> str:
    mode = str(eval_mode).strip().lower()
    if mode in ("", "auto"):
        return str(train_mode).strip().lower()
    return mode


def _warn_self_contrastive_eval_view(config: dict, mode: str) -> None:
    train_mode = str(config.get("neg_mode", "")).strip().lower()
    eval_mode = _resolve_mode(config.get("eval_neg_mode", "auto"), train_mode)
    if eval_mode != "self_contrastive":
        return

    train_view = str(config.get("self_contrastive_view_mode", "shuffle+noise")).strip().lower()
    eval_view = str(config.get("self_contrastive_eval_view_mode", train_view)).strip().lower()
    train_noise = float(
        config.get(
            "self_contrastive_view_noise_std",
            config.get("noise_std", 0.05),
        )
    )
    eval_noise = float(
        config.get(
            "self_contrastive_eval_noise_std",
            train_noise,
        )
    )
    harder_view = ("time_flip" in eval_view and "time_flip" not in train_view) or (
        eval_noise > (1.5 * max(1e-8, train_noise))
    )
    if harder_view:
        print(
            "warning: "
            f"{mode} uses harder self_contrastive eval views than training "
            f"(train={train_view}@{train_noise:.4f}, eval={eval_view}@{eval_noise:.4f}). "
            "This can sharply depress eval_sc_acc/eval_sc_gap."
        )


def _block_endpoint_indices(num_layers: int, block_size: int) -> list[int]:
    if num_layers <= 0:
        return []
    step = max(1, int(block_size))
    endpoints = list(range(step - 1, num_layers, step))
    if endpoints[-1] != num_layers - 1:
        endpoints.append(num_layers - 1)
    return endpoints


def _make_negatives(
    model,
    x,
    batch,
    edge_index,
    edge_attr,
    edge_weight,
    use_mode,
    noise_std,
    hall_cfg: HallucinationConfig,
    window_len: int | None = None,
    summary_dim: int = 0,
):
    if use_mode == "self_contrastive":
        use_mode = "shuffle"
    if use_mode == "hallucinate":
        return hallucinate_negative(
            model,
            x,
            edge_index,
            edge_attr,
            batch,
            hall_cfg,
            edge_weight=edge_weight,
        )
    return make_negative(
        x,
        batch,
        mode=use_mode,
        noise_std=noise_std,
        window_len=window_len,
        summary_dim=summary_dim,
    )


def _make_self_contrastive_view(
    x: torch.Tensor,
    batch: torch.Tensor,
    view_mode: str,
    view_noise_std: float,
    window_len: int | None = None,
    summary_dim: int = 0,
) -> torch.Tensor:
    mode = str(view_mode).strip().lower()
    if mode in ("", "auto"):
        mode = "shuffle+noise"
    valid_modes = {
        "shuffle",
        "noise",
        "shuffle+noise",
        "time_flip",
        "shuffle+time_flip",
        "time_flip+noise",
        "block_bootstrap",
        "cross_asset_mix",
        "phase_randomize",
    }
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported self_contrastive view_mode={view_mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )
    return make_negative(
        x,
        batch,
        mode=mode,
        noise_std=max(0.0, float(view_noise_std)),
        window_len=window_len,
        summary_dim=summary_dim,
    )


def _self_contrastive_step(
    h_pos: torch.Tensor,
    h_view: torch.Tensor,
    batch: torch.Tensor,
    temperature: float,
    max_graphs: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z_pos = global_mean_pool(h_pos, batch)
    z_view = global_mean_pool(h_view, batch)
    if max_graphs and z_pos.size(0) > int(max_graphs):
        idx = torch.randperm(z_pos.size(0), device=z_pos.device)[: int(max_graphs)]
        z_pos = z_pos.index_select(0, idx)
        z_view = z_view.index_select(0, idx)
    loss, pos_score, neg_score = self_contrastive_loss(
        z_pos,
        z_view,
        temperature=temperature,
    )
    return loss, pos_score, neg_score, z_pos, z_view


def _sync(device: torch.device) -> None:
    sync_device(device)


def _eval_ff_metrics(
    model,
    loader,
    goodness_temp,
    goodness_target,
    neg_mode,
    noise_std,
    hall_cfg,
    sc_temp: float = 0.2,
    sc_view_mode: str = "shuffle+noise",
    sc_view_noise_std: float | None = None,
    window_len: int | None = None,
    summary_dim: int = 0,
    ece_bins: int = 10,
):
    eval_mode = str(neg_mode).strip().lower()
    if eval_mode == "self_contrastive":
        prev_mode = model.training
        model.eval()
        if sc_view_noise_std is None:
            sc_view_noise_std_eff = max(1e-5, 0.5 * max(0.0, float(noise_std)))
        else:
            sc_view_noise_std_eff = max(0.0, float(sc_view_noise_std))
        sc_losses = []
        sc_pos = []
        sc_neg = []
        sc_acc = []
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            x = batch.x
            edge_weight = getattr(batch, "edge_weight", None)
            x_view = _make_self_contrastive_view(
                x,
                batch.batch,
                view_mode=sc_view_mode,
                view_noise_std=sc_view_noise_std_eff,
                window_len=window_len,
                summary_dim=summary_dim,
            )
            with torch.no_grad():
                h_a = model(x, batch.edge_index, edge_weight=edge_weight)
                h_b = model(x_view, batch.edge_index, edge_weight=edge_weight)
                z_a = global_mean_pool(h_a, batch.batch)
                z_b = global_mean_pool(h_b, batch.batch)
                loss, pos_score, neg_score = self_contrastive_loss(
                    z_a,
                    z_b,
                    temperature=sc_temp,
                )
                acc = self_contrastive_retrieval_accuracy(z_a, z_b)
            sc_losses.append(float(loss.detach().cpu()))
            sc_pos.append(float(pos_score.detach().cpu()))
            sc_neg.append(float(neg_score.detach().cpu()))
            sc_acc.append(float(acc.detach().cpu()))
        model.train(prev_mode)
        pos_mean = float(np.mean(sc_pos)) if sc_pos else 0.0
        neg_mean = float(np.mean(sc_neg)) if sc_neg else 0.0
        return {
            "eval_objective": "self_contrastive",
            "eval_sc_loss": float(np.mean(sc_losses)) if sc_losses else 0.0,
            "eval_sc_pos": pos_mean,
            "eval_sc_neg": neg_mean,
            "eval_sc_gap": pos_mean - neg_mean,
            "eval_sc_acc": float(np.mean(sc_acc)) if sc_acc else 0.0,
            "eval_sc_view_mode": str(sc_view_mode).strip().lower(),
            "eval_sc_view_noise_std": sc_view_noise_std_eff,
            "eval_g_pos": pos_mean,
            "eval_g_neg": neg_mean,
            "eval_sep": pos_mean - neg_mean,
            "eval_acc": float(np.mean(sc_acc)) if sc_acc else 0.0,
            "eval_auroc": float("nan"),
            "eval_auprc": float("nan"),
            "eval_brier": float("nan"),
            "eval_ece": float("nan"),
        }

    model.eval()
    gpos = []
    gneg = []
    gpos_all = []
    gneg_all = []
    acc_num = 0
    acc_den = 0
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        x = batch.x
        edge_weight = getattr(batch, "edge_weight", None)
        with torch.no_grad():
            h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
            g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)

        if eval_mode == "hallucinate":
            with torch.enable_grad():
                x_neg = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    eval_mode,
                    noise_std,
                    hall_cfg,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )
        else:
            with torch.no_grad():
                x_neg = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    eval_mode,
                    noise_std,
                    hall_cfg,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )

        with torch.no_grad():
            h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
            g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp)
            pred_pos = (g_pos > goodness_target)
            pred_neg = (g_neg <= goodness_target)
            acc_num += (pred_pos.sum() + pred_neg.sum()).item()
            acc_den += 2 * g_pos.numel()
            gpos.append(g_pos.mean().item())
            gneg.append(g_neg.mean().item())
            gpos_all.extend(g_pos.detach().cpu().tolist())
            gneg_all.extend(g_neg.detach().cpu().tolist())
    acc = acc_num / acc_den if acc_den else 0.0
    pos_mean = float(np.mean(gpos)) if gpos else 0.0
    neg_mean = float(np.mean(gneg)) if gneg else 0.0
    cls_metrics = ff_binary_metrics(
        np.asarray(gpos_all, dtype=float),
        np.asarray(gneg_all, dtype=float),
        threshold=float(goodness_target),
        ece_bins=int(ece_bins),
    )
    return {
        "eval_objective": "ff",
        "eval_g_pos": pos_mean,
        "eval_g_neg": neg_mean,
        "eval_sep": pos_mean - neg_mean,
        "eval_acc": float(acc),
        **cls_metrics,
    }


def _finite_or_none(value):
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _objective_primary_metric(metrics: dict) -> tuple[str, float]:
    objective = str(metrics.get("eval_objective", "")).strip().lower()
    if objective == "self_contrastive":
        for key in ("eval_sc_gap", "eval_sep", "eval_sc_acc", "eval_acc"):
            v = _finite_or_none(metrics.get(key))
            if v is not None:
                return key, v
    if objective in {"bce", "backprop"}:
        for key in ("eval_auroc", "eval_auprc", "eval_sep", "eval_acc"):
            v = _finite_or_none(metrics.get(key))
            if v is not None:
                return key, v
    for key in ("eval_sep", "eval_auroc", "eval_auprc", "eval_acc"):
        v = _finite_or_none(metrics.get(key))
        if v is not None:
            return key, v
    return "none", float("nan")


def _objective_track(objective: str) -> str:
    obj = str(objective).strip().lower()
    if obj == "self_contrastive":
        return "encoder"
    if obj in {"ff", "forward_forward", "forward-forward"} or obj.startswith("ff_"):
        return "critic"
    if obj in {"bce", "backprop"}:
        return "classifier"
    return "unknown"


def _objective_primary_metric_robust(metrics: dict) -> tuple[str, float]:
    # Robust metric intentionally mirrors objective-primary metric.
    # Time-flip discrimination belongs to critic evaluation/sanity checks, not self-contrastive ranking.
    return _objective_primary_metric(metrics)


def _attach_primary_metrics(out: dict) -> None:
    metric_name, metric_val = _objective_primary_metric(out)
    robust_name, robust_val = _objective_primary_metric_robust(out)
    out["objective_track"] = _objective_track(out.get("eval_objective", ""))
    out["primary_eval_metric_name"] = metric_name
    out["primary_eval_metric"] = metric_val
    out["primary_eval_metric_robust_name"] = robust_name
    out["primary_eval_metric_robust"] = robust_val


def _calibrate_goodness_target(
    model,
    loader,
    goodness_temp,
    default_target,
    neg_mode,
    noise_std,
    hall_cfg,
    window_len: int | None = None,
    summary_dim: int = 0,
    max_batches: int = 0,
    quantiles: int = 31,
):
    if str(neg_mode).strip().lower() == "self_contrastive":
        return float(default_target), float("nan")
    model.eval()
    pos_vals = []
    neg_vals = []
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = batch.to(next(model.parameters()).device)
        x = batch.x
        edge_weight = getattr(batch, "edge_weight", None)
        with torch.no_grad():
            h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
            g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)

        if neg_mode == "hallucinate":
            with torch.enable_grad():
                x_neg = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    neg_mode,
                    noise_std,
                    hall_cfg,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )
        else:
            with torch.no_grad():
                x_neg = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    neg_mode,
                    noise_std,
                    hall_cfg,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )

        with torch.no_grad():
            h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
            g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp)

        pos_vals.append(g_pos.detach().cpu())
        neg_vals.append(g_neg.detach().cpu())

    if not pos_vals or not neg_vals:
        return float(default_target), 0.0

    pos_all = torch.cat(pos_vals)
    neg_all = torch.cat(neg_vals)
    vals = torch.cat([pos_all, neg_all])
    qn = max(5, int(quantiles))
    qs = torch.linspace(0.02, 0.98, steps=qn)
    cands = torch.quantile(vals, qs).unique()
    if cands.numel() == 0:
        return float(default_target), 0.0

    best_target = float(default_target)
    best_acc = -1.0
    for t in cands.tolist():
        t = float(t)
        acc = 0.5 * (
            float((pos_all > t).float().mean().item())
            + float((neg_all <= t).float().mean().item())
        )
        if acc > best_acc:
            best_acc = acc
            best_target = t

    return best_target, best_acc


def _benchmark_ff(
    graphs,
    device,
    config,
    layerwise: bool,
    train_graphs=None,
    eval_graphs=None,
    eval_dates=None,
):
    if train_graphs is None or eval_graphs is None:
        train_graphs, eval_graphs = _split_graphs(
            graphs,
            eval_frac=config["eval_frac"],
            seed=config["seed"],
            split_mode=config.get("split_mode", "chronological"),
        )
    else:
        train_graphs = list(train_graphs)
        eval_graphs = list(eval_graphs)
        if not train_graphs or not eval_graphs:
            raise ValueError("Explicit fold split must include non-empty train and eval graphs.")
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "drop_last": False,
        "num_workers": config["loader_workers"],
        "pin_memory": bool(config.get("pin_memory", False)) if device.type == "cuda" else False,
    }
    if config["loader_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
        mp_ctx = config.get("multiprocessing_context", "")
        if mp_ctx:
            loader_kwargs["multiprocessing_context"] = mp_ctx
    loader = DataLoader(train_graphs, **loader_kwargs)
    eval_loader = DataLoader(eval_graphs, batch_size=config["batch_size"], shuffle=False)

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    optim = _build_optimizer(
        model.parameters(),
        lr=config["lr"],
        device=device,
        use_fused=bool(config.get("fused_optimizer", True)),
    )
    clip_params = list(model.parameters())

    hall_cfg = HallucinationConfig(
        steps=config["hall_steps"],
        lr=config["hall_lr"],
        l2_weight=config["hall_l2"],
        mean_weight=config["hall_mean"],
        std_weight=config["hall_std"],
        corr_weight=config["hall_corr"],
        clamp_std=config["hall_clamp"],
        goodness_temp=config["goodness_temp"],
        node_fraction=config["hall_node_fraction"],
        node_min=config["hall_node_min"],
        return_slice_len=int(config.get("window_len", 0)),
        penalty_scope=str(config.get("hall_penalty_scope", "returns")),
        corr_scope=str(config.get("hall_corr_scope", "returns")),
        freeze_non_return_features=bool(config.get("hall_freeze_non_return", True)),
        corr_every_n_steps=int(config.get("hall_corr_every_n_steps", 1)),
        corr_edge_fraction=float(config.get("hall_corr_edge_fraction", 1.0)),
        corr_edge_min=int(config.get("hall_corr_edge_min", 1)),
    )
    sc_temp = float(config.get("self_contrastive_temp", 0.2))
    sc_max_graphs = int(config.get("self_contrastive_max_graphs", 0))
    sc_ff_weight = max(0.0, float(config.get("self_contrastive_ff_weight", 0.0)))
    sc_ff_neg_mode = str(config.get("self_contrastive_ff_neg_mode", "shuffle+noise")).strip().lower()
    if sc_ff_neg_mode not in _NEG_AUG_MODES:
        sc_ff_neg_mode = "shuffle+noise"
    sc_ff_noise_std = max(
        0.0,
        float(config.get("self_contrastive_ff_noise_std", config.get("noise_std", 0.05))),
    )
    sc_ff_target = float(config.get("self_contrastive_ff_target", config["goodness_target"]))
    dist_weight = float(config.get("distance_forward_weight", 0.0))
    dist_margin = float(config.get("distance_forward_margin", 0.15))
    dist_max_graphs = int(config.get("distance_forward_max_graphs", 0))
    dist_interval = max(1, int(config.get("distance_forward_interval", 1)))
    amp_dtype = _parse_amp_dtype(config.get("amp_dtype", "float16"))
    amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
    if amp_enabled and amp_dtype == torch.bfloat16:
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if not bf16_ok:
            amp_dtype = torch.float16
    scaler = _make_scaler(amp_enabled and amp_dtype == torch.float16)
    train_neg_mode = str(config["neg_mode"]).strip().lower()
    if layerwise and train_neg_mode == "self_contrastive":
        fallback = str(config.get("layerwise_neg_mode", "shuffle")).strip().lower()
        print(
            "ff_layerwise does not support self_contrastive negatives directly; "
            f"using layerwise_neg_mode={fallback!r} for training."
        )
        train_neg_mode = fallback
    eval_mode = _resolve_mode(config.get("eval_neg_mode", "auto"), train_neg_mode)
    ff_blockwise = bool(config.get("ff_blockwise", False)) and bool(layerwise)
    ff_block_size = max(1, int(config.get("ff_block_size", 2)))
    if ff_block_size <= 1:
        ff_blockwise = False
    ff_block_endpoints = (
        _block_endpoint_indices(len(model.layers), ff_block_size) if ff_blockwise else []
    )
    if ff_blockwise:
        print(
            "benchmark layerwise blockwise endpoints: "
            + ",".join(str(i + 1) for i in ff_block_endpoints)
        )

    epoch_times = []
    for epoch in tqdm(
        range(1, config["epochs"] + 1),
        desc="Benchmark",
        unit="epoch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        model.train()
        t0 = time.perf_counter()
        graphs_seen = 0
        total_loss = 0.0
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            x = batch.x
            edge_weight = getattr(batch, "edge_weight", None)

            use_mode = _get_use_mode(
                epoch,
                train_neg_mode,
                config["neg_warmup_epochs"],
                config["neg_mix_start"],
                config["neg_mix_end"],
                config["neg_mix_ramp_epochs"],
            )
            apply_distance = dist_weight > 0 and (batch_idx % dist_interval == 0)
            step_scaler = scaler if (amp_enabled and (use_mode == "self_contrastive" or layerwise)) else None

            if layerwise:
                if ff_blockwise:
                    layer_mode = use_mode
                    if layer_mode == "self_contrastive":
                        layer_mode = "shuffle"
                    x_neg = _make_negatives(
                        model,
                        x,
                        batch.batch,
                        batch.edge_index,
                        getattr(batch, "edge_attr", None),
                        edge_weight,
                        "hallucinate" if layer_mode == "hallucinate" else config["layerwise_neg_mode"],
                        config["layerwise_noise_std"],
                        hall_cfg,
                        window_len=config.get("window_len"),
                        summary_dim=config.get("summary_dim", 0),
                    )
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        layers_pos = model(x, batch.edge_index, edge_weight=edge_weight, return_all=True)
                        layers_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True)
                    if layer_mode == "hallucinate":
                        last_idx = ff_block_endpoints[-1]
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            g_pos_probe = goodness(
                                layers_pos[last_idx], batch.batch, temperature=config["goodness_temp"]
                            ).mean().item()
                            g_neg_probe = goodness(
                                layers_neg[last_idx], batch.batch, temperature=config["goodness_temp"]
                            ).mean().item()
                        if g_neg_probe > g_pos_probe + config["neg_gate_margin"]:
                            x_neg = make_negative(
                                x,
                                batch.batch,
                                mode="shuffle",
                                noise_std=config["noise_std"],
                                window_len=config.get("window_len"),
                                summary_dim=config.get("summary_dim", 0),
                            )
                            with _autocast_if_needed(step_scaler is not None, amp_dtype):
                                layers_neg = model(
                                    x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True
                                )
                    loss = 0.0
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        for li in ff_block_endpoints:
                            g_pos = goodness(layers_pos[li], batch.batch, temperature=config["goodness_temp"])
                            g_neg = goodness(layers_neg[li], batch.batch, temperature=config["goodness_temp"])
                            loss = loss + ff_loss(g_pos, g_neg, target=config["goodness_target"])
                    loss = loss / max(1, len(ff_block_endpoints))
                    _optimizer_step(
                        optim=optim,
                        loss=loss,
                        grad_clip=float(config["grad_clip"]),
                        clip_params=clip_params,
                        scaler=step_scaler,
                    )
                    total_loss += loss.item()
                else:
                    x_in = x
                    for li in range(len(model.layers)):
                        layer_mode = use_mode
                        if layer_mode == "self_contrastive":
                            layer_mode = "shuffle"
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            h_pos = model.forward_layer(x_in, batch.edge_index, edge_weight, li)
                            g_pos = goodness(h_pos, batch.batch, temperature=config["goodness_temp"])
                        x_neg = _make_negatives(
                            model,
                            x_in,
                            batch.batch,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            edge_weight,
                            "hallucinate" if (layer_mode == "hallucinate" and li == 0) else "shuffle",
                            config["noise_std"],
                            hall_cfg,
                            window_len=config.get("window_len"),
                            summary_dim=config.get("summary_dim", 0),
                        )
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            h_neg = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                            g_neg = goodness(h_neg, batch.batch, temperature=config["goodness_temp"])
                            loss = ff_loss(g_pos, g_neg, target=config["goodness_target"])
                        _optimizer_step(
                            optim=optim,
                            loss=loss,
                            grad_clip=float(config["grad_clip"]),
                            clip_params=clip_params,
                            scaler=step_scaler,
                        )
                        total_loss += loss.item()
                        x_in = h_pos.detach()
            else:
                if use_mode == "self_contrastive":
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                        x_view = _make_self_contrastive_view(
                            x,
                            batch.batch,
                            view_mode=config["self_contrastive_view_mode"],
                            view_noise_std=config["self_contrastive_view_noise_std"],
                            window_len=config.get("window_len"),
                            summary_dim=config.get("summary_dim", 0),
                        )
                        h_view = model(x_view, batch.edge_index, edge_weight=edge_weight)
                        loss, _, _, z_pos, z_view = _self_contrastive_step(
                            h_pos,
                            h_view,
                            batch.batch,
                            temperature=sc_temp,
                            max_graphs=sc_max_graphs,
                        )
                        if apply_distance:
                            z_neg = permute_graph_embeddings(z_view)
                            loss = loss + dist_weight * pairwise_distance_forward_loss(
                                z_pos,
                                z_neg,
                                margin=dist_margin,
                                max_graphs=dist_max_graphs,
                            )
                        if sc_ff_weight > 0:
                            x_neg_aux = make_negative(
                                x,
                                batch.batch,
                                mode=sc_ff_neg_mode,
                                noise_std=sc_ff_noise_std,
                                window_len=config.get("window_len"),
                                summary_dim=config.get("summary_dim", 0),
                            )
                            h_neg_aux = model(x_neg_aux, batch.edge_index, edge_weight=edge_weight)
                            g_pos_aux = goodness(h_pos, batch.batch, temperature=config["goodness_temp"])
                            g_neg_aux = goodness(h_neg_aux, batch.batch, temperature=config["goodness_temp"])
                            loss = loss + sc_ff_weight * ff_loss(
                                g_pos_aux,
                                g_neg_aux,
                                target=sc_ff_target,
                            )
                else:
                    h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                    g_pos = goodness(h_pos, batch.batch, temperature=config["goodness_temp"])
                    x_neg = _make_negatives(
                        model,
                        x,
                        batch.batch,
                        batch.edge_index,
                        getattr(batch, "edge_attr", None),
                        edge_weight,
                        use_mode,
                        config["noise_std"],
                        hall_cfg,
                        window_len=config.get("window_len"),
                        summary_dim=config.get("summary_dim", 0),
                    )
                    h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                    g_neg = goodness(h_neg, batch.batch, temperature=config["goodness_temp"])
                    loss = ff_loss(g_pos, g_neg, target=config["goodness_target"])
                    if apply_distance:
                        z_pos = global_mean_pool(h_pos, batch.batch)
                        z_neg = global_mean_pool(h_neg, batch.batch)
                        loss = loss + dist_weight * pairwise_distance_forward_loss(
                            z_pos,
                            z_neg,
                            margin=dist_margin,
                            max_graphs=dist_max_graphs,
                        )
                _optimizer_step(
                    optim=optim,
                    loss=loss,
                    grad_clip=float(config["grad_clip"]),
                    clip_params=clip_params,
                    scaler=step_scaler,
                )
                total_loss += loss.item()

            graphs_seen += batch.num_graphs

        _sync(device)
        dt = time.perf_counter() - t0
        epoch_times.append((dt, graphs_seen))

    target_eval = float(config["goodness_target"])
    target_cal_acc = float("nan")
    if bool(config.get("calibrate_target", True)) and eval_mode != "self_contrastive":
        calib_loader = DataLoader(
            train_graphs, batch_size=config["batch_size"], shuffle=False
        )
        target_eval, target_cal_acc = _calibrate_goodness_target(
            model,
            calib_loader,
            config["goodness_temp"],
            config["goodness_target"],
            eval_mode,
            config["noise_std"],
            hall_cfg,
            window_len=config.get("window_len"),
            summary_dim=config.get("summary_dim", 0),
            max_batches=int(config.get("calibrate_batches", 0)),
            quantiles=int(config.get("calibrate_quantiles", 31)),
        )
        print(
            "calibrated goodness_target="
            f"{target_eval:.4f} (train-cal acc={target_cal_acc:.4f})"
        )

    eval_metrics = _eval_ff_metrics(
        model,
        eval_loader,
        config["goodness_temp"],
        target_eval,
        eval_mode,
        config["noise_std"],
        hall_cfg,
        sc_temp=sc_temp,
        sc_view_mode=config.get("self_contrastive_eval_view_mode", "shuffle+noise"),
        sc_view_noise_std=config.get("self_contrastive_eval_noise_std"),
        window_len=config.get("window_len"),
        summary_dim=config.get("summary_dim", 0),
        ece_bins=int(config.get("ece_bins", 10)),
    )
    warm = int(config.get("timing_warmup_epochs", 0))
    usable = epoch_times[warm:] if warm < len(epoch_times) else epoch_times
    avg_time = float(np.mean([t for t, _ in usable]))
    avg_gps = float(np.mean([g / t for t, g in usable]))
    out = {
        "avg_epoch_s": avg_time,
        "graphs_per_s": avg_gps,
        "goodness_target_eval": target_eval,
        "target_cal_acc": target_cal_acc,
        "neg_mode_effective": train_neg_mode,
        "eval_neg_mode_effective": eval_mode,
    }
    out.update(eval_metrics)
    econ = _compute_econ_metrics_for_eval(
        model,
        eval_graphs,
        eval_dates or [],
        config,
    )
    if econ:
        out.update(econ)

    eval_neg_modes = config.get("eval_neg_modes", [])
    if isinstance(eval_neg_modes, str):
        eval_neg_modes = [m.strip() for m in eval_neg_modes.split(",") if m.strip()]
    extra_modes = [str(m).strip().lower() for m in eval_neg_modes if str(m).strip()]
    if extra_modes:
        reported = []
        skipped = []
        objective_track = _objective_track(out.get("eval_objective", ""))
        for mode in extra_modes:
            if objective_track == "encoder" and "time_flip" in mode:
                skipped.append(mode)
                continue
            mode_metrics = _eval_ff_metrics(
                model,
                eval_loader,
                config["goodness_temp"],
                target_eval,
                mode,
                config["noise_std"],
                hall_cfg,
                sc_temp=sc_temp,
                sc_view_mode=config.get("self_contrastive_eval_view_mode", "shuffle+noise"),
                sc_view_noise_std=config.get("self_contrastive_eval_noise_std"),
                window_len=config.get("window_len"),
                summary_dim=config.get("summary_dim", 0),
                ece_bins=int(config.get("ece_bins", 10)),
            )
            mode_key = mode.replace("+", "_plus_")
            out[f"eval_{mode_key}_acc"] = mode_metrics.get("eval_acc")
            out[f"eval_{mode_key}_sep"] = mode_metrics.get("eval_sep")
            out[f"eval_{mode_key}_auroc"] = mode_metrics.get("eval_auroc")
            out[f"eval_{mode_key}_auprc"] = mode_metrics.get("eval_auprc")
            out[f"eval_{mode_key}_brier"] = mode_metrics.get("eval_brier")
            out[f"eval_{mode_key}_ece"] = mode_metrics.get("eval_ece")
            reported.append(mode)
        out["eval_neg_modes_reported"] = ",".join(reported)
        if skipped:
            out["eval_neg_modes_skipped"] = ",".join(skipped)
    _attach_primary_metrics(out)
    return out


def _benchmark_backprop(
    graphs,
    device,
    config,
    train_graphs=None,
    eval_graphs=None,
    eval_dates=None,
):
    if train_graphs is None or eval_graphs is None:
        train_graphs, eval_graphs = _split_graphs(
            graphs,
            eval_frac=config["eval_frac"],
            seed=config["seed"],
            split_mode=config.get("split_mode", "chronological"),
        )
    else:
        train_graphs = list(train_graphs)
        eval_graphs = list(eval_graphs)
        if not train_graphs or not eval_graphs:
            raise ValueError("Explicit fold split must include non-empty train and eval graphs.")
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "drop_last": False,
        "num_workers": config["loader_workers"],
        "pin_memory": bool(config.get("pin_memory", False)) if device.type == "cuda" else False,
    }
    if config["loader_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
        mp_ctx = config.get("multiprocessing_context", "")
        if mp_ctx:
            loader_kwargs["multiprocessing_context"] = mp_ctx
    loader = DataLoader(train_graphs, **loader_kwargs)
    eval_loader = DataLoader(eval_graphs, batch_size=config["batch_size"], shuffle=False)

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    head = torch.nn.Linear(config["hidden_dim"], 1).to(device)
    optim_params = list(model.parameters()) + list(head.parameters())
    optim = _build_optimizer(
        optim_params,
        lr=config["lr"],
        device=device,
        use_fused=bool(config.get("backprop_fused_optimizer", True)),
    )
    bce = torch.nn.BCEWithLogitsLoss()
    bp_amp_dtype = _parse_amp_dtype(config.get("backprop_amp_dtype", "float16"))
    bp_amp_enabled = bool(config.get("backprop_amp", False) and device.type == "cuda")
    if bp_amp_enabled and bp_amp_dtype == torch.bfloat16:
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if not bf16_ok:
            bp_amp_dtype = torch.float16
    bp_scaler = _make_scaler(bp_amp_enabled and bp_amp_dtype == torch.float16)

    hall_cfg = HallucinationConfig(
        steps=config["hall_steps"],
        lr=config["hall_lr"],
        l2_weight=config["hall_l2"],
        mean_weight=config["hall_mean"],
        std_weight=config["hall_std"],
        corr_weight=config["hall_corr"],
        clamp_std=config["hall_clamp"],
        goodness_temp=config["goodness_temp"],
        node_fraction=config["hall_node_fraction"],
        node_min=config["hall_node_min"],
        corr_every_n_steps=int(config.get("hall_corr_every_n_steps", 1)),
        corr_edge_fraction=float(config.get("hall_corr_edge_fraction", 1.0)),
        corr_edge_min=int(config.get("hall_corr_edge_min", 1)),
    )
    train_neg_mode = str(config["neg_mode"]).strip().lower()
    if train_neg_mode == "self_contrastive":
        print("backprop mode does not use self_contrastive negatives; using shuffle negatives.")
        train_neg_mode = "shuffle"
    eval_mode = _resolve_mode(config.get("eval_neg_mode", "auto"), train_neg_mode)
    if eval_mode == "self_contrastive":
        eval_mode = "shuffle"

    epoch_times = []
    for epoch in tqdm(
        range(1, config["epochs"] + 1),
        desc="Benchmark",
        unit="epoch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        model.train()
        t0 = time.perf_counter()
        graphs_seen = 0
        for batch in loader:
            batch = batch.to(device)
            edge_weight = getattr(batch, "edge_weight", None)
            x = batch.x
            with _autocast_if_needed(bp_amp_enabled, bp_amp_dtype):
                h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                z_pos = global_mean_pool(h_pos, batch.batch)
                y_pos = torch.ones(z_pos.size(0), device=device)

            use_mode = _get_use_mode(
                epoch,
                train_neg_mode,
                config["neg_warmup_epochs"],
                config["neg_mix_start"],
                config["neg_mix_end"],
                config["neg_mix_ramp_epochs"],
            )
            x_neg = _make_negatives(
                model,
                x,
                batch.batch,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                edge_weight,
                use_mode,
                config["noise_std"],
                hall_cfg,
                window_len=config.get("window_len"),
                summary_dim=config.get("summary_dim", 0),
            )
            with _autocast_if_needed(bp_amp_enabled, bp_amp_dtype):
                h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                z_neg = global_mean_pool(h_neg, batch.batch)
                y_neg = torch.zeros(z_neg.size(0), device=device)

                z = torch.cat([z_pos, z_neg], dim=0)
                y = torch.cat([y_pos, y_neg], dim=0)
                logits = head(z).squeeze(1)
                loss = bce(logits, y)

            _optimizer_step(
                optim=optim,
                loss=loss,
                grad_clip=float(config["grad_clip"]),
                clip_params=optim_params,
                scaler=bp_scaler,
            )
            graphs_seen += batch.num_graphs
        _sync(device)
        dt = time.perf_counter() - t0
        epoch_times.append((dt, graphs_seen))

    # eval accuracy
    model.eval()
    head.eval()
    correct = 0
    total = 0
    gpos = []
    gneg = []
    eval_losses = []
    all_scores = []
    all_labels = []
    for batch in eval_loader:
        batch = batch.to(device)
        edge_weight = getattr(batch, "edge_weight", None)
        x = batch.x
        with torch.no_grad():
            h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
            z_pos = global_mean_pool(h_pos, batch.batch)
            y_pos = torch.ones(z_pos.size(0), device=device)

        if eval_mode == "hallucinate":
            with torch.enable_grad():
                x_neg = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    eval_mode,
                    config["noise_std"],
                    hall_cfg,
                    window_len=config.get("window_len"),
                    summary_dim=config.get("summary_dim", 0),
                )
        else:
            with torch.no_grad():
                x_neg = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    eval_mode,
                    config["noise_std"],
                    hall_cfg,
                    window_len=config.get("window_len"),
                    summary_dim=config.get("summary_dim", 0),
                )

        with torch.no_grad():
            h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
            z_neg = global_mean_pool(h_neg, batch.batch)
            y_neg = torch.zeros(z_neg.size(0), device=device)

            z = torch.cat([z_pos, z_neg], dim=0)
            y = torch.cat([y_pos, y_neg], dim=0)
            logits = head(z).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            eval_losses.append(bce(logits, y).item())
            all_scores.extend(logits.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())
            g_pos = goodness(h_pos, batch.batch, temperature=config["goodness_temp"])
            g_neg = goodness(h_neg, batch.batch, temperature=config["goodness_temp"])
            gpos.append(g_pos.mean().item())
            gneg.append(g_neg.mean().item())

    if all_scores and all_labels:
        scores_np = np.asarray(all_scores, dtype=float)
        labels_np = np.asarray(all_labels, dtype=int)
        auroc = binary_auroc(scores_np, labels_np)
        auprc = binary_auprc(scores_np, labels_np)
        brier, ece = binary_brier_and_ece(
            scores_np,
            labels_np,
            bins=int(config.get("ece_bins", 10)),
        )
    else:
        auroc = float("nan")
        auprc = float("nan")
        brier = float("nan")
        ece = float("nan")

    warm = int(config.get("timing_warmup_epochs", 0))
    usable = epoch_times[warm:] if warm < len(epoch_times) else epoch_times
    avg_time = float(np.mean([t for t, _ in usable]))
    avg_gps = float(np.mean([g / t for t, g in usable]))
    out = {
        "avg_epoch_s": avg_time,
        "graphs_per_s": avg_gps,
        "eval_acc": correct / total if total else 0.0,
        "eval_bce": float(np.mean(eval_losses)) if eval_losses else 0.0,
        "eval_g_pos": float(np.mean(gpos)) if gpos else 0.0,
        "eval_g_neg": float(np.mean(gneg)) if gneg else 0.0,
        "eval_sep": float(np.mean(gpos)) - float(np.mean(gneg)) if gpos and gneg else 0.0,
        "eval_auroc": float(auroc),
        "eval_auprc": float(auprc),
        "eval_brier": float(brier),
        "eval_ece": float(ece),
        "neg_mode_effective": train_neg_mode,
        "eval_neg_mode_effective": eval_mode,
        "eval_objective": "bce",
    }
    econ = _compute_econ_metrics_for_eval(
        model,
        eval_graphs,
        eval_dates or [],
        config,
    )
    if econ:
        out.update(econ)
    _attach_primary_metrics(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FF vs backprop training.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument(
        "--modes",
        default="ff_layerwise,ff_e2e,backprop",
        help="Comma-separated modes: ff_layerwise,ff_e2e,backprop",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})
    bench_cfg = cfg.get("benchmark", {})
    build_cfg = cfg.get("build_graphs", {})

    graphs_path = Path(train_cfg.get("graphs", "data/processed/graphs.pt"))
    try:
        payload = torch.load(graphs_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(graphs_path, map_location="cpu")
    graphs = payload["graphs"] if isinstance(payload, dict) and "graphs" in payload else payload
    graph_dates = payload.get("dates", []) if isinstance(payload, dict) else []
    if graph_dates and len(graph_dates) != len(graphs):
        graph_dates = []

    device = _choose_device(train_cfg.get("device", "auto"))
    _set_seed(int(train_cfg.get("seed", 7)))
    if train_cfg.get("torch_num_threads"):
        torch.set_num_threads(int(train_cfg["torch_num_threads"]))
    if train_cfg.get("torch_num_interop_threads"):
        torch.set_num_interop_threads(int(train_cfg["torch_num_interop_threads"]))

    feature_mode = build_cfg.get("feature_mode", "window")
    window_len = int(build_cfg.get("window", 20))
    returns_len = window_len if feature_mode in ("window", "window_plus_summary", "window_plus_summary_fund") else 1
    if feature_mode == "window_plus_summary":
        summary_dim = 5
    elif feature_mode == "window_plus_summary_fund":
        summary_dim = 10
    else:
        summary_dim = 0

    config = {
        "epochs": int(bench_cfg.get("epochs", 5)),
        "batch_size": int(bench_cfg.get("batch_size", train_cfg.get("batch_size", 16))),
        "hidden_dim": int(train_cfg.get("hidden_dim", 64)),
        "num_layers": int(train_cfg.get("num_layers", 2)),
        "dropout": float(train_cfg.get("dropout", 0.1)),
        "lr": float(train_cfg.get("lr", 1e-3)),
        "neg_mode": str(bench_cfg.get("neg_mode", train_cfg.get("neg_mode", "shuffle"))),
        "eval_neg_mode": str(bench_cfg.get("eval_neg_mode", "auto")),
        "noise_std": float(train_cfg.get("noise_std", 0.05)),
        "neg_warmup_epochs": int(train_cfg.get("neg_warmup_epochs", 0)),
        "neg_mix_start": float(train_cfg.get("neg_mix_start", 0.0)),
        "neg_mix_end": float(train_cfg.get("neg_mix_end", 0.3)),
        "neg_mix_ramp_epochs": int(train_cfg.get("neg_mix_ramp_epochs", 10)),
        "goodness_target": float(train_cfg.get("goodness_target", 1.0)),
        "goodness_temp": float(train_cfg.get("goodness_temp", 1.0)),
        "self_contrastive_temp": float(train_cfg.get("self_contrastive_temp", 0.2)),
        "self_contrastive_view_mode": str(
            train_cfg.get("self_contrastive_view_mode", "shuffle+noise")
        ),
        "self_contrastive_view_noise_std": float(
            train_cfg.get("self_contrastive_view_noise_std", train_cfg.get("noise_std", 0.05))
        ),
        "self_contrastive_eval_view_mode": str(
            bench_cfg.get(
                "self_contrastive_eval_view_mode",
                train_cfg.get("self_contrastive_eval_view_mode", "shuffle+noise"),
            )
        ),
        "self_contrastive_eval_noise_std": float(
            bench_cfg.get(
                "self_contrastive_eval_noise_std",
                train_cfg.get("self_contrastive_eval_noise_std", train_cfg.get("noise_std", 0.05)),
            )
        ),
        "self_contrastive_max_graphs": int(train_cfg.get("self_contrastive_max_graphs", 0)),
        "self_contrastive_ff_weight": float(train_cfg.get("self_contrastive_ff_weight", 0.0)),
        "self_contrastive_ff_neg_mode": str(train_cfg.get("self_contrastive_ff_neg_mode", "shuffle+noise")),
        "self_contrastive_ff_noise_std": float(
            train_cfg.get("self_contrastive_ff_noise_std", train_cfg.get("noise_std", 0.05))
        ),
        "self_contrastive_ff_target": float(
            train_cfg.get("self_contrastive_ff_target", train_cfg.get("goodness_target", 1.0))
        ),
        "distance_forward_weight": float(train_cfg.get("distance_forward_weight", 0.0)),
        "distance_forward_margin": float(train_cfg.get("distance_forward_margin", 0.15)),
        "distance_forward_max_graphs": int(train_cfg.get("distance_forward_max_graphs", 0)),
        "distance_forward_interval": int(train_cfg.get("distance_forward_interval", 1)),
        "ff_blockwise": bool(train_cfg.get("ff_blockwise", False)),
        "ff_block_size": int(train_cfg.get("ff_block_size", 2)),
        "grad_clip": float(train_cfg.get("grad_clip", 1.0)),
        "amp": bool(bench_cfg.get("ff_amp", train_cfg.get("amp", True))),
        "amp_dtype": str(bench_cfg.get("amp_dtype", train_cfg.get("amp_dtype", "float16"))),
        "fused_optimizer": bool(
            bench_cfg.get("fused_optimizer", train_cfg.get("fused_optimizer", True))
        ),
        "backprop_amp": bool(
            bench_cfg.get("backprop_amp", bench_cfg.get("ff_amp", train_cfg.get("amp", True)))
        ),
        "backprop_amp_dtype": str(
            bench_cfg.get("backprop_amp_dtype", bench_cfg.get("amp_dtype", train_cfg.get("amp_dtype", "float16")))
        ),
        "backprop_fused_optimizer": bool(
            bench_cfg.get(
                "backprop_fused_optimizer",
                bench_cfg.get("fused_optimizer", train_cfg.get("fused_optimizer", True)),
            )
        ),
        "loader_workers": int(train_cfg.get("loader_workers", 0)),
        "persistent_workers": bool(train_cfg.get("dataloader_persistent_workers", True)),
        "prefetch_factor": int(train_cfg.get("dataloader_prefetch_factor", 2)),
        "pin_memory": bool(train_cfg.get("dataloader_pin_memory", False)),
        "multiprocessing_context": str(train_cfg.get("dataloader_mp_context", "")),
        "eval_frac": float(bench_cfg.get("eval_frac", 0.2)),
        "split_mode": str(bench_cfg.get("split_mode", "chronological")),
        "walk_forward_train_frac": float(bench_cfg.get("walk_forward_train_frac", 0.6)),
        "walk_forward_eval_frac": float(bench_cfg.get("walk_forward_eval_frac", 0.2)),
        "walk_forward_step_frac": float(
            bench_cfg.get(
                "walk_forward_step_frac",
                bench_cfg.get("walk_forward_eval_frac", 0.2),
            )
        ),
        "walk_forward_min_train_graphs": int(bench_cfg.get("walk_forward_min_train_graphs", 64)),
        "walk_forward_min_eval_graphs": int(bench_cfg.get("walk_forward_min_eval_graphs", 16)),
        "walk_forward_max_folds": int(bench_cfg.get("walk_forward_max_folds", 0)),
        "seed": int(train_cfg.get("seed", 7)),
        "hall_steps": int(train_cfg.get("hallucinate_steps", 3)),
        "hall_lr": float(train_cfg.get("hallucinate_lr", 0.03)),
        "hall_l2": float(train_cfg.get("hallucinate_l2", 0.05)),
        "hall_mean": float(train_cfg.get("hallucinate_mean", 0.01)),
        "hall_std": float(train_cfg.get("hallucinate_std", 0.01)),
        "hall_corr": float(train_cfg.get("hallucinate_corr", 0.3)),
        "hall_clamp": float(train_cfg.get("hallucinate_clamp_std", 3.0)),
        "hall_node_fraction": float(train_cfg.get("hallucinate_node_fraction", 0.5)),
        "hall_node_min": int(train_cfg.get("hallucinate_node_min", 20)),
        "hall_corr_every_n_steps": int(train_cfg.get("hallucinate_corr_every_n_steps", 1)),
        "hall_corr_edge_fraction": float(train_cfg.get("hallucinate_corr_edge_fraction", 1.0)),
        "hall_corr_edge_min": int(train_cfg.get("hallucinate_corr_edge_min", 1)),
        "hall_penalty_scope": str(train_cfg.get("hallucinate_penalty_scope", "returns")),
        "hall_corr_scope": str(train_cfg.get("hallucinate_corr_scope", "returns")),
        "hall_freeze_non_return": bool(
            train_cfg.get("hallucinate_freeze_non_return_features", True)
        ),
        "timing_warmup_epochs": int(bench_cfg.get("timing_warmup_epochs", 1)),
        "calibrate_target": bool(bench_cfg.get("calibrate_target", True)),
        "calibrate_batches": int(bench_cfg.get("calibrate_batches", 0)),
        "calibrate_quantiles": int(bench_cfg.get("calibrate_quantiles", 31)),
        "ece_bins": int(bench_cfg.get("ece_bins", 10)),
        "eval_neg_modes": bench_cfg.get("eval_neg_modes", []),
        "layerwise_neg_mode": str(train_cfg.get("layerwise_neg_mode", "shuffle")),
        "layerwise_noise_std": float(train_cfg.get("layerwise_noise_std", train_cfg.get("noise_std", 0.05))),
        "window_len": int(returns_len),
        "summary_dim": int(summary_dim),
        "econ_enabled": bool(bench_cfg.get("econ_enabled", True)),
        "econ_ticker": str(bench_cfg.get("econ_ticker", "AUTO")),
        "econ_max_abs_logret": float(bench_cfg.get("econ_max_abs_logret", 0.5)),
        "econ_signal_window": int(bench_cfg.get("econ_signal_window", 126)),
        "econ_signal_quantile": float(bench_cfg.get("econ_signal_quantile", 0.5)),
        "econ_turnover_cost_bps": float(bench_cfg.get("econ_turnover_cost_bps", 0.0)),
        "econ_loader_batch_size": int(bench_cfg.get("econ_loader_batch_size", 128)),
        "econ_trading_days": int(bench_cfg.get("econ_trading_days", 252)),
    }
    config["econ_fwd_ret_1"] = None
    config["econ_ticker_effective"] = ""
    config["econ_ticker_source"] = ""
    config["econ_ticker_rows"] = 0
    if bool(config.get("econ_enabled", False)):
        prices_path = str(bench_cfg.get("econ_prices", build_cfg.get("prices", "data/processed/prices.csv")))
        try:
            ticker_eff, ticker_src, ticker_rows = resolve_price_ticker(
                prices_path=prices_path,
                requested_ticker=str(config.get("econ_ticker", "AUTO")),
                min_rows=max(32, int(config.get("econ_signal_window", 126)) // 2),
            )
            config["econ_ticker_effective"] = ticker_eff
            config["econ_ticker_source"] = ticker_src
            config["econ_ticker_rows"] = int(ticker_rows)
            config["econ_fwd_ret_1"] = load_forward_returns_from_prices(
                prices_path=prices_path,
                ticker=ticker_eff,
                max_abs_logret=float(config.get("econ_max_abs_logret", 0.5)),
            )
            print(
                "econ ticker: "
                f"requested={config.get('econ_ticker')} "
                f"effective={ticker_eff} source={ticker_src} rows={ticker_rows}"
            )
        except Exception as exc:
            config["econ_enabled"] = False
            print(f"warning: disabled econ metrics: {exc}")

    mode_overrides = bench_cfg.get("mode_overrides", {})
    if not isinstance(mode_overrides, dict):
        mode_overrides = {}

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    fold_results = []
    split_mode_cfg = str(config.get("split_mode", "chronological"))
    use_walk_forward = is_walk_forward_mode(split_mode_cfg)
    walk_forward = []
    if use_walk_forward:
        walk_forward = walk_forward_splits(
            graphs,
            train_frac=float(config.get("walk_forward_train_frac", 0.6)),
            eval_frac=float(config.get("walk_forward_eval_frac", config.get("eval_frac", 0.2))),
            step_frac=float(config.get("walk_forward_step_frac", 0.2)),
            min_train_size=int(config.get("walk_forward_min_train_graphs", 64)),
            min_eval_size=int(config.get("walk_forward_min_eval_graphs", 16)),
            max_folds=int(config.get("walk_forward_max_folds", 0)),
        )
        print(
            "walk-forward splits="
            f"{len(walk_forward)} "
            f"(train_frac={config.get('walk_forward_train_frac')}, "
            f"eval_frac={config.get('walk_forward_eval_frac')}, "
            f"step_frac={config.get('walk_forward_step_frac')})"
        )

    for mode in modes:
        cfg_mode = config.copy()
        mode_override = mode_overrides.get(mode, {})
        if isinstance(mode_override, dict):
            cfg_mode.update(mode_override)
        _warn_self_contrastive_eval_view(cfg_mode, mode)
        if use_walk_forward:
            fold_rows = []
            for fold in walk_forward:
                train_fold = fold["train_items"]
                eval_fold = fold["eval_items"]
                eval_dates_fold = []
                if graph_dates:
                    s = int(fold["eval_start"])
                    e = int(fold["eval_end"])
                    eval_dates_fold = list(graph_dates[s:e])
                cfg_fold = cfg_mode.copy()
                cfg_fold["split_mode"] = "chronological"
                if mode == "ff_layerwise":
                    res = _benchmark_ff(
                        graphs,
                        device,
                        cfg_fold,
                        layerwise=True,
                        train_graphs=train_fold,
                        eval_graphs=eval_fold,
                        eval_dates=eval_dates_fold,
                    )
                elif mode == "ff_e2e":
                    res = _benchmark_ff(
                        graphs,
                        device,
                        cfg_fold,
                        layerwise=False,
                        train_graphs=train_fold,
                        eval_graphs=eval_fold,
                        eval_dates=eval_dates_fold,
                    )
                elif mode == "backprop":
                    res = _benchmark_backprop(
                        graphs,
                        device,
                        cfg_fold,
                        train_graphs=train_fold,
                        eval_graphs=eval_fold,
                        eval_dates=eval_dates_fold,
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                res["mode"] = mode
                res["row_type"] = "fold"
                res["split_mode_effective"] = "walk_forward"
                res["fold_id"] = int(fold["fold_id"])
                res["fold_train_start_idx"] = int(fold["train_start"])
                res["fold_train_end_idx"] = int(fold["train_end"]) - 1
                res["fold_eval_start_idx"] = int(fold["eval_start"])
                res["fold_eval_end_idx"] = int(fold["eval_end"]) - 1
                if graph_dates:
                    res["fold_eval_start_date"] = str(graph_dates[int(fold["eval_start"])])
                    res["fold_eval_end_date"] = str(graph_dates[int(fold["eval_end"]) - 1])
                if isinstance(mode_override, dict):
                    for key, value in mode_override.items():
                        if key not in res:
                            res[key] = value
                fold_rows.append(res)
                fold_results.append(res)

            agg = _aggregate_fold_results(fold_rows)
            agg["mode"] = mode
            agg["row_type"] = "aggregate"
            if isinstance(mode_override, dict):
                for key, value in mode_override.items():
                    if key not in agg:
                        agg[key] = value
            results.append(agg)
        else:
            split_mode_mode = str(cfg_mode.get("split_mode", config.get("split_mode", "chronological")))
            eval_frac_mode = float(cfg_mode.get("eval_frac", config.get("eval_frac", 0.2)))
            seed_mode = int(cfg_mode.get("seed", config.get("seed", 7)))
            tr_idx, ev_idx = simple_split_indices(
                len(graphs),
                eval_frac=eval_frac_mode,
                seed=seed_mode,
                split_mode=split_mode_mode,
            )
            train_graphs_mode = [graphs[i] for i in tr_idx]
            eval_graphs_mode = [graphs[i] for i in ev_idx]
            eval_dates_mode = [graph_dates[i] for i in ev_idx] if graph_dates else []
            if mode == "ff_layerwise":
                res = _benchmark_ff(
                    graphs,
                    device,
                    cfg_mode,
                    layerwise=True,
                    train_graphs=train_graphs_mode,
                    eval_graphs=eval_graphs_mode,
                    eval_dates=eval_dates_mode,
                )
            elif mode == "ff_e2e":
                res = _benchmark_ff(
                    graphs,
                    device,
                    cfg_mode,
                    layerwise=False,
                    train_graphs=train_graphs_mode,
                    eval_graphs=eval_graphs_mode,
                    eval_dates=eval_dates_mode,
                )
            elif mode == "backprop":
                res = _benchmark_backprop(
                    graphs,
                    device,
                    cfg_mode,
                    train_graphs=train_graphs_mode,
                    eval_graphs=eval_graphs_mode,
                    eval_dates=eval_dates_mode,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
            res["mode"] = mode
            if isinstance(mode_override, dict):
                for key, value in mode_override.items():
                    if key not in res:
                        res[key] = value
            results.append(res)

    out_path = Path(bench_cfg.get("out_csv", "runs/experiments/manual/metrics/benchmark.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    keys = sorted({k for r in results for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"Wrote {out_path}")
    if use_walk_forward and fold_results:
        fold_out = str(bench_cfg.get("walk_forward_out_csv", "")).strip()
        fold_out_path = (
            Path(fold_out)
            if fold_out
            else out_path.with_name(f"{out_path.stem}_walk_forward_folds.csv")
        )
        fold_out_path.parent.mkdir(parents=True, exist_ok=True)
        fold_keys = sorted({k for r in fold_results for k in r.keys()})
        with fold_out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fold_keys)
            w.writeheader()
            for r in fold_results:
                w.writerow(r)
        print(f"Wrote {fold_out_path}")
    for r in results:
        print(r)

    plot_path = bench_cfg.get("plot_path", "runs/experiments/manual/plots/benchmark_speed_sep.png")
    bar_plot_path = bench_cfg.get("bar_plot_path", "runs/experiments/manual/plots/benchmark.png")
    try:
        import matplotlib.pyplot as plt

        xs = [r["graphs_per_s"] for r in results]
        ys = [r.get("eval_sep", 0.0) for r in results]
        labels = [r["mode"] for r in results]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys, color="#4C78A8")
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("graphs/sec")
        ax.set_ylabel("eval_gap (objective-dependent)")
        ax.set_title("Speed vs Separation")
        fig.tight_layout()
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {plot_path}")

        # Bar chart summary (avg_epoch_s + objective-aware quality)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        modes = [r["mode"] for r in results]
        avg_epoch_s = [r["avg_epoch_s"] for r in results]
        quality = []
        for r in results:
            if str(r.get("eval_objective", "")).strip().lower() == "self_contrastive":
                quality.append(r.get("eval_sc_gap", r.get("eval_sep", 0.0)))
            else:
                quality.append(r.get("eval_sep", 0.0))

        ax = axes[0]
        ax.bar(modes, avg_epoch_s, color=["#4C78A8", "#72B7B2", "#F58518"])
        ax.set_title("Avg Epoch Time (s)")
        ax.set_ylabel("seconds")

        ax = axes[1]
        ax.bar(modes, quality, color=["#4C78A8", "#72B7B2", "#F58518"])
        ax.set_title("Eval Separation / SC Gap")
        ax.set_ylabel("quality")

        fig.tight_layout()
        bar_plot_path = Path(bar_plot_path)
        bar_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(bar_plot_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {bar_plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
