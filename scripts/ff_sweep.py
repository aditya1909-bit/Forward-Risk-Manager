#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import itertools
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
from frisk.eval_metrics import ff_binary_metrics
from frisk.econ_eval import (
    evaluate_goodness_strategy,
    infer_graph_goodness,
    load_forward_returns_from_prices,
    resolve_price_ticker,
)
from frisk.splits import is_walk_forward_mode, simple_split_indices, walk_forward_splits

_GRAPH_CACHE: dict[str, tuple[list, list]] = {}
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


def _load_graphs_cached(graphs_path: str):
    key = str(Path(graphs_path).resolve())
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        payload = torch.load(Path(graphs_path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(graphs_path), map_location="cpu")
    graphs = payload["graphs"] if isinstance(payload, dict) else payload
    graph_dates = payload.get("dates", []) if isinstance(payload, dict) else []
    if graph_dates and len(graph_dates) != len(graphs):
        graph_dates = []
    out = (graphs, graph_dates)
    _GRAPH_CACHE[key] = out
    return out


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


def _sync(device: torch.device) -> None:
    sync_device(device)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _to_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _objective_rank_metric(result: dict, rank_mode: str = "objective") -> tuple[str, float]:
    mode = str(rank_mode).strip().lower()
    if mode in {"finance_first", "economic", "econ"}:
        for key in (
            "econ_sharpe_uplift",
            "econ_ann_return_uplift",
            "econ_strategy_sharpe",
            "econ_strategy_ann_return",
        ):
            val = _to_float(result.get(key))
            if np.isfinite(val):
                return key, val
        # Fallback to objective metric if economic metrics are unavailable.

    if mode not in {"objective", "finance_first", "economic", "econ"} and mode != "":
        val = _to_float(result.get(mode))
        if np.isfinite(val):
            return mode, val
        # If custom rank field is missing, fallback to objective metric.

    objective = str(result.get("eval_objective", "ff")).strip().lower()
    if objective == "self_contrastive":
        robust = _to_float(result.get("primary_eval_metric_robust"))
        if np.isfinite(robust):
            return "primary_eval_metric_robust", robust
        if np.isfinite(_to_float(result.get("eval_sc_gap"))):
            return "eval_sc_gap", _to_float(result.get("eval_sc_gap"), 0.0)
        if np.isfinite(_to_float(result.get("eval_sep"))):
            return "eval_sep", _to_float(result.get("eval_sep"), 0.0)
        if np.isfinite(_to_float(result.get("eval_auroc"))):
            return "eval_auroc", _to_float(result.get("eval_auroc"), 0.0)
        return "eval_sc_acc", _to_float(result.get("eval_sc_acc"), 0.0)
    if np.isfinite(_to_float(result.get("eval_sep"))):
        return "eval_sep", _to_float(result.get("eval_sep"), 0.0)
    if np.isfinite(_to_float(result.get("eval_auroc"))):
        return "eval_auroc", _to_float(result.get("eval_auroc"), 0.0)
    return "eval_acc", _to_float(result.get("eval_acc"), 0.0)


def _objective_track(objective: str) -> str:
    obj = str(objective).strip().lower()
    if obj == "self_contrastive":
        return "encoder"
    if obj in {"ff", "forward_forward", "forward-forward"} or obj.startswith("ff_"):
        return "critic"
    if obj in {"bce", "backprop"}:
        return "classifier"
    return "unknown"


def _uniq_values(values, *, as_int: bool = False) -> list:
    out = []
    seen = set()
    for v in values:
        if as_int:
            val = int(round(float(v)))
            key = ("i", val)
        else:
            val = float(v)
            key = ("f", round(val, 8))
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _auto_expand_combos(base: dict, sweep_cfg: dict) -> list[dict]:
    seed = int(sweep_cfg.get("auto_expand_seed", int(base.get("seed", 7)) + 97))
    max_runs = int(sweep_cfg.get("auto_expand_size", 12))
    if max_runs <= 0:
        return [{}]

    temp = float(base.get("goodness_temp", 1.0))
    target = float(base.get("goodness_target", 1.0))
    mix_end = float(base.get("neg_mix_end", 0.7))
    gate_margin = float(base.get("neg_gate_margin", 1.0))
    hall_steps = int(base.get("hall_steps", 6))
    hall_lr = float(base.get("hall_lr", 0.05))
    hall_frac = float(base.get("hall_node_fraction", 0.4))

    candidates = {
        "goodness_temp": _uniq_values(
            [_clamp(temp * 0.8, 0.05, 2.0), temp, _clamp(temp * 1.2, 0.05, 2.0)]
        ),
        "goodness_target": _uniq_values(
            [_clamp(target * 0.85, 0.1, 12.0), target, _clamp(target * 1.1, 0.1, 12.0)]
        ),
        "neg_mix_end": _uniq_values(
            [_clamp(mix_end - 0.15, 0.1, 0.98), mix_end, _clamp(mix_end + 0.1, 0.1, 0.98)]
        ),
        "neg_gate_margin": _uniq_values(
            [
                _clamp(gate_margin - 0.3, 0.05, 3.0),
                gate_margin,
                _clamp(gate_margin + 0.3, 0.05, 3.0),
            ]
        ),
        "hall_steps": _uniq_values(
            [max(2, hall_steps - 2), hall_steps, min(18, hall_steps + 2)],
            as_int=True,
        ),
        "hall_lr": _uniq_values(
            [_clamp(hall_lr * 0.7, 0.001, 0.25), hall_lr, _clamp(hall_lr * 1.4, 0.001, 0.25)]
        ),
        "hall_node_fraction": _uniq_values(
            [
                _clamp(hall_frac - 0.15, 0.1, 1.0),
                hall_frac,
                _clamp(hall_frac + 0.15, 0.1, 1.0),
            ]
        ),
    }

    keys = list(candidates.keys())
    all_combos = [
        dict(zip(keys, vals)) for vals in itertools.product(*(candidates[k] for k in keys))
    ]
    rng = random.Random(seed)
    rng.shuffle(all_combos)

    baseline = {k: base[k] for k in keys}
    picked = all_combos[:max_runs]
    if baseline not in picked:
        if picked:
            picked = [baseline] + picked[:-1]
        else:
            picked = [baseline]
    return picked


def _split_graphs(
    graphs,
    eval_frac: float,
    seed: int,
    split_mode: str = "chronological",
):
    train_idx, eval_idx = simple_split_indices(
        len(graphs),
        eval_frac=eval_frac,
        seed=seed,
        split_mode=split_mode,
    )
    train = [graphs[i] for i in train_idx]
    evals = [graphs[i] for i in eval_idx]
    return train, evals, train_idx, eval_idx


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
    forward_fn=None,
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
            forward_fn=forward_fn,
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


def _metric_value_or_none(row: dict, key: str):
    value = _to_float(row.get(key), float("nan"))
    if not np.isfinite(value):
        return None
    return float(value)


def _objective_primary_metric(row: dict) -> tuple[str, float]:
    objective = str(row.get("eval_objective", "")).strip().lower()
    if objective == "self_contrastive":
        for key in ("eval_sc_gap", "eval_sep", "eval_sc_acc", "eval_acc"):
            value = _metric_value_or_none(row, key)
            if value is not None:
                return key, value
    if objective in {"bce", "backprop"}:
        for key in ("eval_auroc", "eval_auprc", "eval_sep", "eval_acc"):
            value = _metric_value_or_none(row, key)
            if value is not None:
                return key, value
    for key in ("eval_sep", "eval_auroc", "eval_auprc", "eval_acc"):
        value = _metric_value_or_none(row, key)
        if value is not None:
            return key, value
    return "none", float("nan")


def _objective_primary_metric_robust(row: dict) -> tuple[str, float]:
    # Robust metric intentionally mirrors objective-primary metric.
    # Time-flip discrimination belongs to critic evaluation/sanity checks, not self-contrastive ranking.
    return _objective_primary_metric(row)


def _attach_primary_metrics(row: dict) -> None:
    metric_name, metric_value = _objective_primary_metric(row)
    robust_name, robust_value = _objective_primary_metric_robust(row)
    row["objective_track"] = _objective_track(row.get("eval_objective", ""))
    row["primary_eval_metric_name"] = metric_name
    row["primary_eval_metric"] = metric_value
    row["primary_eval_metric_robust_name"] = robust_name
    row["primary_eval_metric_robust"] = robust_value


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _aggregate_fold_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out: dict[str, object] = {}
    keys = sorted({k for row in rows for k in row.keys()})
    skip = {"mode", "row_type", "walk_forward_num_folds"}
    for key in keys:
        if key in skip or key.startswith("fold_"):
            continue
        vals: list[float] = []
        for row in rows:
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
    first = rows[0]
    for key in (
        "eval_objective",
        "objective_track",
        "primary_eval_metric_name",
        "primary_eval_metric_robust_name",
        "neg_mode_effective",
        "eval_neg_mode_effective",
        "risk_head_enabled_effective",
    ):
        if key in first:
            out[key] = first[key]
    out["walk_forward_num_folds"] = len(rows)
    out["split_mode_effective"] = "walk_forward"
    return out


def _compute_econ_metrics_for_eval(
    model,
    eval_graphs,
    eval_dates,
    cfg: dict,
):
    meta = {
        "econ_ticker_requested": str(cfg.get("econ_ticker", "")),
        "econ_ticker_effective": str(cfg.get("econ_ticker_effective", "")),
        "econ_ticker_source": str(cfg.get("econ_ticker_source", "")),
        "econ_ticker_rows": float(cfg.get("econ_ticker_rows", 0) or 0),
    }
    if not bool(cfg.get("econ_enabled", False)):
        return meta
    if not eval_graphs or not eval_dates:
        return meta
    fwd_ret_1 = cfg.get("econ_fwd_ret_1")
    if fwd_ret_1 is None:
        return meta
    g = infer_graph_goodness(
        model,
        eval_graphs,
        goodness_temp=float(cfg.get("goodness_temp", 1.0)),
        batch_size=int(cfg.get("econ_loader_batch_size", cfg.get("batch_size", 64))),
    )
    if g.size == 0:
        return meta
    out = evaluate_goodness_strategy(
        eval_dates,
        g,
        fwd_ret_1=fwd_ret_1,
        signal_window=int(cfg.get("econ_signal_window", 126)),
        signal_quantile=float(cfg.get("econ_signal_quantile", 0.5)),
        turnover_cost_bps=float(cfg.get("econ_turnover_cost_bps", 0.0)),
        slippage_bps=float(cfg.get("econ_slippage_bps", 0.0)),
        slippage_vol_scale=float(cfg.get("econ_slippage_vol_scale", 0.0)),
        slippage_vol_lookback=int(cfg.get("econ_slippage_vol_lookback", 21)),
        trading_days=int(cfg.get("econ_trading_days", 252)),
    )
    out.update(meta)
    return out


def _run_ff_trial(
    graphs,
    graph_dates,
    device,
    cfg,
    layerwise: bool,
    train_graphs=None,
    eval_graphs=None,
    eval_dates_override=None,
):
    if (
        train_graphs is None
        and eval_graphs is None
        and is_walk_forward_mode(str(cfg.get("split_mode", "chronological")))
    ):
        folds = walk_forward_splits(
            graphs,
            train_frac=float(cfg.get("walk_forward_train_frac", 0.6)),
            eval_frac=float(cfg.get("walk_forward_eval_frac", cfg.get("eval_frac", 0.2))),
            step_frac=float(cfg.get("walk_forward_step_frac", cfg.get("eval_frac", 0.2))),
            min_train_size=int(cfg.get("walk_forward_min_train_graphs", 64)),
            min_eval_size=int(cfg.get("walk_forward_min_eval_graphs", 16)),
            max_folds=int(cfg.get("walk_forward_max_folds", 0)),
        )
        fold_rows = []
        for fold in folds:
            cfg_fold = dict(cfg)
            cfg_fold["split_mode"] = "chronological"
            eval_dates_fold = []
            if graph_dates:
                s = int(fold["eval_start"])
                e = int(fold["eval_end"])
                eval_dates_fold = list(graph_dates[s:e])
            row = _run_ff_trial(
                graphs,
                graph_dates,
                device,
                cfg_fold,
                layerwise=layerwise,
                train_graphs=list(fold["train_items"]),
                eval_graphs=list(fold["eval_items"]),
                eval_dates_override=eval_dates_fold,
            )
            row["row_type"] = "fold"
            row["fold_id"] = int(fold["fold_id"])
            fold_rows.append(row)
        return _aggregate_fold_rows(fold_rows)

    if train_graphs is None or eval_graphs is None:
        train_graphs, eval_graphs, _, eval_idx = _split_graphs(
            graphs,
            cfg["eval_frac"],
            cfg["seed"],
            cfg.get("split_mode", "chronological"),
        )
        eval_dates = [graph_dates[i] for i in eval_idx] if graph_dates else []
    else:
        train_graphs = list(train_graphs)
        eval_graphs = list(eval_graphs)
        eval_dates = list(eval_dates_override or [])
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "shuffle": True,
        "drop_last": False,
        "num_workers": cfg["loader_workers"],
        "pin_memory": bool(cfg.get("pin_memory", False)) if device.type == "cuda" else False,
    }
    if cfg["loader_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        mp_ctx = cfg.get("multiprocessing_context", "")
        if mp_ctx:
            loader_kwargs["multiprocessing_context"] = mp_ctx
    loader = DataLoader(train_graphs, **loader_kwargs)
    eval_loader = DataLoader(eval_graphs, batch_size=cfg["batch_size"], shuffle=False)

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        conv_type=str(cfg.get("encoder_conv_type", "gcn")).strip().lower(),
        gat_heads=int(cfg.get("encoder_gat_heads", 2)),
    ).to(device)
    optim = _build_optimizer(
        model.parameters(),
        lr=cfg["lr"],
        device=device,
        use_fused=bool(cfg.get("fused_optimizer", True)),
    )

    hall_cfg = HallucinationConfig(
        steps=cfg["hall_steps"],
        lr=cfg["hall_lr"],
        l2_weight=cfg["hall_l2"],
        mean_weight=cfg["hall_mean"],
        std_weight=cfg["hall_std"],
        corr_weight=cfg["hall_corr"],
        clamp_std=cfg["hall_clamp"],
        goodness_temp=cfg["goodness_temp"],
        node_fraction=cfg["hall_node_fraction"],
        node_min=cfg["hall_node_min"],
        return_slice_len=int(cfg.get("window_len", 0)),
        penalty_scope=str(cfg.get("hall_penalty_scope", "returns")),
        corr_scope=str(cfg.get("hall_corr_scope", "returns")),
        freeze_non_return_features=bool(cfg.get("hall_freeze_non_return", True)),
        corr_every_n_steps=int(cfg.get("hall_corr_every_n_steps", 1)),
        corr_edge_fraction=float(cfg.get("hall_corr_edge_fraction", 1.0)),
        corr_edge_min=int(cfg.get("hall_corr_edge_min", 1)),
        adaptive_lr=bool(cfg.get("hall_adaptive_lr", False)),
        adaptive_lr_patience=int(cfg.get("hall_adaptive_lr_patience", 2)),
        adaptive_lr_decay=float(cfg.get("hall_adaptive_lr_decay", 0.5)),
        adaptive_lr_min=float(cfg.get("hall_adaptive_lr_min", 1e-4)),
        early_stop_on_target_hit=bool(cfg.get("hall_early_stop_on_target_hit", False)),
        target_hit_patience=int(cfg.get("hall_target_hit_patience", 1)),
        moment_mean_weight=float(cfg.get("hall_moment_mean", 0.0)),
        moment_var_weight=float(cfg.get("hall_moment_var", 0.0)),
        moment_skew_weight=float(cfg.get("hall_moment_skew", 0.0)),
        moment_scope=str(cfg.get("hall_moment_scope", "returns")),
    )
    hall_cfg_layer = HallucinationConfig(
        steps=cfg["hall_steps"],
        lr=cfg["hall_lr"],
        l2_weight=cfg["hall_l2"],
        mean_weight=cfg["layerwise_hall_mean"],
        std_weight=cfg["layerwise_hall_std"],
        corr_weight=cfg["layerwise_hall_corr"],
        clamp_std=cfg["hall_clamp"],
        goodness_temp=cfg["goodness_temp"],
        node_fraction=cfg["hall_node_fraction"],
        node_min=cfg["hall_node_min"],
        return_slice_len=int(cfg.get("window_len", 0)),
        penalty_scope=str(cfg.get("hall_penalty_scope", "returns")),
        corr_scope=str(cfg.get("hall_corr_scope", "returns")),
        freeze_non_return_features=bool(cfg.get("hall_freeze_non_return", True)),
        corr_every_n_steps=int(cfg.get("hall_corr_every_n_steps", 1)),
        corr_edge_fraction=float(cfg.get("hall_corr_edge_fraction", 1.0)),
        corr_edge_min=int(cfg.get("hall_corr_edge_min", 1)),
        adaptive_lr=bool(cfg.get("hall_adaptive_lr", False)),
        adaptive_lr_patience=int(cfg.get("hall_adaptive_lr_patience", 2)),
        adaptive_lr_decay=float(cfg.get("hall_adaptive_lr_decay", 0.5)),
        adaptive_lr_min=float(cfg.get("hall_adaptive_lr_min", 1e-4)),
        early_stop_on_target_hit=bool(cfg.get("hall_early_stop_on_target_hit", False)),
        target_hit_patience=int(cfg.get("hall_target_hit_patience", 1)),
        moment_mean_weight=float(cfg.get("hall_moment_mean", 0.0)),
        moment_var_weight=float(cfg.get("hall_moment_var", 0.0)),
        moment_skew_weight=float(cfg.get("hall_moment_skew", 0.0)),
        moment_scope=str(cfg.get("hall_moment_scope", "returns")),
    )
    sc_temp = float(cfg.get("self_contrastive_temp", 0.2))
    sc_ff_weight = max(0.0, float(cfg.get("self_contrastive_ff_weight", 0.0)))
    sc_ff_neg_mode = str(cfg.get("self_contrastive_ff_neg_mode", "shuffle+noise")).strip().lower()
    if sc_ff_neg_mode not in _NEG_AUG_MODES:
        sc_ff_neg_mode = "shuffle+noise"
    sc_ff_noise_std = max(
        0.0,
        float(cfg.get("self_contrastive_ff_noise_std", cfg.get("noise_std", 0.05))),
    )
    sc_ff_target = float(cfg.get("self_contrastive_ff_target", cfg["goodness_target"]))
    dist_weight = float(cfg.get("distance_forward_weight", 0.0))
    dist_margin = float(cfg.get("distance_forward_margin", 0.15))
    sc_max_graphs = max(0, int(cfg.get("self_contrastive_max_graphs", 0)))
    dist_max_graphs = max(0, int(cfg.get("distance_forward_max_graphs", 0)))
    dist_interval = max(1, int(cfg.get("distance_forward_interval", 1)))
    train_neg_mode = str(cfg["neg_mode"]).strip().lower()
    if layerwise and train_neg_mode == "self_contrastive":
        fallback = str(cfg.get("layerwise_neg_mode", "shuffle")).strip().lower()
        print(
            "ff_layerwise does not support self_contrastive negatives directly; "
            f"using layerwise_neg_mode={fallback!r} for training."
        )
        train_neg_mode = fallback
    eval_mode = _resolve_mode(cfg.get("eval_neg_mode", "auto"), train_neg_mode)
    ff_blockwise = bool(cfg.get("ff_blockwise", False)) and bool(layerwise)
    ff_block_size = max(1, int(cfg.get("ff_block_size", 2)))
    if ff_block_size <= 1:
        ff_blockwise = False
    ff_block_endpoints = (
        _block_endpoint_indices(len(model.layers), ff_block_size) if ff_blockwise else []
    )
    clip_params = tuple(model.parameters())
    amp_enabled = bool(cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = _parse_amp_dtype(cfg.get("amp_dtype", "float16"))
    if amp_enabled and amp_dtype == torch.bfloat16:
        bf16_supported = (
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )
        if not bf16_supported:
            amp_dtype = torch.float16
    scaler = _make_scaler(amp_enabled and amp_dtype == torch.float16)

    epoch_times = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        graphs_seen = 0
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            x = batch.x
            edge_weight = getattr(batch, "edge_weight", None)

            use_mode = _get_use_mode(
                epoch,
                train_neg_mode,
                cfg["neg_warmup_epochs"],
                cfg["neg_mix_start"],
                cfg["neg_mix_end"],
                cfg["neg_mix_ramp_epochs"],
            )
            apply_distance = dist_weight > 0 and (batch_idx % dist_interval == 0)
            step_scaler = scaler if (amp_enabled and (use_mode == "self_contrastive" or layerwise)) else None

            if layerwise:
                if ff_blockwise:
                    layer_mode = use_mode
                    if layer_mode == "self_contrastive":
                        layer_mode = "shuffle"
                    if layer_mode == "hallucinate":
                        x_neg = _make_negatives(
                            model,
                            x,
                            batch.batch,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            edge_weight,
                            layer_mode,
                            cfg["noise_std"],
                            hall_cfg_layer,
                            window_len=cfg.get("window_len"),
                            summary_dim=cfg.get("summary_dim", 0),
                        )
                    else:
                        x_neg = _make_negatives(
                            model,
                            x,
                            batch.batch,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            edge_weight,
                            cfg["layerwise_neg_mode"],
                            cfg["layerwise_noise_std"],
                            hall_cfg,
                            window_len=cfg.get("window_len"),
                            summary_dim=cfg.get("summary_dim", 0),
                        )
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        layers_pos = model(x, batch.edge_index, edge_weight=edge_weight, return_all=True)
                        layers_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True)
                    if layer_mode == "hallucinate":
                        last_idx = ff_block_endpoints[-1]
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            g_pos_probe = goodness(
                                layers_pos[last_idx], batch.batch, temperature=cfg["goodness_temp"]
                            ).mean().item()
                            g_neg_probe = goodness(
                                layers_neg[last_idx], batch.batch, temperature=cfg["goodness_temp"]
                            ).mean().item()
                        if g_neg_probe > g_pos_probe + cfg["neg_gate_margin"]:
                            x_neg = make_negative(
                                x,
                                batch.batch,
                                mode="shuffle",
                                noise_std=cfg["noise_std"],
                                window_len=cfg.get("window_len"),
                                summary_dim=cfg.get("summary_dim", 0),
                            )
                            with _autocast_if_needed(step_scaler is not None, amp_dtype):
                                layers_neg = model(
                                    x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True
                                )

                    loss = 0.0
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        for li in ff_block_endpoints:
                            g_pos = goodness(layers_pos[li], batch.batch, temperature=cfg["goodness_temp"])
                            g_neg = goodness(layers_neg[li], batch.batch, temperature=cfg["goodness_temp"])
                            loss = loss + ff_loss(
                                g_pos,
                                g_neg,
                                target=cfg["goodness_target"],
                                margin=float(cfg.get("ff_margin", 0.0)),
                                margin_weight=float(cfg.get("ff_margin_weight", 1.0)),
                            )
                    loss = loss / max(1, len(ff_block_endpoints))
                    _optimizer_step(
                        optim=optim,
                        loss=loss,
                        grad_clip=float(cfg["grad_clip"]),
                        clip_params=clip_params,
                        scaler=step_scaler,
                    )
                else:
                    x_in = x
                    for li in range(len(model.layers)):
                        layer_mode = use_mode
                        if layer_mode == "self_contrastive":
                            layer_mode = "shuffle"
                        if use_mode == "hallucinate" and li > 0:
                            layer_mode = "shuffle"
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            h_pos = model.forward_layer(x_in, batch.edge_index, edge_weight, li)
                            g_pos = goodness(h_pos, batch.batch, temperature=cfg["goodness_temp"])

                        if layer_mode == "hallucinate":
                            forward_fn = lambda x_var, li=li: model.forward_layer(
                                x_var, batch.edge_index, edge_weight, li
                            )
                            x_neg = _make_negatives(
                                model,
                                x_in,
                                batch.batch,
                                batch.edge_index,
                                getattr(batch, "edge_attr", None),
                                edge_weight,
                                layer_mode,
                                cfg["noise_std"],
                                hall_cfg_layer,
                                forward_fn=forward_fn,
                                window_len=cfg.get("window_len"),
                                summary_dim=cfg.get("summary_dim", 0),
                            )
                        else:
                            x_neg = _make_negatives(
                                model,
                                x_in,
                                batch.batch,
                                batch.edge_index,
                                getattr(batch, "edge_attr", None),
                                edge_weight,
                                cfg["layerwise_neg_mode"],
                                cfg["layerwise_noise_std"],
                                hall_cfg,
                                window_len=cfg.get("window_len"),
                                summary_dim=cfg.get("summary_dim", 0),
                            )

                        if layer_mode == "hallucinate":
                            with _autocast_if_needed(step_scaler is not None, amp_dtype):
                                h_neg_probe = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                                g_neg_probe = goodness(
                                    h_neg_probe, batch.batch, temperature=cfg["goodness_temp"]
                                ).mean().item()
                            g_pos_probe = g_pos.mean().item()
                            if g_neg_probe > g_pos_probe + cfg["neg_gate_margin"]:
                                x_neg = make_negative(
                                    x_in, batch.batch, mode="shuffle", noise_std=cfg["noise_std"]
                                )

                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            h_neg = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                            g_neg = goodness(h_neg, batch.batch, temperature=cfg["goodness_temp"])
                            loss = ff_loss(
                                g_pos,
                                g_neg,
                                target=cfg["goodness_target"],
                                margin=float(cfg.get("ff_margin", 0.0)),
                                margin_weight=float(cfg.get("ff_margin_weight", 1.0)),
                            )

                        _optimizer_step(
                            optim=optim,
                            loss=loss,
                            grad_clip=float(cfg["grad_clip"]),
                            clip_params=clip_params,
                            scaler=step_scaler,
                        )

                        x_in = h_pos.detach()
            else:
                if use_mode == "self_contrastive":
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                        x_view = _make_self_contrastive_view(
                            x,
                            batch.batch,
                            view_mode=cfg["self_contrastive_view_mode"],
                            view_noise_std=cfg["self_contrastive_view_noise_std"],
                            window_len=cfg.get("window_len"),
                            summary_dim=cfg.get("summary_dim", 0),
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
                                window_len=cfg.get("window_len"),
                                summary_dim=cfg.get("summary_dim", 0),
                            )
                            h_neg_aux = model(x_neg_aux, batch.edge_index, edge_weight=edge_weight)
                            g_pos_aux = goodness(h_pos, batch.batch, temperature=cfg["goodness_temp"])
                            g_neg_aux = goodness(h_neg_aux, batch.batch, temperature=cfg["goodness_temp"])
                            loss = loss + sc_ff_weight * ff_loss(
                                g_pos_aux,
                                g_neg_aux,
                                target=sc_ff_target,
                                margin=float(cfg.get("ff_margin", 0.0)),
                                margin_weight=float(cfg.get("ff_margin_weight", 1.0)),
                            )
                else:
                    h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                    g_pos = goodness(h_pos, batch.batch, temperature=cfg["goodness_temp"])
                    x_neg = _make_negatives(
                        model,
                        x,
                        batch.batch,
                        batch.edge_index,
                        getattr(batch, "edge_attr", None),
                        edge_weight,
                        use_mode,
                        cfg["noise_std"],
                        hall_cfg,
                        window_len=cfg.get("window_len"),
                        summary_dim=cfg.get("summary_dim", 0),
                    )

                    if use_mode == "hallucinate":
                        h_neg_probe = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                        g_neg_probe = goodness(
                            h_neg_probe, batch.batch, temperature=cfg["goodness_temp"]
                        ).mean().item()
                        g_pos_probe = g_pos.mean().item()
                        if g_neg_probe > g_pos_probe + cfg["neg_gate_margin"]:
                            x_neg = make_negative(x, batch.batch, mode="shuffle", noise_std=cfg["noise_std"])

                    h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                    g_neg = goodness(h_neg, batch.batch, temperature=cfg["goodness_temp"])
                    loss = ff_loss(
                        g_pos,
                        g_neg,
                        target=cfg["goodness_target"],
                        margin=float(cfg.get("ff_margin", 0.0)),
                        margin_weight=float(cfg.get("ff_margin_weight", 1.0)),
                    )
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
                    grad_clip=float(cfg["grad_clip"]),
                    clip_params=clip_params,
                    scaler=step_scaler,
                )

            graphs_seen += batch.num_graphs

        _sync(device)
        dt = time.perf_counter() - t0
        epoch_times.append((dt, graphs_seen))

    eval_metrics = _eval_ff_metrics(
        model,
        eval_loader,
        cfg["goodness_temp"],
        cfg["goodness_target"],
        eval_mode,
        cfg["noise_std"],
        hall_cfg,
        sc_temp=sc_temp,
        sc_view_mode=cfg.get("self_contrastive_eval_view_mode", "shuffle+noise"),
        sc_view_noise_std=cfg.get("self_contrastive_eval_noise_std"),
        window_len=cfg.get("window_len"),
        summary_dim=cfg.get("summary_dim", 0),
        ece_bins=int(cfg.get("ece_bins", 10)),
    )
    warm = int(cfg.get("timing_warmup_epochs", 0))
    usable = epoch_times[warm:] if warm < len(epoch_times) else epoch_times
    avg_time = float(np.mean([t for t, _ in usable]))
    avg_gps = float(np.mean([g / t for t, g in usable]))
    out = {
        "avg_epoch_s": avg_time,
        "graphs_per_s": avg_gps,
        "neg_mode_effective": train_neg_mode,
        "eval_neg_mode_effective": eval_mode,
    }
    out.update(eval_metrics)
    econ = _compute_econ_metrics_for_eval(
        model,
        eval_graphs,
        eval_dates,
        cfg,
    )
    if econ:
        out.update(econ)

    eval_neg_modes = cfg.get("eval_neg_modes", [])
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
                cfg["goodness_temp"],
                cfg["goodness_target"],
                mode,
                cfg["noise_std"],
                hall_cfg,
                sc_temp=sc_temp,
                sc_view_mode=cfg.get("self_contrastive_eval_view_mode", "shuffle+noise"),
                sc_view_noise_std=cfg.get("self_contrastive_eval_noise_std"),
                window_len=cfg.get("window_len"),
                summary_dim=cfg.get("summary_dim", 0),
                ece_bins=int(cfg.get("ece_bins", 10)),
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


def _run_trial_worker(args):
    (
        graphs_path,
        cfg,
        combo,
        layerwise,
        device_str,
        seed,
        worker_threads,
        worker_interop_threads,
    ) = args
    if worker_threads:
        torch.set_num_threads(int(worker_threads))
    if worker_interop_threads:
        torch.set_num_interop_threads(int(worker_interop_threads))
    _set_seed(seed)
    device = _choose_device(device_str)
    graphs, graph_dates = _load_graphs_cached(graphs_path)
    return _run_ff_trial(graphs, graph_dates, device, cfg, layerwise=layerwise)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep FF hyperparams and rank by finance/evaluation metrics."
    )
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument(
        "--section",
        default="sweep",
        help="Config section to use (default: sweep, e.g., sweep_layerwise)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})
    sweep_cfg = cfg.get(args.section, {})
    build_cfg = cfg.get("build_graphs", {})

    graphs_path = Path(train_cfg.get("graphs", "data/processed/graphs.pt"))
    device_str = str(train_cfg.get("device", "auto"))

    neg_mode_val = sweep_cfg.get("neg_mode", train_cfg.get("neg_mode", "shuffle"))
    if isinstance(neg_mode_val, list):
        neg_mode_val = neg_mode_val[0] if neg_mode_val else "shuffle"

    feature_mode = build_cfg.get("feature_mode", "window")
    window_len = int(build_cfg.get("window", 20))
    returns_len = window_len if feature_mode in ("window", "window_plus_summary", "window_plus_summary_fund") else 1
    if feature_mode == "window_plus_summary":
        summary_dim = 5
    elif feature_mode == "window_plus_summary_fund":
        summary_dim = 10
    else:
        summary_dim = 0

    base = {
        "epochs": int(sweep_cfg.get("epochs", 3)),
        "batch_size": int(sweep_cfg.get("batch_size", train_cfg.get("batch_size", 16))),
        "hidden_dim": int(train_cfg.get("hidden_dim", 64)),
        "num_layers": int(train_cfg.get("num_layers", 2)),
        "dropout": float(train_cfg.get("dropout", 0.1)),
        "encoder_conv_type": str(train_cfg.get("encoder_conv_type", "gcn")),
        "encoder_gat_heads": int(train_cfg.get("encoder_gat_heads", 2)),
        "lr": float(train_cfg.get("lr", 1e-3)),
        "neg_mode": str(neg_mode_val),
        "noise_std": float(train_cfg.get("noise_std", 0.05)),
        "neg_warmup_epochs": int(train_cfg.get("neg_warmup_epochs", 0)),
        "neg_mix_start": float(train_cfg.get("neg_mix_start", 0.0)),
        "neg_mix_end": float(train_cfg.get("neg_mix_end", 0.3)),
        "neg_mix_ramp_epochs": int(train_cfg.get("neg_mix_ramp_epochs", 10)),
        "goodness_target": float(train_cfg.get("goodness_target", 1.0)),
        "goodness_temp": float(train_cfg.get("goodness_temp", 1.0)),
        "ff_margin": float(train_cfg.get("ff_margin", 0.0)),
        "ff_margin_weight": float(train_cfg.get("ff_margin_weight", 1.0)),
        "self_contrastive_temp": float(train_cfg.get("self_contrastive_temp", 0.2)),
        "self_contrastive_view_mode": str(
            train_cfg.get("self_contrastive_view_mode", "shuffle+noise")
        ),
        "self_contrastive_view_noise_std": float(
            train_cfg.get("self_contrastive_view_noise_std", train_cfg.get("noise_std", 0.05))
        ),
        "self_contrastive_eval_view_mode": str(
            sweep_cfg.get(
                "self_contrastive_eval_view_mode",
                train_cfg.get("self_contrastive_eval_view_mode", "shuffle+noise"),
            )
        ),
        "self_contrastive_eval_noise_std": float(
            sweep_cfg.get(
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
        "amp": bool(sweep_cfg.get("ff_amp", train_cfg.get("amp", True))),
        "amp_dtype": str(sweep_cfg.get("amp_dtype", train_cfg.get("amp_dtype", "float16"))),
        "fused_optimizer": bool(
            sweep_cfg.get("fused_optimizer", train_cfg.get("fused_optimizer", True))
        ),
        "grad_clip": float(train_cfg.get("grad_clip", 1.0)),
        "loader_workers": int(train_cfg.get("loader_workers", 0)),
        "persistent_workers": bool(train_cfg.get("dataloader_persistent_workers", True)),
        "prefetch_factor": int(train_cfg.get("dataloader_prefetch_factor", 2)),
        "pin_memory": bool(train_cfg.get("dataloader_pin_memory", False)),
        "multiprocessing_context": str(train_cfg.get("dataloader_mp_context", "")),
        "eval_frac": float(sweep_cfg.get("eval_frac", 0.2)),
        "split_mode": str(sweep_cfg.get("split_mode", "chronological")),
        "walk_forward_train_frac": float(sweep_cfg.get("walk_forward_train_frac", 0.6)),
        "walk_forward_eval_frac": float(sweep_cfg.get("walk_forward_eval_frac", 0.2)),
        "walk_forward_step_frac": float(
            sweep_cfg.get(
                "walk_forward_step_frac",
                sweep_cfg.get("walk_forward_eval_frac", sweep_cfg.get("eval_frac", 0.2)),
            )
        ),
        "walk_forward_min_train_graphs": int(sweep_cfg.get("walk_forward_min_train_graphs", 64)),
        "walk_forward_min_eval_graphs": int(sweep_cfg.get("walk_forward_min_eval_graphs", 16)),
        "walk_forward_max_folds": int(sweep_cfg.get("walk_forward_max_folds", 0)),
        "seed": int(sweep_cfg.get("seed", train_cfg.get("seed", 7))),
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
        "hall_adaptive_lr": bool(train_cfg.get("hallucinate_adaptive_lr", False)),
        "hall_adaptive_lr_patience": int(train_cfg.get("hallucinate_adaptive_lr_patience", 2)),
        "hall_adaptive_lr_decay": float(train_cfg.get("hallucinate_adaptive_lr_decay", 0.5)),
        "hall_adaptive_lr_min": float(train_cfg.get("hallucinate_adaptive_lr_min", 1e-4)),
        "hall_early_stop_on_target_hit": bool(
            train_cfg.get("hallucinate_early_stop_on_target_hit", False)
        ),
        "hall_target_hit_patience": int(train_cfg.get("hallucinate_target_hit_patience", 1)),
        "hall_moment_mean": float(train_cfg.get("hallucinate_moment_mean", 0.0)),
        "hall_moment_var": float(train_cfg.get("hallucinate_moment_var", 0.0)),
        "hall_moment_skew": float(train_cfg.get("hallucinate_moment_skew", 0.0)),
        "hall_moment_scope": str(train_cfg.get("hallucinate_moment_scope", "returns")),
        "neg_gate_margin": float(train_cfg.get("neg_gate_margin", 1.0)),
        "eval_neg_mode": str(sweep_cfg.get("eval_neg_mode", "auto")),
        "eval_neg_modes": sweep_cfg.get("eval_neg_modes", []),
        "ece_bins": int(sweep_cfg.get("ece_bins", 10)),
        "timing_warmup_epochs": int(sweep_cfg.get("timing_warmup_epochs", 1)),
        "layerwise_neg_mode": str(train_cfg.get("layerwise_neg_mode", "shuffle")),
        "layerwise_noise_std": float(train_cfg.get("layerwise_noise_std", train_cfg.get("noise_std", 0.05))),
        "layerwise_hall_corr": float(train_cfg.get("layerwise_hall_corr", 0.0)),
        "layerwise_hall_mean": float(train_cfg.get("layerwise_hall_mean", train_cfg.get("hallucinate_mean", 0.01))),
        "layerwise_hall_std": float(train_cfg.get("layerwise_hall_std", train_cfg.get("hallucinate_std", 0.01))),
        "ff_blockwise": bool(train_cfg.get("ff_blockwise", False)),
        "ff_block_size": int(train_cfg.get("ff_block_size", 2)),
        "window_len": int(returns_len),
        "summary_dim": int(summary_dim),
        "econ_enabled": bool(sweep_cfg.get("econ_enabled", True)),
        "econ_ticker": str(sweep_cfg.get("econ_ticker", "AUTO")),
        "econ_max_abs_logret": float(sweep_cfg.get("econ_max_abs_logret", 0.5)),
        "econ_signal_window": int(sweep_cfg.get("econ_signal_window", 126)),
        "econ_signal_quantile": float(sweep_cfg.get("econ_signal_quantile", 0.5)),
        "econ_turnover_cost_bps": float(sweep_cfg.get("econ_turnover_cost_bps", 0.0)),
        "econ_slippage_bps": float(sweep_cfg.get("econ_slippage_bps", 0.0)),
        "econ_slippage_vol_scale": float(sweep_cfg.get("econ_slippage_vol_scale", 0.0)),
        "econ_slippage_vol_lookback": int(sweep_cfg.get("econ_slippage_vol_lookback", 21)),
        "econ_loader_batch_size": int(sweep_cfg.get("econ_loader_batch_size", 128)),
        "econ_trading_days": int(sweep_cfg.get("econ_trading_days", 252)),
    }
    base["econ_fwd_ret_1"] = None
    base["econ_ticker_effective"] = ""
    base["econ_ticker_source"] = ""
    base["econ_ticker_rows"] = 0
    if bool(base.get("econ_enabled", False)):
        prices_path = str(sweep_cfg.get("econ_prices", build_cfg.get("prices", "data/processed/prices.csv")))
        try:
            ticker_eff, ticker_src, ticker_rows = resolve_price_ticker(
                prices_path=prices_path,
                requested_ticker=str(base.get("econ_ticker", "AUTO")),
                min_rows=max(32, int(base.get("econ_signal_window", 126)) // 2),
            )
            base["econ_ticker_effective"] = ticker_eff
            base["econ_ticker_source"] = ticker_src
            base["econ_ticker_rows"] = int(ticker_rows)
            base["econ_fwd_ret_1"] = load_forward_returns_from_prices(
                prices_path=prices_path,
                ticker=ticker_eff,
                max_abs_logret=float(base.get("econ_max_abs_logret", 0.5)),
            )
            print(
                "econ ticker: "
                f"requested={base.get('econ_ticker')} "
                f"effective={ticker_eff} source={ticker_src} rows={ticker_rows}"
            )
        except Exception as exc:
            base["econ_enabled"] = False
            print(f"warning: disabled econ metrics: {exc}")

    modes = sweep_cfg.get("modes", ["ff_layerwise", "ff_e2e"])
    if isinstance(modes, str):
        modes = [m.strip() for m in modes.split(",") if m.strip()]
    mode_overrides = sweep_cfg.get("mode_overrides", {})
    if not isinstance(mode_overrides, dict):
        mode_overrides = {}

    meta_keys = {
        "epochs",
        "batch_size",
        "eval_frac",
        "split_mode",
        "walk_forward_train_frac",
        "walk_forward_eval_frac",
        "walk_forward_step_frac",
        "walk_forward_min_train_graphs",
        "walk_forward_min_eval_graphs",
        "walk_forward_max_folds",
        "out_csv",
        "modes",
        "seed",
        "max_runs",
        "timing_warmup_epochs",
        "eval_neg_mode",
        "top_k",
        "auto_expand",
        "auto_expand_size",
        "auto_expand_seed",
        "speed_weight",
        "parallel_workers",
        "parallel_backend",
        "parallel_mp_context",
        "parallel_force_cpu",
        "worker_torch_threads",
        "worker_torch_interop_threads",
        "worker_loader_workers",
        "mode_overrides",
        "rank_mode",
        "econ_enabled",
        "econ_ticker",
        "econ_prices",
        "econ_max_abs_logret",
        "econ_signal_window",
        "econ_signal_quantile",
        "econ_turnover_cost_bps",
        "econ_slippage_bps",
        "econ_slippage_vol_scale",
        "econ_slippage_vol_lookback",
        "econ_loader_batch_size",
        "econ_trading_days",
    }

    grid_keys = []
    grid_vals = []
    for k, v in sweep_cfg.items():
        if k in meta_keys:
            continue
        if isinstance(v, list):
            grid_keys.append(k)
            grid_vals.append(v)
        else:
            base[k] = v

    combos = [dict(zip(grid_keys, vals)) for vals in itertools.product(*grid_vals)] if grid_keys else [{}]
    auto_expand = bool(sweep_cfg.get("auto_expand", True))
    if auto_expand and not grid_keys:
        combos = _auto_expand_combos(base, sweep_cfg)
        print(
            f"Auto-expanded sweep grid: {len(combos)} trials "
            "(set [sweep] auto_expand=false to disable)"
        )

    max_runs = sweep_cfg.get("max_runs", None)
    if max_runs is not None:
        combos = combos[: int(max_runs)]

    parallel_workers = int(sweep_cfg.get("parallel_workers", 1))
    parallel_backend = str(sweep_cfg.get("parallel_backend", "process")).lower()
    parallel_mp_context = str(sweep_cfg.get("parallel_mp_context", "spawn"))
    parallel_force_cpu = bool(sweep_cfg.get("parallel_force_cpu", True))
    worker_threads = int(sweep_cfg.get("worker_torch_threads", 1 if parallel_workers > 1 else 0))
    worker_interop = int(sweep_cfg.get("worker_torch_interop_threads", 1 if parallel_workers > 1 else 0))
    worker_loader_workers = int(sweep_cfg.get("worker_loader_workers", base["loader_workers"]))

    if parallel_workers > 1 and device_str != "cpu":
        if parallel_force_cpu:
            print(f"Parallel sweep forcing device=cpu (was {device_str})")
            device_str = "cpu"
        else:
            print(f"Parallel sweep disabled on device={device_str}; using serial execution.")
            parallel_workers = 1

    if parallel_workers > 1:
        print(
            f"Running sweep in parallel: workers={parallel_workers}, backend={parallel_backend}, "
            f"mp_context={parallel_mp_context}"
        )
        base["loader_workers"] = worker_loader_workers

    if parallel_workers <= 1:
        # serial execution; load graphs once
        try:
            payload = torch.load(graphs_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(graphs_path, map_location="cpu")
        graphs = payload["graphs"] if isinstance(payload, dict) else payload
        graph_dates = payload.get("dates", []) if isinstance(payload, dict) else []
        if graph_dates and len(graph_dates) != len(graphs):
            graph_dates = []
        if not graphs:
            raise ValueError("No graphs found in the provided file.")
        device = _choose_device(device_str)
        if train_cfg.get("torch_num_threads"):
            torch.set_num_threads(int(train_cfg["torch_num_threads"]))
        if train_cfg.get("torch_num_interop_threads"):
            torch.set_num_interop_threads(int(train_cfg["torch_num_interop_threads"]))

    results = []
    run_idx = 0
    tasks = []
    total_trials = len(combos) * len(modes)
    pbar = tqdm(
        total=total_trials,
        desc="Sweep",
        unit="trial",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for combo in combos:
        cfg_run = base.copy()
        cfg_run.update(combo)
        for mode in modes:
            run_idx += 1
            layerwise = mode == "ff_layerwise"
            cfg_mode = cfg_run.copy()
            mode_override = mode_overrides.get(mode, {})
            if isinstance(mode_override, dict):
                cfg_mode.update(mode_override)
            if run_idx <= len(modes):
                _warn_self_contrastive_eval_view(cfg_mode, mode)
            seed = int(cfg_mode.get("seed", 7)) + run_idx
            if parallel_workers > 1:
                tasks.append(
                    (
                        str(graphs_path),
                        cfg_mode,
                        combo,
                        layerwise,
                        device_str,
                        seed,
                        worker_threads,
                        worker_interop,
                    )
                )
            else:
                _set_seed(seed)
                res = _run_ff_trial(
                    graphs,
                    graph_dates,
                    device,
                    cfg_mode,
                    layerwise=layerwise,
                )
                res["mode"] = mode
                res.update(combo)
                for k in (
                    "neg_mode",
                    "eval_neg_mode",
                    "goodness_temp",
                    "goodness_target",
                    "neg_mix_end",
                    "hall_steps",
                    "hall_lr",
                    "hall_node_fraction",
                    "layerwise_neg_mode",
                    "layerwise_noise_std",
                    "layerwise_hall_corr",
                    "layerwise_hall_mean",
                    "layerwise_hall_std",
                    "self_contrastive_max_graphs",
                    "distance_forward_max_graphs",
                    "distance_forward_interval",
                    "self_contrastive_ff_weight",
                    "self_contrastive_ff_neg_mode",
                    "self_contrastive_ff_noise_std",
                    "self_contrastive_ff_target",
                    "amp",
                    "amp_dtype",
                    "fused_optimizer",
                    "ff_blockwise",
                    "ff_block_size",
                ):
                    if k in cfg_mode:
                        res[k] = cfg_mode[k]
                results.append(res)
                pbar.update(1)

    if parallel_workers > 1 and tasks:
        if parallel_backend not in ("process", "thread", "threads"):
            raise ValueError(f"Unknown parallel_backend: {parallel_backend}")
        if parallel_backend in ("thread", "threads"):
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                for res, task in zip(ex.map(_run_trial_worker, tasks), tasks):
                    _, cfg_mode, combo, layerwise, *_ = task
                    res["mode"] = "ff_layerwise" if layerwise else "ff_e2e"
                    res.update(combo)
                    for k in (
                        "neg_mode",
                        "eval_neg_mode",
                        "goodness_temp",
                        "goodness_target",
                        "neg_mix_end",
                        "hall_steps",
                        "hall_lr",
                        "hall_node_fraction",
                        "layerwise_neg_mode",
                        "layerwise_noise_std",
                        "layerwise_hall_corr",
                        "layerwise_hall_mean",
                        "layerwise_hall_std",
                        "self_contrastive_max_graphs",
                        "distance_forward_max_graphs",
                        "distance_forward_interval",
                        "self_contrastive_ff_weight",
                        "self_contrastive_ff_neg_mode",
                        "self_contrastive_ff_noise_std",
                        "self_contrastive_ff_target",
                        "amp",
                        "amp_dtype",
                        "fused_optimizer",
                        "ff_blockwise",
                        "ff_block_size",
                    ):
                        if k in cfg_mode:
                            res[k] = cfg_mode[k]
                    results.append(res)
                    pbar.update(1)
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp

            ctx = mp.get_context(parallel_mp_context)
            with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=ctx) as ex:
                for res, task in zip(ex.map(_run_trial_worker, tasks), tasks):
                    _, cfg_mode, combo, layerwise, *_ = task
                    res["mode"] = "ff_layerwise" if layerwise else "ff_e2e"
                    res.update(combo)
                    for k in (
                        "neg_mode",
                        "eval_neg_mode",
                        "goodness_temp",
                        "goodness_target",
                        "neg_mix_end",
                        "hall_steps",
                        "hall_lr",
                        "hall_node_fraction",
                        "layerwise_neg_mode",
                        "layerwise_noise_std",
                        "layerwise_hall_corr",
                        "layerwise_hall_mean",
                        "layerwise_hall_std",
                        "self_contrastive_max_graphs",
                        "distance_forward_max_graphs",
                        "distance_forward_interval",
                        "self_contrastive_ff_weight",
                        "self_contrastive_ff_neg_mode",
                        "self_contrastive_ff_noise_std",
                        "self_contrastive_ff_target",
                        "amp",
                        "amp_dtype",
                        "fused_optimizer",
                        "ff_blockwise",
                        "ff_block_size",
                    ):
                        if k in cfg_mode:
                            res[k] = cfg_mode[k]
                    results.append(res)
                    pbar.update(1)
    pbar.close()

    if results:
        default_rank_mode = "finance_first" if bool(base.get("econ_enabled", False)) else "objective"
        rank_mode = str(sweep_cfg.get("rank_mode", default_rank_mode))
        for r in results:
            rank_metric, rank_value = _objective_rank_metric(r, rank_mode=rank_mode)
            r["rank_metric"] = rank_metric
            r["rank_value"] = rank_value
            r["rank_mode"] = rank_mode

        econ_metric_name = str(sweep_cfg.get("composite_econ_metric", "econ_sharpe_uplift"))
        sep_metric_name = str(sweep_cfg.get("composite_sep_metric", "eval_sep"))
        econ_weight = max(0.0, float(sweep_cfg.get("econ_weight", 0.45)))
        sep_weight = max(0.0, float(sweep_cfg.get("sep_weight", 0.35)))
        speed_weight = max(0.0, float(sweep_cfg.get("speed_weight", 0.20)))
        wsum = econ_weight + sep_weight + speed_weight
        if wsum <= 0:
            econ_weight, sep_weight, speed_weight = 0.45, 0.35, 0.20
            wsum = 1.0
        econ_weight /= wsum
        sep_weight /= wsum
        speed_weight /= wsum

        def _metric_or_fallback(row: dict, key: str, fallback: str | None = None) -> float:
            val = _to_float(row.get(key), float("nan"))
            if np.isfinite(val):
                return val
            if fallback:
                v2 = _to_float(row.get(fallback), float("nan"))
                if np.isfinite(v2):
                    return v2
            return _to_float(row.get("rank_value"), 0.0)

        econ_vals = [
            _metric_or_fallback(r, econ_metric_name, fallback="econ_ann_return_uplift")
            for r in results
        ]
        sep_vals = [_metric_or_fallback(r, sep_metric_name, fallback="eval_sc_gap") for r in results]
        speeds = [float(r.get("graphs_per_s", 0.0)) for r in results]

        econ_min, econ_max = min(econ_vals), max(econ_vals)
        sep_min, sep_max = min(sep_vals), max(sep_vals)
        spd_min, spd_max = min(speeds), max(speeds)
        econ_den = econ_max - econ_min
        sep_den = sep_max - sep_min
        spd_den = spd_max - spd_min
        for r in results:
            econ_raw = _metric_or_fallback(r, econ_metric_name, fallback="econ_ann_return_uplift")
            sep_raw = _metric_or_fallback(r, sep_metric_name, fallback="eval_sc_gap")
            econ_norm = 0.0 if econ_den <= 0 else (econ_raw - econ_min) / econ_den
            sep_norm = 0.0 if sep_den <= 0 else (sep_raw - sep_min) / sep_den
            speed_norm = 0.0 if spd_den <= 0 else (float(r.get("graphs_per_s", 0.0)) - spd_min) / spd_den
            r["composite_econ_metric"] = econ_metric_name
            r["composite_sep_metric"] = sep_metric_name
            r["econ_weight"] = econ_weight
            r["sep_weight"] = sep_weight
            r["speed_weight"] = speed_weight
            r["score"] = econ_weight * econ_norm + sep_weight * sep_norm + speed_weight * speed_norm

    out_path = Path(sweep_cfg.get("out_csv", "runs/experiments/manual/metrics/ff_sweep.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    keys = sorted({k for r in results for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    if results:
        best = max(results, key=lambda r: _to_float(r.get("rank_value"), float("-inf")))
        best_score = max(results, key=lambda r: r.get("score", float("-inf")))
        top_k = int(sweep_cfg.get("top_k", 10))
        ranked = sorted(
            results,
            key=lambda r: _to_float(r.get("rank_value"), float("-inf")),
            reverse=True,
        )
        ranked_score = sorted(results, key=lambda r: r.get("score", float("-inf")), reverse=True)
        print(f"Wrote {out_path}")
        print(f"Best by rank_value ({best.get('rank_metric')}): {best}")
        print(f"Best by composite score: {best_score}")
        print(f"Top {top_k} by rank_value:")
        for r in ranked[:top_k]:
            print(r)
        print(f"Top {top_k} by composite score:")
        for r in ranked_score[:top_k]:
            print(r)
    else:
        print("No sweep results produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
