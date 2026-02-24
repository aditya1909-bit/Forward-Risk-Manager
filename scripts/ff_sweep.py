#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import itertools
import math
import random
import time
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import (
    CompositeEnergyCritic,
    EnergyCritic,
    EnergyCriticEnsemble,
    GCNEncoder,
    SequenceEnergyCritic,
)
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
    infer_graph_goodness_with_uncertainty,
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
    "edge_attack",
    "sector_swap",
    "factor_hard",
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


def _build_critic(cfg: dict, hidden_dim: int, device: torch.device):
    critic_hidden_dim = max(1, int(cfg.get("critic_hidden_dim", hidden_dim)))
    critic_num_layers = max(1, int(cfg.get("critic_num_layers", 2)))
    critic_dropout = max(0.0, float(cfg.get("critic_dropout", cfg.get("dropout", 0.1))))
    critic_positive = str(cfg.get("critic_positive_activation", "softplus")).strip().lower()
    if critic_positive not in {"softplus", "square"}:
        critic_positive = "softplus"

    ensemble_size = max(1, int(cfg.get("critic_ensemble_size", 1)))
    seed_base = int(cfg.get("seed", 7))
    seed_stride = max(1, int(cfg.get("critic_ensemble_seed_stride", 1009)))
    critics = []
    for i in range(ensemble_size):
        if ensemble_size > 1:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed_base + i * seed_stride)
                member = EnergyCritic(
                    in_dim=hidden_dim,
                    hidden_dim=critic_hidden_dim,
                    num_layers=critic_num_layers,
                    dropout=critic_dropout,
                    positive_activation=critic_positive,
                )
        else:
            member = EnergyCritic(
                in_dim=hidden_dim,
                hidden_dim=critic_hidden_dim,
                num_layers=critic_num_layers,
                dropout=critic_dropout,
                positive_activation=critic_positive,
            )
        critics.append(member.to(device))

    if len(critics) == 1:
        base_critic = critics[0]
    else:
        base_critic = EnergyCriticEnsemble(critics=critics).to(device)

    seq_enabled = bool(cfg.get("sequence_critic_enabled", False))
    if not seq_enabled:
        return base_critic

    seq_hidden = max(1, int(cfg.get("sequence_critic_hidden_dim", hidden_dim)))
    seq_layers = max(1, int(cfg.get("sequence_critic_num_layers", 1)))
    seq_dropout = max(0.0, float(cfg.get("sequence_critic_dropout", 0.0)))
    seq_positive = str(cfg.get("sequence_critic_positive_activation", "softplus")).strip().lower()
    if seq_positive not in {"softplus", "square"}:
        seq_positive = "softplus"
    seq_weight = float(cfg.get("sequence_critic_weight", 0.0))
    seq_critic = SequenceEnergyCritic(
        in_dim=hidden_dim,
        hidden_dim=seq_hidden,
        num_layers=seq_layers,
        dropout=seq_dropout,
        positive_activation=seq_positive,
    ).to(device)
    return CompositeEnergyCritic(
        base_critic=base_critic,
        sequence_critic=seq_critic,
        sequence_weight=seq_weight,
    ).to(device)


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


def _parse_mode_list(value) -> list[str]:
    if value is None:
        return []
    items: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in text.split(",")]
        items = [p for p in parts if p]
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            items.extend(_parse_mode_list(v))
    else:
        text = str(value).strip()
        if text:
            items = [text]
    out: list[str] = []
    seen = set()
    for item in items:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _mode_key(mode: str) -> str:
    return str(mode).strip().lower().replace("+", "_plus_")


def _attach_eval_neg_aggregate_metrics(
    row: dict,
    *,
    include_base: bool = True,
    agg_mode: str = "mean",
) -> None:
    agg = str(agg_mode).strip().lower() or "mean"
    if agg not in {"mean", "median", "min", "max"}:
        agg = "mean"
    base_mode = str(row.get("eval_neg_mode_effective", "")).strip().lower()
    reported_modes = _parse_mode_list(row.get("eval_neg_modes_reported", ""))
    if not reported_modes:
        reported_modes = _parse_mode_list(row.get("eval_neg_modes", ""))

    modes_aggregated: list[str] = []
    seen_modes = set()
    if include_base and base_mode:
        modes_aggregated.append(base_mode)
        seen_modes.add(base_mode)
    for m in reported_modes:
        if m in seen_modes:
            continue
        modes_aggregated.append(m)
        seen_modes.add(m)

    metric_names = ("sep", "auroc", "auprc", "acc", "brier", "ece")
    max_count = 0
    for metric in metric_names:
        vals: list[float] = []
        if include_base:
            base_val = _to_float(row.get(f"eval_{metric}"), float("nan"))
            if np.isfinite(base_val):
                vals.append(base_val)
        for mode in reported_modes:
            if include_base and mode == base_mode:
                continue
            v = _to_float(row.get(f"eval_{_mode_key(mode)}_{metric}"), float("nan"))
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        row[f"eval_{metric}_agg_mean"] = float(np.mean(arr))
        row[f"eval_{metric}_agg_median"] = float(np.median(arr))
        row[f"eval_{metric}_agg_min"] = float(np.min(arr))
        row[f"eval_{metric}_agg_max"] = float(np.max(arr))
        row[f"eval_{metric}_agg_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        if agg == "median":
            row[f"eval_{metric}_agg"] = row[f"eval_{metric}_agg_median"]
        elif agg == "min":
            row[f"eval_{metric}_agg"] = row[f"eval_{metric}_agg_min"]
        elif agg == "max":
            row[f"eval_{metric}_agg"] = row[f"eval_{metric}_agg_max"]
        else:
            row[f"eval_{metric}_agg"] = row[f"eval_{metric}_agg_mean"]
        max_count = max(max_count, int(arr.size))

    row["eval_neg_aggregate_mode"] = agg
    row["eval_neg_aggregate_include_base"] = int(bool(include_base))
    row["eval_neg_aggregate_count"] = int(max_count)
    row["eval_neg_modes_aggregated"] = ",".join(modes_aggregated)


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
    critic=None,
    forward_fn=None,
    window_len: int | None = None,
    summary_dim: int = 0,
):
    if use_mode == "self_contrastive":
        use_mode = "shuffle"
    if use_mode == "edge_attack":
        out = make_negative(
            x,
            batch,
            mode="shuffle+noise",
            noise_std=max(0.0, float(noise_std)),
            window_len=window_len,
            summary_dim=summary_dim,
        )
        if out.numel() == 0:
            return out
        hub_frac = float(getattr(hall_cfg, "adversarial_hub_fraction", 0.2))
        noise_mult = float(getattr(hall_cfg, "adversarial_feature_noise_mult", 3.0))
        flip_prob = float(getattr(hall_cfg, "adversarial_timeflip_prob", 0.5))
        hub_frac = max(0.0, min(1.0, hub_frac))
        noise_mult = max(1.0, noise_mult)
        flip_prob = max(0.0, min(1.0, flip_prob))
        if hub_frac <= 0:
            return out
        row = edge_index[0]
        col = edge_index[1]
        deg = torch.zeros(out.size(0), device=out.device, dtype=out.dtype)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=out.dtype))
        deg.scatter_add_(0, col, torch.ones_like(col, dtype=out.dtype))
        out_adv = out.clone()
        for gid in batch.unique():
            idx = (batch == gid).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            k = max(1, int(idx.numel() * hub_frac))
            k = min(int(idx.numel()), int(k))
            if k <= 0:
                continue
            local_deg = deg.index_select(0, idx)
            top_local = torch.topk(local_deg, k=k, largest=True).indices
            hub_idx = idx.index_select(0, top_local)
            out_adv.index_add_(
                0,
                hub_idx,
                (max(0.0, float(noise_std)) * noise_mult) * torch.randn_like(out_adv.index_select(0, hub_idx)),
            )
            if flip_prob > 0 and torch.rand((), device=out.device).item() < flip_prob:
                if window_len is not None and int(window_len) > 1 and out_adv.size(1) >= int(window_len):
                    wlen = int(window_len)
                    out_adv[hub_idx, :wlen] = torch.flip(out_adv[hub_idx, :wlen], dims=[1])
                elif out_adv.size(1) > 1:
                    out_adv[hub_idx] = torch.flip(out_adv[hub_idx], dims=[1])
        return out_adv
    if use_mode == "hallucinate":
        if x.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                return hallucinate_negative(
                    model,
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    hall_cfg,
                    edge_weight=edge_weight,
                    forward_fn=forward_fn,
                    critic=critic,
                )
        return hallucinate_negative(
            model,
            x,
            edge_index,
            edge_attr,
            batch,
            hall_cfg,
            edge_weight=edge_weight,
            forward_fn=forward_fn,
            critic=critic,
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
        "sector_swap",
        "factor_hard",
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
    critic,
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
    pos_cache: dict | None = None,
    return_pos_cache: bool = False,
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
        metrics = {
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
        if return_pos_cache:
            return metrics, None
        return metrics

    model.eval()
    gpos = []
    gneg = []
    gpos_all = []
    gneg_all = []
    acc_num = 0
    acc_den = 0
    cached_g_pos_batches = []
    use_cached_gpos_all = False
    if isinstance(pos_cache, dict):
        cached = pos_cache.get("g_pos_batches")
        if isinstance(cached, list):
            cached_g_pos_batches = cached
        cached_all = pos_cache.get("g_pos_all")
        if isinstance(cached_all, list):
            gpos_all = list(cached_all)
            use_cached_gpos_all = True
    g_pos_batches_out = [] if return_pos_cache and not cached_g_pos_batches else cached_g_pos_batches
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(next(model.parameters()).device)
        x = batch.x
        edge_weight = getattr(batch, "edge_weight", None)
        g_pos = None
        if batch_idx < len(cached_g_pos_batches):
            try:
                g_pos = cached_g_pos_batches[batch_idx].to(x.device, non_blocking=True)
            except Exception:
                g_pos = None
        if g_pos is None:
            with torch.no_grad():
                h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)
            if return_pos_cache:
                g_pos_batches_out.append(g_pos.detach().cpu())

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
                    critic=critic,
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
                    critic=critic,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )

        with torch.no_grad():
            h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
            g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp, critic=critic)
            pred_pos = (g_pos > goodness_target)
            pred_neg = (g_neg <= goodness_target)
            acc_num += (pred_pos.sum() + pred_neg.sum()).item()
            acc_den += 2 * g_pos.numel()
            gpos.append(g_pos.mean().item())
            gneg.append(g_neg.mean().item())
            if not use_cached_gpos_all:
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
    metrics = {
        "eval_objective": "ff",
        "eval_g_pos": pos_mean,
        "eval_g_neg": neg_mean,
        "eval_sep": pos_mean - neg_mean,
        "eval_acc": float(acc),
        **cls_metrics,
    }
    if return_pos_cache:
        cache = {
            "g_pos_batches": g_pos_batches_out,
            "g_pos_all": list(gpos_all),
        }
        return metrics, cache
    return metrics


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
    objective = str(row.get("eval_objective", "")).strip().lower()
    if objective == "self_contrastive":
        for key in ("eval_sc_gap", "eval_sep_agg_min", "eval_sep_agg_mean", "eval_sep"):
            value = _metric_value_or_none(row, key)
            if value is not None:
                return key, value
        return _objective_primary_metric(row)
    if objective in {"bce", "backprop"}:
        for key in ("eval_auroc_agg_min", "eval_auroc_agg_mean", "eval_auroc", "eval_auprc"):
            value = _metric_value_or_none(row, key)
            if value is not None:
                return key, value
        return _objective_primary_metric(row)
    for key in ("eval_sep_agg_min", "eval_sep_agg_mean", "eval_sep", "eval_auroc_agg_min", "eval_auroc"):
        value = _metric_value_or_none(row, key)
        if value is not None:
            return key, value
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


def _aggregate_numeric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out: dict[str, object] = {}
    keys = sorted({k for row in rows for k in row.keys()})
    for key in keys:
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
            out[f"{key}_min"] = float(np.min(vals))
            out[f"{key}_max"] = float(np.max(vals))
    first = rows[0]
    for key, value in first.items():
        if isinstance(value, str):
            out.setdefault(key, value)
    return out


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
            out[f"{key}_min"] = float(np.min(vals))
            out[f"{key}_max"] = float(np.max(vals))
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
    econ_payloads = []
    for row in rows:
        payload = row.get("__econ_payloads")
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            g = entry.get("goodness")
            d = entry.get("dates")
            if isinstance(g, list) and isinstance(d, list) and len(g) == len(d):
                econ_payloads.append({"goodness": list(g), "dates": list(d)})
    if econ_payloads:
        out["__econ_payloads"] = econ_payloads
    return out


def _compute_econ_metrics_for_eval(
    model,
    critic,
    eval_graphs,
    eval_dates,
    cfg: dict,
    goodness_values=None,
    goodness_uncertainty=None,
):
    meta = {
        "econ_ticker_requested": str(cfg.get("econ_ticker", "")),
        "econ_ticker_effective": str(cfg.get("econ_ticker_effective", "")),
        "econ_ticker_source": str(cfg.get("econ_ticker_source", "")),
        "econ_ticker_rows": float(cfg.get("econ_ticker_rows", 0) or 0),
    }
    if not bool(cfg.get("econ_enabled", False)):
        return meta
    if not eval_dates:
        return meta
    if goodness_values is None and not eval_graphs:
        return meta
    fwd_ret_1 = cfg.get("econ_fwd_ret_1")
    if fwd_ret_1 is None:
        return meta
    if goodness_values is not None:
        g = np.nan_to_num(
            np.asarray(goodness_values, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        g_unc = (
            None
            if goodness_uncertainty is None
            else np.asarray(goodness_uncertainty, dtype=float)
        )
    else:
        g, g_unc = infer_graph_goodness_with_uncertainty(
            model,
            eval_graphs,
            goodness_temp=float(cfg.get("goodness_temp", 1.0)),
            batch_size=int(cfg.get("econ_loader_batch_size", cfg.get("batch_size", 64))),
            critic=critic,
        )
    if g.size == 0:
        return meta
    if int(g.shape[0]) != int(len(eval_dates)):
        return meta
    risk_signal = None
    if bool(cfg.get("econ_regime_gate_enabled", False)):
        rgw = max(10, int(cfg.get("econ_regime_gate_window", 63)))
        g_ser = np.asarray(g, dtype=float)
        if g_ser.size > 0:
            g_roll_mean = np.asarray(
                pd.Series(g_ser).rolling(rgw, min_periods=max(5, rgw // 3)).mean(),
                dtype=float,
            )
            g_roll_std = np.asarray(
                pd.Series(g_ser).rolling(rgw, min_periods=max(5, rgw // 3)).std(),
                dtype=float,
            )
            risk_signal = (g_roll_mean - g_ser) / (g_roll_std + 1e-8)
            if g_unc is not None and g_unc.shape[0] == risk_signal.shape[0]:
                risk_signal = risk_signal + np.nan_to_num(np.asarray(g_unc, dtype=float), nan=0.0)
    out = evaluate_goodness_strategy(
        eval_dates,
        g,
        fwd_ret_1=fwd_ret_1,
        signal_window=int(cfg.get("econ_signal_window", 126)),
        signal_quantile=float(cfg.get("econ_signal_quantile", 0.5)),
        signal_polarity=str(cfg.get("econ_signal_polarity", "high")),
        turnover_cost_bps=float(cfg.get("econ_turnover_cost_bps", 0.0)),
        slippage_bps=float(cfg.get("econ_slippage_bps", 0.0)),
        slippage_vol_scale=float(cfg.get("econ_slippage_vol_scale", 0.0)),
        slippage_vol_lookback=int(cfg.get("econ_slippage_vol_lookback", 21)),
        trading_days=int(cfg.get("econ_trading_days", 252)),
        regime_gate_enabled=bool(cfg.get("econ_regime_gate_enabled", False)),
        regime_gate_window=int(cfg.get("econ_regime_gate_window", 63)),
        regime_confidence_temp=float(cfg.get("econ_regime_confidence_temp", 1.0)),
        regime_neutral_exposure=float(cfg.get("econ_regime_neutral_exposure", 0.0)),
        regime_min_confidence=float(cfg.get("econ_regime_min_confidence", 0.0)),
        goodness_uncertainty=g_unc,
        regime_uncertainty_scale=float(cfg.get("econ_regime_uncertainty_scale", 0.0)),
        risk_signal=risk_signal,
        regime_risk_scale=float(cfg.get("econ_regime_risk_scale", 0.0)),
        regime_thresholding_enabled=bool(cfg.get("econ_regime_thresholding_enabled", True)),
        regime_threshold_window=int(
            cfg.get("econ_regime_threshold_window", cfg.get("econ_signal_window", 126))
        ),
        regime_threshold_quantile=float(
            cfg.get("econ_regime_threshold_quantile", cfg.get("econ_signal_quantile", 0.5))
        ),
        regime_vol_window=int(cfg.get("econ_regime_vol_window", 21)),
        regime_low_quantile=float(cfg.get("econ_regime_low_quantile", 0.33)),
        regime_high_quantile=float(cfg.get("econ_regime_high_quantile", 0.67)),
    )
    out.update(meta)
    return out


def _build_econ_payload(
    model,
    critic,
    eval_graphs,
    eval_dates,
    cfg: dict,
    eval_pos_cache: dict | None,
):
    g_for_econ = None
    payloads = []
    if isinstance(eval_pos_cache, dict):
        gvals = eval_pos_cache.get("g_pos_all")
        if isinstance(gvals, list):
            g_for_econ = [float(_to_float(v, float("nan"))) for v in gvals]
    if eval_dates and isinstance(g_for_econ, list) and len(g_for_econ) == len(eval_dates):
        payloads.append({"goodness": list(g_for_econ), "dates": list(eval_dates)})

    need_fallback = (
        g_for_econ is None
        and model is not None
        and bool(eval_graphs)
        and bool(eval_dates)
        and (
            bool(cfg.get("econ_enabled", False))
            or bool(cfg.get("econ_payload_enabled", False))
        )
    )
    if need_fallback:
        g_np, _ = infer_graph_goodness_with_uncertainty(
            model,
            eval_graphs,
            goodness_temp=float(cfg.get("goodness_temp", 1.0)),
            batch_size=int(cfg.get("econ_loader_batch_size", cfg.get("batch_size", 64))),
            critic=critic,
        )
        if g_np.size > 0 and int(g_np.shape[0]) == int(len(eval_dates)):
            g_for_econ = [float(_to_float(v, float("nan"))) for v in g_np.tolist()]
            payloads = [{"goodness": list(g_for_econ), "dates": list(eval_dates)}]
    return g_for_econ, payloads


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
    train_shuffle = True
    if bool(cfg.get("sequence_critic_enabled", False)) and bool(
        cfg.get("sequence_critic_force_chrono", True)
    ):
        train_shuffle = False
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "shuffle": train_shuffle,
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
    eval_batch_size = max(1, int(cfg.get("eval_batch_size", cfg["batch_size"])))
    eval_loader_workers = max(0, int(cfg.get("eval_loader_workers", cfg.get("loader_workers", 0))))
    eval_loader_kwargs = {
        "batch_size": eval_batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": eval_loader_workers,
        "pin_memory": bool(cfg.get("eval_pin_memory", cfg.get("pin_memory", False)))
        if device.type == "cuda"
        else False,
    }
    if eval_loader_workers > 0:
        eval_loader_kwargs["persistent_workers"] = bool(
            cfg.get("eval_persistent_workers", cfg.get("persistent_workers", True))
        )
        eval_loader_kwargs["prefetch_factor"] = int(
            cfg.get("eval_prefetch_factor", cfg.get("prefetch_factor", 2))
        )
        eval_mp_ctx = str(
            cfg.get("eval_multiprocessing_context", cfg.get("multiprocessing_context", ""))
        ).strip()
        if eval_mp_ctx:
            eval_loader_kwargs["multiprocessing_context"] = eval_mp_ctx
    eval_loader = DataLoader(eval_graphs, **eval_loader_kwargs)

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        conv_type=str(cfg.get("encoder_conv_type", "gcn")).strip().lower(),
        gat_heads=int(cfg.get("encoder_gat_heads", 2)),
        residual_edge_enabled=bool(cfg.get("residual_edge_weight_enabled", False)),
        residual_edge_hidden_dim=int(cfg.get("residual_edge_hidden_dim", 32)),
        residual_edge_max_delta=float(cfg.get("residual_edge_max_delta", 0.25)),
        residual_edge_detach_features=bool(cfg.get("residual_edge_detach_features", True)),
    ).to(device)
    critic = _build_critic(cfg, hidden_dim=int(cfg["hidden_dim"]), device=device)
    optim = _build_optimizer(
        list(model.parameters()) + list(critic.parameters()),
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
        adversarial_hub_fraction=float(cfg.get("hall_attack_hub_fraction", 0.2)),
        adversarial_feature_noise_mult=float(cfg.get("hall_attack_noise_mult", 3.0)),
        adversarial_timeflip_prob=float(cfg.get("hall_attack_timeflip_prob", 0.5)),
        adversarial_edge_drop_prob=float(cfg.get("hall_attack_edge_drop_prob", 0.2)),
        adversarial_sign_flip_prob=float(cfg.get("hall_attack_sign_flip_prob", 0.2)),
        adversarial_hub_weight_scale=float(cfg.get("hall_attack_hub_weight_scale", 0.5)),
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
        adversarial_hub_fraction=float(cfg.get("hall_attack_hub_fraction", 0.2)),
        adversarial_feature_noise_mult=float(cfg.get("hall_attack_noise_mult", 3.0)),
        adversarial_timeflip_prob=float(cfg.get("hall_attack_timeflip_prob", 0.5)),
        adversarial_edge_drop_prob=float(cfg.get("hall_attack_edge_drop_prob", 0.2)),
        adversarial_sign_flip_prob=float(cfg.get("hall_attack_sign_flip_prob", 0.2)),
        adversarial_hub_weight_scale=float(cfg.get("hall_attack_hub_weight_scale", 0.5)),
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
    clip_params = tuple(list(model.parameters()) + list(critic.parameters()))
    amp_enabled = bool(cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = _parse_amp_dtype(cfg.get("amp_dtype", "float16"))
    if amp_enabled and amp_dtype == torch.bfloat16:
        bf16_supported = (
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )
        if not bf16_supported:
            amp_dtype = torch.float16
    scaler = _make_scaler(amp_enabled and amp_dtype == torch.float16)

    early_stop_enabled = bool(cfg.get("early_stop_enabled", False))
    early_stop_min_epochs = max(1, int(cfg.get("early_stop_min_epochs", cfg["epochs"])))
    early_stop_patience = max(1, int(cfg.get("early_stop_patience", 2)))
    early_stop_min_delta = float(cfg.get("early_stop_min_delta", 1e-4))
    early_stop_eval_every = max(1, int(cfg.get("early_stop_eval_every", 1)))
    early_stop_eval_graphs = int(cfg.get("early_stop_eval_graphs", 128))
    early_stop_rank_mode = str(cfg.get("early_stop_rank_mode", "objective")).strip() or "objective"
    early_stop_triggered = False
    early_stop_epoch = int(cfg["epochs"])
    early_stop_metric_name = ""
    early_stop_best = float("-inf")
    early_stop_bad_epochs = 0
    early_eval_loader = None
    if early_stop_enabled:
        if int(cfg["epochs"]) <= early_stop_min_epochs or len(eval_graphs) < 2:
            early_stop_enabled = False
        else:
            if early_stop_eval_graphs > 0 and early_stop_eval_graphs < len(eval_graphs):
                early_graphs = list(eval_graphs[:early_stop_eval_graphs])
                early_eval_loader = DataLoader(early_graphs, **eval_loader_kwargs)
            else:
                early_eval_loader = eval_loader

    epoch_times = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        critic.train()
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
            step_scaler = scaler if amp_enabled else None

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
                            critic=critic,
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
                            critic=critic,
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
                                layers_pos[last_idx],
                                batch.batch,
                                temperature=cfg["goodness_temp"],
                                critic=critic,
                            ).mean().item()
                            g_neg_probe = goodness(
                                layers_neg[last_idx],
                                batch.batch,
                                temperature=cfg["goodness_temp"],
                                critic=critic,
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
                            g_pos = goodness(
                                layers_pos[li],
                                batch.batch,
                                temperature=cfg["goodness_temp"],
                                critic=critic,
                            )
                            g_neg = goodness(
                                layers_neg[li],
                                batch.batch,
                                temperature=cfg["goodness_temp"],
                                critic=critic,
                            )
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
                            g_pos = goodness(
                                h_pos,
                                batch.batch,
                                temperature=cfg["goodness_temp"],
                                critic=critic,
                            )

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
                                critic=critic,
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
                                critic=critic,
                                window_len=cfg.get("window_len"),
                                summary_dim=cfg.get("summary_dim", 0),
                            )

                        if layer_mode == "hallucinate":
                            with _autocast_if_needed(step_scaler is not None, amp_dtype):
                                h_neg_probe = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                                g_neg_probe = goodness(
                                    h_neg_probe,
                                    batch.batch,
                                    temperature=cfg["goodness_temp"],
                                    critic=critic,
                                ).mean().item()
                            g_pos_probe = g_pos.mean().item()
                            if g_neg_probe > g_pos_probe + cfg["neg_gate_margin"]:
                                x_neg = make_negative(
                                    x_in, batch.batch, mode="shuffle", noise_std=cfg["noise_std"]
                                )

                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            h_neg = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                            g_neg = goodness(
                                h_neg,
                                batch.batch,
                                temperature=cfg["goodness_temp"],
                                critic=critic,
                            )
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
                            g_pos_aux = goodness(
                                h_pos,
                                batch.batch,
                                temperature=cfg["goodness_temp"],
                                critic=critic,
                            )
                            g_neg_aux = goodness(
                                h_neg_aux,
                                batch.batch,
                                temperature=cfg["goodness_temp"],
                                critic=critic,
                            )
                            loss = loss + sc_ff_weight * ff_loss(
                                g_pos_aux,
                                g_neg_aux,
                                target=sc_ff_target,
                                margin=float(cfg.get("ff_margin", 0.0)),
                                margin_weight=float(cfg.get("ff_margin_weight", 1.0)),
                            )
                else:
                    h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                    g_pos = goodness(
                        h_pos,
                        batch.batch,
                        temperature=cfg["goodness_temp"],
                        critic=critic,
                    )
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
                        critic=critic,
                        window_len=cfg.get("window_len"),
                        summary_dim=cfg.get("summary_dim", 0),
                    )

                    if use_mode == "hallucinate":
                        h_neg_probe = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                        g_neg_probe = goodness(
                            h_neg_probe,
                            batch.batch,
                            temperature=cfg["goodness_temp"],
                            critic=critic,
                        ).mean().item()
                        g_pos_probe = g_pos.mean().item()
                        if g_neg_probe > g_pos_probe + cfg["neg_gate_margin"]:
                            x_neg = make_negative(x, batch.batch, mode="shuffle", noise_std=cfg["noise_std"])

                    h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                    g_neg = goodness(
                        h_neg,
                        batch.batch,
                        temperature=cfg["goodness_temp"],
                        critic=critic,
                    )
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
        if (
            early_stop_enabled
            and early_eval_loader is not None
            and epoch >= early_stop_min_epochs
            and (epoch % early_stop_eval_every == 0)
        ):
            early_metrics = _eval_ff_metrics(
                model,
                critic,
                early_eval_loader,
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
            metric_name, metric_val = _objective_rank_metric(
                early_metrics, rank_mode=early_stop_rank_mode
            )
            early_stop_metric_name = metric_name
            if np.isfinite(_to_float(metric_val, float("nan"))) and (
                metric_val > early_stop_best + early_stop_min_delta
            ):
                early_stop_best = float(metric_val)
                early_stop_bad_epochs = 0
            else:
                early_stop_bad_epochs += 1
                if early_stop_bad_epochs >= early_stop_patience:
                    early_stop_triggered = True
                    early_stop_epoch = int(epoch)
                    break

    eval_out = _eval_ff_metrics(
        model,
        critic,
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
        return_pos_cache=True,
    )
    if isinstance(eval_out, tuple):
        eval_metrics, eval_pos_cache = eval_out
    else:
        eval_metrics, eval_pos_cache = eval_out, None
    warm = int(cfg.get("timing_warmup_epochs", 0))
    usable = epoch_times[warm:] if warm < len(epoch_times) else epoch_times
    avg_time = float(np.mean([t for t, _ in usable]))
    avg_gps = float(np.mean([g / t for t, g in usable]))
    out = {
        "avg_epoch_s": avg_time,
        "graphs_per_s": avg_gps,
        "epochs_target": int(cfg["epochs"]),
        "epochs_run": int(len(epoch_times)),
        "neg_mode_effective": train_neg_mode,
        "eval_neg_mode_effective": eval_mode,
        "early_stop_enabled": int(bool(early_stop_enabled)),
        "early_stop_triggered": int(bool(early_stop_triggered)),
        "early_stop_epoch": int(early_stop_epoch),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_epochs": int(early_stop_min_epochs),
        "early_stop_eval_every": int(early_stop_eval_every),
        "early_stop_rank_mode": early_stop_rank_mode,
        "early_stop_metric_name": early_stop_metric_name,
        "early_stop_best_metric": float(early_stop_best)
        if np.isfinite(_to_float(early_stop_best, float("nan")))
        else float("nan"),
    }
    out.update(eval_metrics)
    g_for_econ, econ_payloads = _build_econ_payload(
        model,
        critic,
        eval_graphs,
        eval_dates,
        cfg,
        eval_pos_cache,
    )
    if econ_payloads:
        out["__econ_payloads"] = econ_payloads
    econ = _compute_econ_metrics_for_eval(
        model,
        critic,
        eval_graphs,
        eval_dates,
        cfg,
        goodness_values=g_for_econ,
    )
    if econ:
        out.update(econ)

    eval_neg_modes_requested = cfg.get("eval_neg_modes", [])
    if isinstance(eval_neg_modes_requested, str):
        eval_neg_modes_requested = [
            m.strip() for m in eval_neg_modes_requested.split(",") if m.strip()
        ]
    requested_modes = [
        str(m).strip().lower() for m in eval_neg_modes_requested if str(m).strip()
    ]
    eval_neg_modes_enabled = bool(cfg.get("eval_neg_modes_enabled", True))
    extra_modes = list(requested_modes) if eval_neg_modes_enabled else []
    out["eval_neg_modes_enabled"] = int(bool(eval_neg_modes_enabled))
    out["eval_neg_modes_configured"] = ",".join(requested_modes)
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
                critic,
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
                pos_cache=eval_pos_cache,
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
    elif requested_modes and not eval_neg_modes_enabled:
        out["eval_neg_modes_skipped"] = ",".join(requested_modes)
    _attach_eval_neg_aggregate_metrics(
        out,
        include_base=bool(cfg.get("eval_neg_aggregate_include_base", True)),
        agg_mode=str(cfg.get("eval_neg_aggregate", "mean")),
    )
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
        trial_idx,
    ) = args
    if worker_threads:
        torch.set_num_threads(int(worker_threads))
    if worker_interop_threads:
        torch.set_num_interop_threads(int(worker_interop_threads))
    _set_seed(seed)
    device = _choose_device(device_str)
    graphs, graph_dates = _load_graphs_cached(graphs_path)
    out = _run_ff_trial(graphs, graph_dates, device, cfg, layerwise=layerwise)
    out["__trial_idx"] = int(trial_idx)
    return out


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
        "critic_hidden_dim": int(train_cfg.get("critic_hidden_dim", train_cfg.get("hidden_dim", 64))),
        "critic_num_layers": int(train_cfg.get("critic_num_layers", 2)),
        "critic_dropout": float(train_cfg.get("critic_dropout", train_cfg.get("dropout", 0.1))),
        "critic_positive_activation": str(train_cfg.get("critic_positive_activation", "softplus")),
        "critic_ensemble_size": int(train_cfg.get("critic_ensemble_size", 1)),
        "critic_ensemble_seed_stride": int(train_cfg.get("critic_ensemble_seed_stride", 1009)),
        "sequence_critic_enabled": bool(train_cfg.get("sequence_critic_enabled", False)),
        "sequence_critic_weight": float(train_cfg.get("sequence_critic_weight", 0.0)),
        "sequence_critic_hidden_dim": int(
            train_cfg.get("sequence_critic_hidden_dim", train_cfg.get("hidden_dim", 64))
        ),
        "sequence_critic_num_layers": int(train_cfg.get("sequence_critic_num_layers", 1)),
        "sequence_critic_dropout": float(train_cfg.get("sequence_critic_dropout", 0.0)),
        "sequence_critic_positive_activation": str(
            train_cfg.get("sequence_critic_positive_activation", "softplus")
        ),
        "sequence_critic_force_chrono": bool(train_cfg.get("sequence_critic_force_chrono", True)),
        "residual_edge_weight_enabled": bool(train_cfg.get("residual_edge_weight_enabled", False)),
        "residual_edge_hidden_dim": int(train_cfg.get("residual_edge_hidden_dim", 32)),
        "residual_edge_max_delta": float(train_cfg.get("residual_edge_max_delta", 0.25)),
        "residual_edge_detach_features": bool(train_cfg.get("residual_edge_detach_features", True)),
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
        "hall_attack_hub_fraction": float(train_cfg.get("hall_attack_hub_fraction", 0.2)),
        "hall_attack_noise_mult": float(train_cfg.get("hall_attack_noise_mult", 3.0)),
        "hall_attack_timeflip_prob": float(train_cfg.get("hall_attack_timeflip_prob", 0.5)),
        "hall_attack_edge_drop_prob": float(train_cfg.get("hall_attack_edge_drop_prob", 0.2)),
        "hall_attack_sign_flip_prob": float(train_cfg.get("hall_attack_sign_flip_prob", 0.2)),
        "hall_attack_hub_weight_scale": float(train_cfg.get("hall_attack_hub_weight_scale", 0.5)),
        "neg_gate_margin": float(train_cfg.get("neg_gate_margin", 1.0)),
        "eval_neg_mode": str(sweep_cfg.get("eval_neg_mode", "auto")),
        "eval_neg_modes": sweep_cfg.get("eval_neg_modes", []),
        "eval_neg_aggregate": str(sweep_cfg.get("eval_neg_aggregate", "mean")),
        "eval_neg_aggregate_include_base": bool(
            sweep_cfg.get("eval_neg_aggregate_include_base", True)
        ),
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
        "econ_signal_polarity": str(sweep_cfg.get("econ_signal_polarity", "high")),
        "econ_turnover_cost_bps": float(sweep_cfg.get("econ_turnover_cost_bps", 0.0)),
        "econ_slippage_bps": float(sweep_cfg.get("econ_slippage_bps", 0.0)),
        "econ_slippage_vol_scale": float(sweep_cfg.get("econ_slippage_vol_scale", 0.0)),
        "econ_slippage_vol_lookback": int(sweep_cfg.get("econ_slippage_vol_lookback", 21)),
        "econ_regime_gate_enabled": bool(sweep_cfg.get("econ_regime_gate_enabled", False)),
        "econ_regime_gate_window": int(sweep_cfg.get("econ_regime_gate_window", 63)),
        "econ_regime_confidence_temp": float(sweep_cfg.get("econ_regime_confidence_temp", 1.0)),
        "econ_regime_neutral_exposure": float(sweep_cfg.get("econ_regime_neutral_exposure", 0.0)),
        "econ_regime_min_confidence": float(sweep_cfg.get("econ_regime_min_confidence", 0.0)),
        "econ_regime_uncertainty_scale": float(sweep_cfg.get("econ_regime_uncertainty_scale", 0.0)),
        "econ_regime_risk_scale": float(sweep_cfg.get("econ_regime_risk_scale", 0.0)),
        "econ_regime_thresholding_enabled": bool(
            sweep_cfg.get("econ_regime_thresholding_enabled", True)
        ),
        "econ_regime_threshold_window": int(
            sweep_cfg.get("econ_regime_threshold_window", sweep_cfg.get("econ_signal_window", 126))
        ),
        "econ_regime_threshold_quantile": float(
            sweep_cfg.get("econ_regime_threshold_quantile", sweep_cfg.get("econ_signal_quantile", 0.5))
        ),
        "econ_regime_vol_window": int(sweep_cfg.get("econ_regime_vol_window", 21)),
        "econ_regime_low_quantile": float(sweep_cfg.get("econ_regime_low_quantile", 0.33)),
        "econ_regime_high_quantile": float(sweep_cfg.get("econ_regime_high_quantile", 0.67)),
        "econ_loader_batch_size": int(sweep_cfg.get("econ_loader_batch_size", 128)),
        "econ_trading_days": int(sweep_cfg.get("econ_trading_days", 252)),
    }
    walk_forward_cap_applied = False
    wf_cap = int(sweep_cfg.get("walk_forward_max_folds_cap", 3))
    if (
        is_walk_forward_mode(str(base.get("split_mode", "chronological")))
        and int(base.get("walk_forward_max_folds", 0)) <= 0
        and wf_cap > 0
    ):
        base["walk_forward_max_folds"] = int(wf_cap)
        walk_forward_cap_applied = True
        print(
            "sweep walk_forward_max_folds capped to "
            f"{int(wf_cap)} (set walk_forward_max_folds>0 or walk_forward_max_folds_cap<=0 to override)."
        )
    base["walk_forward_max_folds_cap_applied"] = int(walk_forward_cap_applied)
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
        "walk_forward_max_folds_cap",
        "out_csv",
        "modes",
        "seed",
        "max_runs",
        "timing_warmup_epochs",
        "eval_neg_mode",
        "eval_neg_modes",
        "eval_neg_top_k",
        "eval_neg_pre_rank_mode",
        "eval_neg_aggregate",
        "eval_neg_aggregate_include_base",
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
        "stability_penalty",
        "stability_penalty_lambda",
        "econ_sharpe_uplift_min_floor",
        "finance_sep_gate_metric",
        "finance_sep_gate_floor",
        "finance_auroc_gate_metric",
        "finance_auroc_gate_floor",
        "rank_gate_penalty",
        "econ_top_k",
        "econ_pre_rank_mode",
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
        "econ_regime_gate_enabled",
        "econ_regime_gate_window",
        "econ_regime_confidence_temp",
        "econ_regime_neutral_exposure",
        "econ_regime_min_confidence",
        "econ_regime_uncertainty_scale",
        "econ_regime_risk_scale",
        "econ_regime_thresholding_enabled",
        "econ_regime_threshold_window",
        "econ_regime_threshold_quantile",
        "econ_regime_vol_window",
        "econ_regime_low_quantile",
        "econ_regime_high_quantile",
        "econ_loader_batch_size",
        "econ_trading_days",
        "successive_halving_enabled",
        "successive_halving_stage_fracs",
        "successive_halving_keep_ratio",
        "successive_halving_min_keep",
        "successive_halving_rank_mode",
        "successive_halving_disable_eval_neg_before_final",
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
    parallel_force_cpu = bool(sweep_cfg.get("parallel_force_cpu", False))
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

    total_trials = len(combos) * len(modes)
    econ_top_k = max(0, int(sweep_cfg.get("econ_top_k", 0)))
    econ_pre_rank_mode = str(sweep_cfg.get("econ_pre_rank_mode", "objective")).strip() or "objective"
    econ_two_stage = (
        bool(base.get("econ_enabled", False))
        and econ_top_k > 0
        and econ_top_k < max(1, total_trials)
    )
    if econ_two_stage:
        print(
            f"Two-stage econ sweep enabled: preselect top-{econ_top_k} by {econ_pre_rank_mode}, "
            "then run econ metrics only for selected trials."
        )
    eval_neg_top_k = max(0, int(sweep_cfg.get("eval_neg_top_k", 0)))
    eval_neg_pre_rank_mode = (
        str(sweep_cfg.get("eval_neg_pre_rank_mode", "objective")).strip() or "objective"
    )
    eval_neg_two_stage = eval_neg_top_k > 0 and eval_neg_top_k < max(1, total_trials)

    sh_enabled = bool(sweep_cfg.get("successive_halving_enabled", False))
    sh_stage_fracs_raw = sweep_cfg.get("successive_halving_stage_fracs", [1.0])
    sh_keep_ratio = float(sweep_cfg.get("successive_halving_keep_ratio", 0.5))
    sh_keep_ratio = _clamp(sh_keep_ratio, 0.05, 1.0)
    sh_min_keep = max(1, int(sweep_cfg.get("successive_halving_min_keep", 4)))
    sh_rank_mode = str(sweep_cfg.get("successive_halving_rank_mode", "objective")).strip() or "objective"
    sh_disable_eval_neg_before_final = bool(
        sweep_cfg.get("successive_halving_disable_eval_neg_before_final", True)
    )
    sh_stage_fracs: list[float] = []
    if isinstance(sh_stage_fracs_raw, (list, tuple)):
        sh_items = list(sh_stage_fracs_raw)
    else:
        sh_items = str(sh_stage_fracs_raw).split(",")
    for item in sh_items:
        try:
            frac = float(item)
        except (TypeError, ValueError):
            continue
        if frac <= 0:
            continue
        sh_stage_fracs.append(_clamp(frac, 0.01, 1.0))
    if not sh_stage_fracs:
        sh_stage_fracs = [1.0]
    sh_stage_fracs = sorted(set(float(f) for f in sh_stage_fracs))
    if sh_stage_fracs[-1] < 1.0:
        sh_stage_fracs.append(1.0)
    if not sh_enabled or len(sh_stage_fracs) <= 1 or total_trials <= sh_min_keep:
        sh_enabled = False
        sh_stage_fracs = [1.0]
    if eval_neg_two_stage and not sh_enabled:
        print(
            "eval_neg_top_k requested but successive halving is disabled; "
            "robust eval-neg modes will run for all trials."
        )
        eval_neg_two_stage = False
    if sh_enabled:
        print(
            "Successive halving enabled: "
            f"stages={sh_stage_fracs}, keep_ratio={sh_keep_ratio:.2f}, min_keep={sh_min_keep}"
        )
    if eval_neg_two_stage:
        print(
            f"Two-stage eval-neg enabled: run robust eval-neg modes only on top-{eval_neg_top_k} "
            f"by {eval_neg_pre_rank_mode} in final stage."
        )

    tracked_cfg_keys = (
        "neg_mode",
        "eval_neg_mode",
        "goodness_temp",
        "goodness_target",
        "eval_batch_size",
        "eval_loader_workers",
        "eval_pin_memory",
        "eval_prefetch_factor",
        "eval_persistent_workers",
        "early_stop_enabled",
        "early_stop_min_epochs",
        "early_stop_patience",
        "early_stop_min_delta",
        "early_stop_eval_every",
        "early_stop_eval_graphs",
        "early_stop_rank_mode",
        "neg_mix_end",
        "hall_steps",
        "hall_lr",
        "hall_node_fraction",
        "hall_attack_hub_fraction",
        "hall_attack_noise_mult",
        "hall_attack_timeflip_prob",
        "hall_attack_edge_drop_prob",
        "hall_attack_sign_flip_prob",
        "hall_attack_hub_weight_scale",
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
        "critic_hidden_dim",
        "critic_num_layers",
        "critic_dropout",
        "critic_positive_activation",
        "critic_ensemble_size",
        "critic_ensemble_seed_stride",
        "sequence_critic_enabled",
        "sequence_critic_weight",
        "sequence_critic_hidden_dim",
        "sequence_critic_num_layers",
        "sequence_critic_dropout",
        "sequence_critic_positive_activation",
        "sequence_critic_force_chrono",
        "econ_regime_thresholding_enabled",
        "econ_regime_threshold_window",
        "econ_regime_threshold_quantile",
        "econ_regime_vol_window",
        "econ_regime_low_quantile",
        "econ_regime_high_quantile",
        "econ_signal_polarity",
        "residual_edge_weight_enabled",
        "residual_edge_hidden_dim",
        "residual_edge_max_delta",
        "residual_edge_detach_features",
    )

    run_idx = 0
    trial_records: list[dict] = []
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
            if econ_two_stage:
                cfg_mode["econ_enabled"] = False
                cfg_mode["econ_payload_enabled"] = True
            if run_idx <= len(modes):
                _warn_self_contrastive_eval_view(cfg_mode, mode)
            eval_neg_modes_cfg = cfg_mode.get("eval_neg_modes", [])
            if isinstance(eval_neg_modes_cfg, str):
                eval_neg_modes_cfg = [
                    m.strip() for m in str(eval_neg_modes_cfg).split(",") if m.strip()
                ]
            has_eval_neg_modes = any(str(m).strip() for m in eval_neg_modes_cfg)
            seed = int(cfg_mode.get("seed", 7)) + run_idx
            trial_idx = len(trial_records)
            trial_records.append(
                {
                    "trial_idx": trial_idx,
                    "cfg_mode": cfg_mode.copy(),
                    "combo": dict(combo),
                    "layerwise": bool(layerwise),
                    "mode": str(mode),
                    "seed": int(seed),
                    "has_eval_neg_modes": bool(has_eval_neg_modes),
                }
            )
    trial_by_id = {int(r["trial_idx"]): r for r in trial_records}

    def _build_stage_cfg(rec: dict, stage_frac: float, enable_eval_neg_modes: bool):
        cfg_stage = dict(rec["cfg_mode"])
        target_epochs = max(1, int(cfg_stage.get("epochs", base.get("epochs", 1))))
        stage_epochs = max(1, int(math.ceil(float(target_epochs) * float(stage_frac))))
        cfg_stage["epochs"] = int(stage_epochs)
        cfg_stage["eval_neg_modes_enabled"] = bool(enable_eval_neg_modes)
        if econ_two_stage:
            cfg_stage["econ_enabled"] = False
        return cfg_stage, stage_epochs, target_epochs

    def _decorate_result(
        res: dict,
        rec: dict,
        cfg_stage: dict,
        *,
        stage_idx: int,
        stage_frac: float,
        stage_epochs: int,
        target_epochs: int,
        eval_neg_enabled: bool,
    ) -> dict:
        res["mode"] = rec["mode"]
        res["__trial_idx"] = int(rec["trial_idx"])
        res.update(rec["combo"])
        for k in tracked_cfg_keys:
            if k in cfg_stage:
                res[k] = cfg_stage[k]
        res["successive_halving_enabled"] = int(bool(sh_enabled))
        res["successive_halving_stage"] = int(stage_idx + 1)
        res["successive_halving_stage_frac"] = float(stage_frac)
        res["successive_halving_stage_epochs"] = int(stage_epochs)
        res["successive_halving_target_epochs"] = int(target_epochs)
        res["successive_halving_pruned"] = 0
        res["eval_neg_topk_enabled"] = int(bool(eval_neg_two_stage))
        res["eval_neg_eval_topk"] = int(bool(eval_neg_enabled and rec.get("has_eval_neg_modes", False)))
        return res

    def _run_stage(records: list[dict], stage_idx: int, stage_frac: float, enable_eval_neg_modes: bool):
        stage_results: list[dict] = []
        stage_desc = f"Sweep S{stage_idx + 1}/{len(sh_stage_fracs)}"
        pbar = tqdm(
            total=len(records),
            desc=stage_desc,
            unit="trial",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        if parallel_workers > 1 and records:
            if parallel_backend not in ("process", "thread", "threads"):
                raise ValueError(f"Unknown parallel_backend: {parallel_backend}")
            tasks = []
            task_meta = []
            for rec in records:
                cfg_stage, stage_epochs, target_epochs = _build_stage_cfg(
                    rec, stage_frac, enable_eval_neg_modes
                )
                tasks.append(
                    (
                        str(graphs_path),
                        cfg_stage,
                        rec["combo"],
                        rec["layerwise"],
                        device_str,
                        rec["seed"],
                        worker_threads,
                        worker_interop,
                        rec["trial_idx"],
                    )
                )
                task_meta.append((rec, cfg_stage, stage_epochs, target_epochs))
            if parallel_backend in ("thread", "threads"):
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                    for res, meta in zip(ex.map(_run_trial_worker, tasks), task_meta):
                        rec, cfg_stage, stage_epochs, target_epochs = meta
                        stage_results.append(
                            _decorate_result(
                                res,
                                rec,
                                cfg_stage,
                                stage_idx=stage_idx,
                                stage_frac=stage_frac,
                                stage_epochs=stage_epochs,
                                target_epochs=target_epochs,
                                eval_neg_enabled=enable_eval_neg_modes,
                            )
                        )
                        pbar.update(1)
            else:
                from concurrent.futures import ProcessPoolExecutor
                import multiprocessing as mp

                ctx = mp.get_context(parallel_mp_context)
                with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=ctx) as ex:
                    for res, meta in zip(ex.map(_run_trial_worker, tasks), task_meta):
                        rec, cfg_stage, stage_epochs, target_epochs = meta
                        stage_results.append(
                            _decorate_result(
                                res,
                                rec,
                                cfg_stage,
                                stage_idx=stage_idx,
                                stage_frac=stage_frac,
                                stage_epochs=stage_epochs,
                                target_epochs=target_epochs,
                                eval_neg_enabled=enable_eval_neg_modes,
                            )
                        )
                        pbar.update(1)
        else:
            for rec in records:
                cfg_stage, stage_epochs, target_epochs = _build_stage_cfg(
                    rec, stage_frac, enable_eval_neg_modes
                )
                _set_seed(int(rec["seed"]))
                res = _run_ff_trial(
                    graphs,
                    graph_dates,
                    device,
                    cfg_stage,
                    layerwise=bool(rec["layerwise"]),
                )
                stage_results.append(
                    _decorate_result(
                        res,
                        rec,
                        cfg_stage,
                        stage_idx=stage_idx,
                        stage_frac=stage_frac,
                        stage_epochs=stage_epochs,
                        target_epochs=target_epochs,
                        eval_neg_enabled=enable_eval_neg_modes,
                    )
                )
                pbar.update(1)
        pbar.close()
        return stage_results

    results_by_trial: dict[int, dict] = {}
    active_records = list(trial_records)
    for stage_idx, stage_frac in enumerate(sh_stage_fracs):
        is_final_stage = stage_idx == (len(sh_stage_fracs) - 1)
        enable_eval_neg_modes = True
        if eval_neg_two_stage and sh_disable_eval_neg_before_final and not is_final_stage:
            enable_eval_neg_modes = False
        stage_rows = _run_stage(
            active_records,
            stage_idx=stage_idx,
            stage_frac=float(stage_frac),
            enable_eval_neg_modes=enable_eval_neg_modes,
        )
        for row in stage_rows:
            trial_idx = int(row.get("__trial_idx", -1))
            if trial_idx >= 0:
                results_by_trial[trial_idx] = row
        if is_final_stage:
            break

        rank_mode_stage = sh_rank_mode
        if eval_neg_two_stage and stage_idx == (len(sh_stage_fracs) - 2):
            rank_mode_stage = eval_neg_pre_rank_mode
        ranked_stage = []
        for row in stage_rows:
            trial_idx = int(row.get("__trial_idx", -1))
            metric_name, metric_val = _objective_rank_metric(row, rank_mode=rank_mode_stage)
            row["successive_halving_rank_metric"] = metric_name
            row["successive_halving_rank_value"] = metric_val
            ranked_stage.append((trial_idx, metric_val))
        ranked_stage.sort(
            key=lambda iv: (
                np.isfinite(_to_float(iv[1], float("nan"))),
                _to_float(iv[1], float("-inf")),
            ),
            reverse=True,
        )
        keep_count = max(sh_min_keep, int(math.ceil(len(stage_rows) * sh_keep_ratio)))
        keep_count = min(keep_count, len(stage_rows))
        if eval_neg_two_stage and stage_idx == (len(sh_stage_fracs) - 2):
            keep_count = min(keep_count, max(1, int(eval_neg_top_k)))
        keep_count = max(1, keep_count)
        keep_ids = {int(idx) for idx, _ in ranked_stage[:keep_count] if int(idx) >= 0}
        pruned = 0
        for row in stage_rows:
            trial_idx = int(row.get("__trial_idx", -1))
            if trial_idx >= 0 and trial_idx not in keep_ids:
                row["successive_halving_pruned"] = 1
                row["successive_halving_pruned_stage"] = int(stage_idx + 1)
                pruned += 1
        if pruned > 0:
            print(
                f"Successive halving stage {stage_idx + 1}: kept {len(keep_ids)} / "
                f"{len(stage_rows)} trials."
            )
        active_records = [trial_by_id[tid] for tid in sorted(keep_ids) if tid in trial_by_id]

    final_trial_ids = {int(rec["trial_idx"]) for rec in active_records}
    results = []
    for rec in trial_records:
        trial_idx = int(rec["trial_idx"])
        row = results_by_trial.get(trial_idx)
        if row is None:
            continue
        if sh_enabled and trial_idx not in final_trial_ids:
            row["successive_halving_pruned"] = 1
            row.setdefault("successive_halving_pruned_stage", max(1, len(sh_stage_fracs) - 1))
        else:
            row["successive_halving_pruned"] = 0
        if not eval_neg_two_stage and rec.get("has_eval_neg_modes", False):
            row["eval_neg_eval_topk"] = 1
        results.append(row)
    eval_neg_evaluated_trial_ids: set[int] | None = None
    if results:
        eval_neg_evaluated_trial_ids = {
            int(r.get("__trial_idx", -1))
            for r in results
            if int(r.get("eval_neg_eval_topk", 0)) > 0 and int(r.get("__trial_idx", -1)) >= 0
        }
    econ_evaluated_trial_ids: set[int] | None = None
    if econ_two_stage and results:
        candidate_indices = [
            idx
            for idx, r in enumerate(results)
            if int(r.get("successive_halving_pruned", 0)) == 0
        ]
        if not candidate_indices:
            candidate_indices = list(range(len(results)))
        ranked_pre = []
        for idx in candidate_indices:
            r = results[idx]
            pre_metric, pre_val = _objective_rank_metric(r, rank_mode=econ_pre_rank_mode)
            r["econ_pre_rank_metric"] = pre_metric
            r["econ_pre_rank_value"] = pre_val
            ranked_pre.append((idx, pre_val))
        ranked_pre.sort(
            key=lambda iv: (
                np.isfinite(_to_float(iv[1], float("nan"))),
                _to_float(iv[1], float("-inf")),
            ),
            reverse=True,
        )
        selected_idx = [idx for idx, _ in ranked_pre[: min(econ_top_k, len(ranked_pre))]]
        econ_evaluated_trial_ids = set()
        if selected_idx:
            econ_bar = tqdm(
                total=len(selected_idx),
                desc="Econ Top-K",
                unit="trial",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
            for idx in selected_idx:
                trial_idx = int(results[idx].get("__trial_idx", -1))
                if not (0 <= trial_idx < len(trial_records)):
                    econ_bar.update(1)
                    continue
                rec = trial_records[trial_idx]
                cfg_mode = dict(rec["cfg_mode"])
                cfg_mode["econ_enabled"] = True
                payloads = results[idx].get("__econ_payloads")
                if not isinstance(payloads, list):
                    payloads = []
                econ_rows = []
                for payload in payloads:
                    if not isinstance(payload, dict):
                        continue
                    dates_payload = payload.get("dates")
                    goodness_payload = payload.get("goodness")
                    if not isinstance(dates_payload, list) or not isinstance(goodness_payload, list):
                        continue
                    if len(dates_payload) != len(goodness_payload) or not dates_payload:
                        continue
                    econ_row = _compute_econ_metrics_for_eval(
                        None,
                        None,
                        [],
                        dates_payload,
                        cfg_mode,
                        goodness_values=goodness_payload,
                    )
                    if econ_row:
                        econ_rows.append(econ_row)
                if econ_rows:
                    econ_merged = (
                        econ_rows[0] if len(econ_rows) == 1 else _aggregate_numeric_rows(econ_rows)
                    )
                    for k, v in econ_merged.items():
                        if str(k).startswith("econ_"):
                            results[idx][k] = v
                    results[idx]["econ_eval_topk"] = 1
                    econ_evaluated_trial_ids.add(trial_idx)
                econ_bar.update(1)
            econ_bar.close()
        for r in results:
            r.setdefault("econ_eval_topk", 0)

    if results:
        default_rank_mode = "finance_first" if bool(base.get("econ_enabled", False)) else "objective"
        rank_mode = str(sweep_cfg.get("rank_mode", default_rank_mode))
        rank_mode_norm = rank_mode.strip().lower()
        finance_rank = rank_mode_norm in {"finance_first", "economic", "econ"}
        stability_penalty = bool(sweep_cfg.get("stability_penalty", finance_rank))
        stability_lambda = max(0.0, float(sweep_cfg.get("stability_penalty_lambda", 0.5)))
        rank_gate_floor = _to_float(sweep_cfg.get("econ_sharpe_uplift_min_floor"), float("nan"))
        sep_gate_metric = str(
            sweep_cfg.get("finance_sep_gate_metric", "eval_sep_agg_min")
        ).strip() or "eval_sep_agg_min"
        sep_gate_floor = _to_float(sweep_cfg.get("finance_sep_gate_floor"), float("nan"))
        auroc_gate_metric = str(
            sweep_cfg.get("finance_auroc_gate_metric", "eval_auroc_agg_min")
        ).strip() or "eval_auroc_agg_min"
        auroc_gate_floor = _to_float(sweep_cfg.get("finance_auroc_gate_floor"), float("nan"))
        rank_gate_penalty = abs(float(sweep_cfg.get("rank_gate_penalty", 1e6)))
        for r in results:
            rank_metric, rank_value = _objective_rank_metric(r, rank_mode=rank_mode)
            rank_base_metric = rank_metric
            rank_base_value = rank_value
            gate_failed = False
            gate_reasons: list[str] = []
            trial_idx = int(r.get("__trial_idx", -1))

            if sh_enabled and int(r.get("successive_halving_pruned", 0)) > 0:
                gate_failed = True
                gate_reasons.append("successive_halving_pruned")

            if (
                finance_rank
                and eval_neg_two_stage
                and bool(str(r.get("eval_neg_modes_configured", "")).strip())
                and eval_neg_evaluated_trial_ids is not None
                and trial_idx not in eval_neg_evaluated_trial_ids
            ):
                gate_failed = True
                gate_reasons.append("eval_neg_not_evaluated")

            if (
                finance_rank
                and econ_two_stage
                and econ_evaluated_trial_ids is not None
                and trial_idx not in econ_evaluated_trial_ids
            ):
                gate_failed = True
                gate_reasons.append("econ_not_evaluated")

            if finance_rank and stability_penalty:
                sharpe_mean = _to_float(r.get("econ_sharpe_uplift"), float("nan"))
                sharpe_std = _to_float(r.get("econ_sharpe_uplift_std"), float("nan"))
                if np.isfinite(sharpe_mean) and np.isfinite(sharpe_std):
                    rank_metric = "econ_sharpe_uplift_stability_adj"
                    rank_value = sharpe_mean - stability_lambda * max(0.0, sharpe_std)
                    r["econ_sharpe_uplift_stability_adj"] = rank_value
                    r["econ_sharpe_uplift_stability_lambda"] = stability_lambda

            if finance_rank and np.isfinite(rank_gate_floor):
                floor_metric = _to_float(r.get("econ_sharpe_uplift_min"), float("nan"))
                floor_metric_name = "econ_sharpe_uplift_min"
                if not np.isfinite(floor_metric):
                    floor_metric = _to_float(r.get("econ_sharpe_uplift"), float("nan"))
                    floor_metric_name = "econ_sharpe_uplift"
                if np.isfinite(floor_metric) and floor_metric < rank_gate_floor:
                    gate_failed = True
                    gate_reasons.append(floor_metric_name)
                    r["rank_gate_metric"] = floor_metric_name
                    r["rank_gate_floor"] = float(rank_gate_floor)
                    r["rank_gate_value"] = float(floor_metric)
            if finance_rank and np.isfinite(sep_gate_floor):
                sep_val = _to_float(r.get(sep_gate_metric), float("nan"))
                if np.isfinite(sep_val) and sep_val < sep_gate_floor:
                    gate_failed = True
                    gate_reasons.append(sep_gate_metric)
                    if "rank_gate_metric" not in r:
                        r["rank_gate_metric"] = sep_gate_metric
                        r["rank_gate_floor"] = float(sep_gate_floor)
                        r["rank_gate_value"] = float(sep_val)
            if finance_rank and np.isfinite(auroc_gate_floor):
                auroc_val = _to_float(r.get(auroc_gate_metric), float("nan"))
                if np.isfinite(auroc_val) and auroc_val < auroc_gate_floor:
                    gate_failed = True
                    gate_reasons.append(auroc_gate_metric)
                    if "rank_gate_metric" not in r:
                        r["rank_gate_metric"] = auroc_gate_metric
                        r["rank_gate_floor"] = float(auroc_gate_floor)
                        r["rank_gate_value"] = float(auroc_val)
            if gate_failed:
                rank_value = rank_value - rank_gate_penalty
                r["rank_gate_reasons"] = ",".join(gate_reasons)

            r["rank_base_metric"] = rank_base_metric
            r["rank_base_value"] = rank_base_value
            r["rank_metric"] = rank_metric
            r["rank_value"] = rank_value
            r["rank_mode"] = rank_mode
            r["rank_gate_failed"] = int(gate_failed)

        econ_metric_name = str(sweep_cfg.get("composite_econ_metric", "econ_sharpe_uplift"))
        sep_metric_name = str(sweep_cfg.get("composite_sep_metric", "eval_sep_agg_mean"))
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

    for r in results:
        for k in tuple(r.keys()):
            if str(k).startswith("__"):
                r.pop(k, None)

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
