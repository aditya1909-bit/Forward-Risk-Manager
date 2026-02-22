#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import random
import sys
import tomllib
import csv
import math
import hashlib
import time

import torch
import torch.nn.functional as F
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
    rank_spread_loss,
    self_contrastive_loss,
)
from frisk.hallucinate import HallucinationConfig, hallucinate_negative
from frisk.device import collect_device_diagnostics, empty_device_cache, resolve_device
from frisk.econ_eval import resolve_price_ticker

_RISK_TARGET_MEM_CACHE: dict[str, tuple[list[float | None], float, float]] = {}
_PORT_TARGET_MEM_CACHE: dict[str, tuple[list[float | None], float, float]] = {}
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


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _get_setting(args: argparse.Namespace, section: dict, key: str, default):
    if hasattr(args, key):
        return getattr(args, key)
    if key in section:
        return section[key]
    return default


def _load_state_dict_compat(path: str):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if isinstance(state.get("state_dict"), dict):
            return state["state_dict"]
        if isinstance(state.get("model"), dict):
            return state["model"]
    return state


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _block_endpoint_indices(num_layers: int, block_size: int) -> list[int]:
    if num_layers <= 0:
        return []
    step = max(1, int(block_size))
    endpoints = list(range(step - 1, num_layers, step))
    if endpoints[-1] != num_layers - 1:
        endpoints.append(num_layers - 1)
    return endpoints


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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


def _forward_encoder(model, *args, **kwargs):
    compiler_ns = getattr(torch, "compiler", None)
    mark_step = (
        getattr(compiler_ns, "cudagraph_mark_step_begin", None) if compiler_ns is not None else None
    )
    if callable(mark_step):
        mark_step()
    return model(*args, **kwargs)


def _optimizer_step(
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


def _parse_positive_int_list(value, fallback: int) -> list[int]:
    vals: list[int] = []
    raw_items = []
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    elif value is not None:
        raw_items = str(value).split(",")

    seen = set()
    for item in raw_items:
        s = str(item).strip()
        if not s:
            continue
        try:
            v = int(float(s))
        except ValueError:
            continue
        if v <= 0 or v in seen:
            continue
        seen.add(v)
        vals.append(v)

    if vals:
        return vals
    return [max(1, int(fallback))]


def _parse_str_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    elif value is None:
        items = []
    else:
        items = [s.strip() for s in str(value).split(",")]
    return [s for s in items if s]


def _parse_float_list(value) -> list[float]:
    out: list[float] = []
    for s in _parse_str_list(value):
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def _normalize_mode_weights(modes: list[str], weights: list[float]) -> dict[str, float]:
    if not modes:
        return {}
    if not weights:
        return {m: 1.0 / len(modes) for m in modes}
    vals = []
    for i, _ in enumerate(modes):
        if i < len(weights):
            vals.append(max(0.0, float(weights[i])))
        else:
            vals.append(0.0)
    s = sum(vals)
    if s <= 0:
        return {m: 1.0 / len(modes) for m in modes}
    return {m: vals[i] / s for i, m in enumerate(modes)}


def _curriculum_phase(epoch: int, total_epochs: int, ratios: list[float]) -> int:
    if not ratios:
        r = [0.33, 0.34, 0.33]
    else:
        r = [max(0.0, float(v)) for v in ratios[:3]]
        if len(r) < 3:
            r.extend([0.0] * (3 - len(r)))
    s = sum(r)
    if s <= 0:
        cuts = [0.33, 0.67]
    else:
        n = [v / s for v in r]
        cuts = [n[0], n[0] + n[1]]
    p = float(epoch) / max(1.0, float(total_epochs))
    if p <= cuts[0]:
        return 0
    if p <= cuts[1]:
        return 1
    return 2


def _pick_curriculum_neg_mode(
    base_mode: str,
    epoch: int,
    epochs: int,
    mix_modes: list[str],
    mix_weights: dict[str, float],
    phase_ratios: list[float],
) -> str:
    mode = str(base_mode).strip().lower()
    if mode in {"hallucinate", "self_contrastive"}:
        return mode
    if not mix_modes:
        return mode
    phase = _curriculum_phase(epoch, epochs, phase_ratios)
    easy = {"noise", "time_flip", "shuffle+noise", "shuffle"}
    mid = easy | {"sector_swap"}
    if phase == 0:
        allowed = [m for m in mix_modes if m in easy]
    elif phase == 1:
        allowed = [m for m in mix_modes if m in mid]
    else:
        allowed = list(mix_modes)
    if not allowed:
        return mode
    probs = [max(0.0, float(mix_weights.get(m, 0.0))) for m in allowed]
    ps = sum(probs)
    if ps <= 0:
        return random.choice(allowed)
    r = random.random() * ps
    c = 0.0
    for m, p in zip(allowed, probs):
        c += p
        if r <= c:
            return m
    return allowed[-1]


def _infer_neg_feature_slices(window_len: int, summary_dim: int) -> tuple[int | None, int | None, int]:
    if window_len <= 0:
        return None, None, 0
    if summary_dim >= 10:
        # window_plus_summary_fund: [returns | summary(5) | fund(5)].
        sector_idx = int(window_len) + 5
        factor_start = sector_idx + 1
        return sector_idx, factor_start, 4
    if summary_dim >= 2:
        return None, int(window_len), min(4, int(summary_dim))
    return None, None, 0


def _should_use_hallucination(
    epoch: int,
    step_idx: int,
    warmup_epochs: int,
    every_n_batches: int,
) -> bool:
    if epoch <= int(max(0, warmup_epochs)):
        return False
    n = max(1, int(every_n_batches))
    return (int(step_idx) % n) == 0


def _concat_forward_pos_neg(
    model,
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    batch_nodes: torch.Tensor,
    return_all: bool = False,
):
    if x_pos.shape != x_neg.shape:
        raise ValueError(
            f"concat forward expects matching shapes, got {tuple(x_pos.shape)} vs {tuple(x_neg.shape)}"
        )
    n_nodes = int(x_pos.size(0))
    if n_nodes == 0:
        if return_all:
            return [], []
        return x_pos, x_neg
    num_graphs = int(batch_nodes.max().item()) + 1 if batch_nodes.numel() else 0
    x_cat = torch.cat([x_pos, x_neg], dim=0)
    edge_index_cat = torch.cat([edge_index, edge_index + n_nodes], dim=1)
    edge_weight_cat = None
    if edge_weight is not None:
        edge_weight_cat = torch.cat([edge_weight, edge_weight], dim=0)
    batch_cat = torch.cat([batch_nodes, batch_nodes + num_graphs], dim=0)
    if return_all:
        layers = _forward_encoder(model, x_cat, edge_index_cat, edge_weight=edge_weight_cat, return_all=True)
        pos_layers = [h[:n_nodes] for h in layers]
        neg_layers = [h[n_nodes:] for h in layers]
        return pos_layers, neg_layers
    h = _forward_encoder(model, x_cat, edge_index_cat, edge_weight=edge_weight_cat)
    return h[:n_nodes], h[n_nodes:]


def _goodness_rank_alignment_loss(
    g_scores: torch.Tensor,
    graph_idx,
    portfolio_targets: list[float | None] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if portfolio_targets is None:
        return None
    if torch.is_tensor(graph_idx):
        idx_list = graph_idx.detach().cpu().tolist()
    elif isinstance(graph_idx, (list, tuple)):
        idx_list = list(graph_idx)
    else:
        idx_list = [int(graph_idx)]
    target_vals = []
    for gi in idx_list:
        if 0 <= int(gi) < len(portfolio_targets):
            tv = portfolio_targets[int(gi)]
        else:
            tv = None
        target_vals.append(float(tv) if tv is not None else float("nan"))
    t = torch.tensor(target_vals, dtype=g_scores.dtype, device=device)
    mask = torch.isfinite(t)
    if int(mask.sum().item()) < 4:
        return None
    g = g_scores[mask]
    t = t[mask]
    k = max(1, min(int(g.numel() // 2), int(round(0.2 * g.numel()))))
    hi = torch.topk(t, k=k, largest=True).indices
    lo = torch.topk(t, k=k, largest=False).indices
    spread = g.index_select(0, hi).mean() - g.index_select(0, lo).mean()
    # Encourage high-return graphs to receive higher goodness scores.
    return F.softplus(0.1 - spread)


def _compute_risk_targets(
    prices_path: str,
    ticker: str,
    dates: list[str],
    horizon: int,
    standardize: bool,
    max_abs_logret: float,
    cache_dir: str | None = "runs/cache",
) -> tuple[list[float | None], float, float]:
    prices_file = Path(prices_path)
    try:
        st = prices_file.stat()
        file_sig = f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        file_sig = "missing"
    dates_hash = hashlib.sha1("\n".join(dates).encode("utf-8")).hexdigest()
    cache_key = hashlib.sha1(
        "|".join(
            [
                str(prices_file.resolve()),
                str(ticker).upper(),
                str(horizon),
                str(int(bool(standardize))),
                f"{float(max_abs_logret):.8f}",
                file_sig,
                dates_hash,
            ]
        ).encode("utf-8")
    ).hexdigest()

    cached = _RISK_TARGET_MEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_path: Path | None = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"risk_targets_{cache_key}.pt"
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                targets = payload["targets"]
                mean = float(payload["mean"])
                std = float(payload["std"])
                result = (targets, mean, std)
                _RISK_TARGET_MEM_CACHE[cache_key] = result
                return result
            except Exception:
                pass

    prices_by_date: dict[str, list[float]] = {}
    ticker_norm = str(ticker).upper()
    with Path(prices_path).open() as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("prices.csv missing header")
        price_col = "adj_close" if "adj_close" in r.fieldnames else "close"
        for row in r:
            if str(row.get("ticker", "")).upper() != ticker_norm:
                continue
            date = row.get("date")
            if not date:
                continue
            val = row.get(price_col, "")
            if not val:
                continue
            try:
                price = float(val)
            except ValueError:
                continue
            if not math.isfinite(price) or price <= 0:
                continue
            prices_by_date.setdefault(date, []).append(price)
    prices: list[tuple[str, float]] = []
    for date, vals in prices_by_date.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        mid = len(vals_sorted) // 2
        if len(vals_sorted) % 2 == 1:
            px = vals_sorted[mid]
        else:
            px = 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
        prices.append((date, float(px)))
    if not prices:
        raise ValueError(f"No prices found for ticker {ticker} in {prices_path}")

    prices.sort(key=lambda x: x[0])
    date_list = [d for d, _ in prices]
    price_list = [p for _, p in prices]
    returns = []
    clip = float(max_abs_logret)
    for i in range(len(price_list) - 1):
        if price_list[i] <= 0 or price_list[i + 1] <= 0:
            returns.append(0.0)
            continue
        ret = math.log(price_list[i + 1] / price_list[i])
        if clip > 0 and abs(ret) > clip:
            ret = math.copysign(clip, ret)
        returns.append(ret)
    idx_map = {d: i for i, d in enumerate(date_list)}

    targets: list[float | None] = []
    for d in dates:
        idx = idx_map.get(d)
        if idx is None:
            targets.append(None)
            continue
        if idx + horizon > len(returns):
            targets.append(None)
            continue
        window = returns[idx : idx + horizon]
        if not window:
            targets.append(None)
            continue
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / len(window)
        vol = math.sqrt(var)
        targets.append(vol)

    finite = [t for t in targets if t is not None]
    if not finite:
        return targets, 0.0, 1.0
    mean = sum(finite) / len(finite)
    var = sum((x - mean) ** 2 for x in finite) / len(finite)
    std = math.sqrt(var) if var > 0 else 1.0

    if standardize:
        targets = [((t - mean) / (std + 1e-6)) if t is not None else None for t in targets]
    result = (targets, mean, std)
    _RISK_TARGET_MEM_CACHE[cache_key] = result
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"targets": targets, "mean": mean, "std": std}, cache_path)
        except Exception:
            pass
    return result


def _compute_forward_return_targets(
    prices_path: str,
    ticker: str,
    dates: list[str],
    horizon: int,
    standardize: bool,
    max_abs_logret: float,
    cache_dir: str | None = "runs/cache",
) -> tuple[list[float | None], float, float]:
    prices_file = Path(prices_path)
    try:
        st = prices_file.stat()
        file_sig = f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        file_sig = "missing"
    dates_hash = hashlib.sha1("\n".join(dates).encode("utf-8")).hexdigest()
    cache_key = hashlib.sha1(
        "|".join(
            [
                "portfolio_targets",
                str(prices_file.resolve()),
                str(ticker).upper(),
                str(horizon),
                str(int(bool(standardize))),
                f"{float(max_abs_logret):.8f}",
                file_sig,
                dates_hash,
            ]
        ).encode("utf-8")
    ).hexdigest()

    cached = _PORT_TARGET_MEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_path: Path | None = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"portfolio_targets_{cache_key}.pt"
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                targets = payload["targets"]
                mean = float(payload["mean"])
                std = float(payload["std"])
                result = (targets, mean, std)
                _PORT_TARGET_MEM_CACHE[cache_key] = result
                return result
            except Exception:
                pass

    prices_by_date: dict[str, list[float]] = {}
    ticker_norm = str(ticker).upper()
    with Path(prices_path).open() as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("prices.csv missing header")
        price_col = "adj_close" if "adj_close" in r.fieldnames else "close"
        for row in r:
            if str(row.get("ticker", "")).upper() != ticker_norm:
                continue
            date = row.get("date")
            if not date:
                continue
            val = row.get(price_col, "")
            if not val:
                continue
            try:
                price = float(val)
            except ValueError:
                continue
            if not math.isfinite(price) or price <= 0:
                continue
            prices_by_date.setdefault(date, []).append(price)
    prices: list[tuple[str, float]] = []
    for date, vals in prices_by_date.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        mid = len(vals_sorted) // 2
        if len(vals_sorted) % 2 == 1:
            px = vals_sorted[mid]
        else:
            px = 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
        prices.append((date, float(px)))
    if not prices:
        raise ValueError(f"No prices found for ticker {ticker} in {prices_path}")

    prices.sort(key=lambda x: x[0])
    date_list = [d for d, _ in prices]
    price_list = [p for _, p in prices]
    log_returns = []
    clip = float(max_abs_logret)
    for i in range(len(price_list) - 1):
        if price_list[i] <= 0 or price_list[i + 1] <= 0:
            log_returns.append(0.0)
            continue
        ret = math.log(price_list[i + 1] / price_list[i])
        if clip > 0 and abs(ret) > clip:
            ret = math.copysign(clip, ret)
        log_returns.append(ret)
    idx_map = {d: i for i, d in enumerate(date_list)}

    targets: list[float | None] = []
    horizon = max(1, int(horizon))
    for d in dates:
        idx = idx_map.get(d)
        if idx is None:
            targets.append(None)
            continue
        if idx + horizon > len(log_returns):
            targets.append(None)
            continue
        window = log_returns[idx : idx + horizon]
        if not window:
            targets.append(None)
            continue
        cum_log = float(sum(window))
        targets.append(float(math.exp(cum_log) - 1.0))

    finite = [t for t in targets if t is not None]
    if not finite:
        result = (targets, 0.0, 1.0)
        _PORT_TARGET_MEM_CACHE[cache_key] = result
        return result
    mean = sum(finite) / len(finite)
    var = sum((x - mean) ** 2 for x in finite) / len(finite)
    std = math.sqrt(var) if var > 0 else 1.0

    if standardize:
        targets = [((t - mean) / (std + 1e-6)) if t is not None else None for t in targets]
    result = (targets, mean, std)
    _PORT_TARGET_MEM_CACHE[cache_key] = result
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"targets": targets, "mean": mean, "std": std}, cache_path)
        except Exception:
            pass
    return result


def _compute_portfolio_head_loss(
    portfolio_head: torch.nn.Module,
    embeddings: torch.Tensor,
    graph_idx,
    portfolio_targets: list[float | None],
    device: torch.device,
    loss_type: str,
) -> torch.Tensor | None:
    if not portfolio_targets:
        return None
    if torch.is_tensor(graph_idx):
        idx_list = graph_idx.detach().cpu().tolist()
    elif isinstance(graph_idx, (list, tuple)):
        idx_list = list(graph_idx)
    else:
        idx_list = [int(graph_idx)]
    target_vals = []
    for gi in idx_list:
        if 0 <= gi < len(portfolio_targets):
            tv = portfolio_targets[gi]
        else:
            tv = None
        target_vals.append(float(tv) if tv is not None else float("nan"))
    target = torch.tensor(target_vals, dtype=torch.float32, device=device)
    mask = torch.isfinite(target)
    if not mask.any():
        return None

    pred_raw = portfolio_head(embeddings)
    if pred_raw.ndim == 2 and pred_raw.size(1) == 1:
        pred_raw = pred_raw.squeeze(1)
    if pred_raw.ndim != 1:
        raise RuntimeError(f"portfolio head output shape mismatch: {tuple(pred_raw.shape)}")
    pred = torch.tanh(pred_raw)

    loss_mode = str(loss_type).strip().lower()
    if loss_mode == "mse":
        return F.mse_loss(pred[mask], target[mask])

    pnl = pred[mask] * target[mask]
    if pnl.numel() == 0:
        return None
    if pnl.numel() == 1:
        return -pnl.mean()
    mean = pnl.mean()
    std = pnl.std(unbiased=False) + 1e-6
    return -(mean / std)


def _compute_multi_horizon_risk_loss(
    risk_head: torch.nn.Module,
    embeddings: torch.Tensor,
    graph_idx,
    risk_targets_by_horizon: list[list[float | None]],
    device: torch.device,
    risk_loss_type: str,
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
    for gi in idx_list:
        row = []
        for horizon_targets in risk_targets_by_horizon:
            if 0 <= gi < len(horizon_targets):
                t = horizon_targets[gi]
            else:
                t = None
            row.append(float(t) if t is not None else float("nan"))
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
    if risk_loss_type == "mse":
        return F.mse_loss(pred[mask], target[mask])
    return F.smooth_l1_loss(pred[mask], target[mask])


def _self_contrastive_batch_loss(
    h_pos: torch.Tensor,
    h_view: torch.Tensor,
    batch: torch.Tensor,
    temperature: float,
    max_graphs: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z_pos = global_mean_pool(h_pos, batch)
    z_view = global_mean_pool(h_view, batch)
    if max_graphs and z_pos.size(0) > max_graphs:
        idx = torch.randperm(z_pos.size(0), device=z_pos.device)[:max_graphs]
        z_pos = z_pos.index_select(0, idx)
        z_view = z_view.index_select(0, idx)
    loss, pos_score, neg_score = self_contrastive_loss(
        z_pos,
        z_view,
        temperature=temperature,
    )
    return loss, pos_score, neg_score, z_pos, z_view


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
    sector_idx: int | None = None,
    factor_start_idx: int | None = None,
    factor_dim: int = 0,
):
    if use_mode == "self_contrastive":
        use_mode = "shuffle"
    if use_mode in {"schedule", "mix"}:
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
        edge_drop_prob = float(getattr(hall_cfg, "adversarial_edge_drop_prob", 0.2))
        sign_flip_prob = float(getattr(hall_cfg, "adversarial_sign_flip_prob", 0.2))
        hub_weight_scale = float(getattr(hall_cfg, "adversarial_hub_weight_scale", 0.5))
        hub_frac = max(0.0, min(1.0, hub_frac))
        noise_mult = max(1.0, noise_mult)
        flip_prob = max(0.0, min(1.0, flip_prob))
        edge_drop_prob = max(0.0, min(1.0, edge_drop_prob))
        sign_flip_prob = max(0.0, min(1.0, sign_flip_prob))
        hub_weight_scale = max(0.0, hub_weight_scale)
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
                (max(0.0, float(noise_std)) * noise_mult)
                * torch.randn_like(out_adv.index_select(0, hub_idx)),
            )
            # Approximate hub-edge drops by muting a subset of hub-node return windows.
            if edge_drop_prob > 0 and torch.rand((), device=out.device).item() < edge_drop_prob:
                if window_len is not None and int(window_len) > 0 and out_adv.size(1) >= int(window_len):
                    out_adv[hub_idx, : int(window_len)] *= max(0.0, 1.0 - hub_weight_scale)
                else:
                    out_adv[hub_idx] *= max(0.0, 1.0 - hub_weight_scale)
            if sign_flip_prob > 0 and torch.rand((), device=out.device).item() < sign_flip_prob:
                if window_len is not None and int(window_len) > 0 and out_adv.size(1) >= int(window_len):
                    out_adv[hub_idx, : int(window_len)] *= -1.0
                else:
                    out_adv[hub_idx] *= -1.0
            if flip_prob > 0 and torch.rand((), device=out.device).item() < flip_prob:
                if window_len is not None and int(window_len) > 1 and out_adv.size(1) >= int(window_len):
                    wlen = int(window_len)
                    out_adv[hub_idx, :wlen] = torch.flip(out_adv[hub_idx, :wlen], dims=[1])
                elif out_adv.size(1) > 1:
                    out_adv[hub_idx] = torch.flip(out_adv[hub_idx], dims=[1])
        return out_adv
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
            critic=critic,
        )
    if window_len is not None and int(window_len) > 0:
        if summary_dim >= 10:
            if sector_idx is None:
                sector_idx = int(window_len) + 5
            if factor_start_idx is None:
                factor_start_idx = int(window_len) + 6
            if factor_dim <= 0:
                factor_dim = 4
        elif summary_dim >= 2:
            if factor_start_idx is None:
                factor_start_idx = int(window_len)
            if factor_dim <= 0:
                factor_dim = min(4, int(summary_dim))
    return make_negative(
        x,
        batch,
        mode=use_mode,
        noise_std=noise_std,
        window_len=window_len,
        summary_dim=summary_dim,
        sector_idx=sector_idx,
        factor_start_idx=factor_start_idx,
        factor_dim=factor_dim,
    )


def _build_critic(config: dict, hidden_dim: int, device: torch.device):
    critic_hidden_dim = max(1, int(config.get("critic_hidden_dim", hidden_dim)))
    critic_num_layers = max(1, int(config.get("critic_num_layers", 2)))
    critic_dropout = max(0.0, float(config.get("critic_dropout", config.get("dropout", 0.1))))
    critic_positive = str(config.get("critic_positive_activation", "softplus")).strip().lower()
    if critic_positive not in {"softplus", "square"}:
        critic_positive = "softplus"

    ensemble_size = max(1, int(config.get("critic_ensemble_size", 1)))
    seed_base = int(config.get("seed", 7))
    seed_stride = max(1, int(config.get("critic_ensemble_seed_stride", 1009)))
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

    seq_enabled = bool(config.get("sequence_critic_enabled", False))
    if not seq_enabled:
        return base_critic

    seq_hidden = max(1, int(config.get("sequence_critic_hidden_dim", hidden_dim)))
    seq_layers = max(1, int(config.get("sequence_critic_num_layers", 1)))
    seq_dropout = max(0.0, float(config.get("sequence_critic_dropout", 0.0)))
    seq_positive = str(config.get("sequence_critic_positive_activation", "softplus")).strip().lower()
    if seq_positive not in {"softplus", "square"}:
        seq_positive = "softplus"
    seq_weight = float(config.get("sequence_critic_weight", 0.0))
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


def _try_batch_size(
    graphs,
    model,
    critic,
    device,
    batch_size,
    loader_workers,
    neg_mode,
    noise_std,
    goodness_target,
    goodness_temp,
    hall_cfg: HallucinationConfig,
    window_len: int | None,
    summary_dim: int,
    multiscale: bool,
    self_contrastive_temp: float,
    self_contrastive_max_graphs: int,
    self_contrastive_view_mode: str,
    self_contrastive_view_noise_std: float,
    self_contrastive_ff_weight: float,
    self_contrastive_ff_neg_mode: str,
    self_contrastive_ff_noise_std: float,
    self_contrastive_ff_target: float,
    distance_forward_weight: float,
    distance_forward_margin: float,
    distance_forward_max_graphs: int,
    ff_margin: float,
    ff_margin_weight: float,
    loader_shuffle: bool = True,
):
    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=bool(loader_shuffle),
        drop_last=False,
        num_workers=loader_workers,
    )
    batch = next(iter(loader))
    batch = batch.to(device)
    x = batch.x
    edge_weight = getattr(batch, "edge_weight", None)
    if multiscale:
        layers_pos = _forward_encoder(model, x, batch.edge_index, edge_weight=edge_weight, return_all=True)
        if neg_mode == "self_contrastive":
            x_view = _make_self_contrastive_view(
                x,
                batch.batch,
                view_mode=self_contrastive_view_mode,
                view_noise_std=self_contrastive_view_noise_std,
                window_len=window_len,
                summary_dim=summary_dim,
            )
            layers_view = _forward_encoder(model, 
                x_view, batch.edge_index, edge_weight=edge_weight, return_all=True
            )
            loss = 0.0
            z_pos_last = None
            z_neg_last = None
            for h_pos, h_view in zip(layers_pos, layers_view):
                sc_loss, _, _, z_pos, z_view = _self_contrastive_batch_loss(
                    h_pos,
                    h_view,
                    batch.batch,
                    temperature=self_contrastive_temp,
                    max_graphs=self_contrastive_max_graphs,
                )
                loss = loss + sc_loss
                z_pos_last = z_pos
                z_neg_last = permute_graph_embeddings(z_view)
            loss = loss / max(1, len(layers_pos))
            if (
                distance_forward_weight > 0
                and z_pos_last is not None
                and z_neg_last is not None
            ):
                loss = loss + distance_forward_weight * pairwise_distance_forward_loss(
                    z_pos_last,
                    z_neg_last,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
            if self_contrastive_ff_weight > 0:
                x_neg_aux = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    self_contrastive_ff_neg_mode,
                    self_contrastive_ff_noise_std,
                    hall_cfg,
                    critic=critic,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )
                layers_neg_aux = _forward_encoder(model, 
                    x_neg_aux,
                    batch.edge_index,
                    edge_weight=edge_weight,
                    return_all=True,
                )
                g_pos_aux = goodness(
                    layers_pos[-1],
                    batch.batch,
                    temperature=goodness_temp,
                    critic=critic,
                )
                g_neg_aux = goodness(
                    layers_neg_aux[-1],
                    batch.batch,
                    temperature=goodness_temp,
                    critic=critic,
                )
                loss = loss + self_contrastive_ff_weight * ff_loss(
                    g_pos_aux,
                    g_neg_aux,
                    target=self_contrastive_ff_target,
                    margin=ff_margin,
                    margin_weight=ff_margin_weight,
                )
        else:
            x_neg_hall = _make_negatives(
                model,
                x,
                batch.batch,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                edge_weight,
                neg_mode,
                noise_std,
                hall_cfg,
                critic=critic,
                window_len=window_len,
                summary_dim=summary_dim,
            )
            x_neg_time = make_negative(
                x,
                batch.batch,
                mode="time_flip",
                noise_std=noise_std,
                window_len=window_len,
                summary_dim=summary_dim,
            )
            layers_neg_h = _forward_encoder(model, 
                x_neg_hall, batch.edge_index, edge_weight=edge_weight, return_all=True
            )
            layers_neg_t = _forward_encoder(model, 
                x_neg_time, batch.edge_index, edge_weight=edge_weight, return_all=True
            )
            loss = 0.0
            for h_pos, h_neg_h, h_neg_t in zip(layers_pos, layers_neg_h, layers_neg_t):
                g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)
                g_neg_h = goodness(h_neg_h, batch.batch, temperature=goodness_temp, critic=critic)
                g_neg_t = goodness(h_neg_t, batch.batch, temperature=goodness_temp, critic=critic)
                loss = loss + ff_loss(
                    g_pos,
                    g_neg_h,
                    target=goodness_target,
                    margin=ff_margin,
                    margin_weight=ff_margin_weight,
                )
                loss = loss + ff_loss(
                    g_pos,
                    g_neg_t,
                    target=goodness_target,
                    margin=ff_margin,
                    margin_weight=ff_margin_weight,
                )
            loss = loss / max(1, len(layers_pos))
            if distance_forward_weight > 0:
                z_pos = global_mean_pool(layers_pos[-1], batch.batch)
                z_neg_h = global_mean_pool(layers_neg_h[-1], batch.batch)
                z_neg_t = global_mean_pool(layers_neg_t[-1], batch.batch)
                dist_loss_h = pairwise_distance_forward_loss(
                    z_pos,
                    z_neg_h,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
                dist_loss_t = pairwise_distance_forward_loss(
                    z_pos,
                    z_neg_t,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
                loss = loss + distance_forward_weight * 0.5 * (dist_loss_h + dist_loss_t)
    else:
        h_pos = _forward_encoder(model, x, batch.edge_index, edge_weight=edge_weight)
        if neg_mode == "self_contrastive":
            x_view = _make_self_contrastive_view(
                x,
                batch.batch,
                view_mode=self_contrastive_view_mode,
                view_noise_std=self_contrastive_view_noise_std,
                window_len=window_len,
                summary_dim=summary_dim,
            )
            h_view = _forward_encoder(model, x_view, batch.edge_index, edge_weight=edge_weight)
            loss, _, _, z_pos, z_view = _self_contrastive_batch_loss(
                h_pos,
                h_view,
                batch.batch,
                temperature=self_contrastive_temp,
                max_graphs=self_contrastive_max_graphs,
            )
            if distance_forward_weight > 0:
                z_neg = permute_graph_embeddings(z_view)
                loss = loss + distance_forward_weight * pairwise_distance_forward_loss(
                    z_pos,
                    z_neg,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
            if self_contrastive_ff_weight > 0:
                x_neg_aux = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    self_contrastive_ff_neg_mode,
                    self_contrastive_ff_noise_std,
                    hall_cfg,
                    critic=critic,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )
                h_neg_aux = _forward_encoder(model, x_neg_aux, batch.edge_index, edge_weight=edge_weight)
                g_pos_aux = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)
                g_neg_aux = goodness(h_neg_aux, batch.batch, temperature=goodness_temp, critic=critic)
                loss = loss + self_contrastive_ff_weight * ff_loss(
                    g_pos_aux,
                    g_neg_aux,
                    target=self_contrastive_ff_target,
                    margin=ff_margin,
                    margin_weight=ff_margin_weight,
                )
        else:
            g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)
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
                critic=critic,
                window_len=window_len,
                summary_dim=summary_dim,
            )
            h_neg = _forward_encoder(model, x_neg, batch.edge_index, edge_weight=edge_weight)
            g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp, critic=critic)
            loss = ff_loss(
                g_pos,
                g_neg,
                target=goodness_target,
                margin=ff_margin,
                margin_weight=ff_margin_weight,
            )
            if distance_forward_weight > 0:
                z_pos = global_mean_pool(h_pos, batch.batch)
                z_neg = global_mean_pool(h_neg, batch.batch)
                loss = loss + distance_forward_weight * pairwise_distance_forward_loss(
                    z_pos,
                    z_neg,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
    loss.backward()
    model.zero_grad(set_to_none=True)
    if critic is not None:
        critic.zero_grad(set_to_none=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Forward-Forward GNN on rolling correlation graphs.")
    parser.add_argument("--config", help="Path to TOML config")
    parser.add_argument("--graphs", help="Path to graphs.pt from build_graphs.py", default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--hidden-dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-layers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--encoder-conv-type",
        choices=["gcn", "sage", "gat"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--encoder-gat-heads", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--goodness-target", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--neg-mode",
        choices=[
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
            "hallucinate",
            "schedule",
            "mix",
            "self_contrastive",
        ],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--noise-std", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--loader-workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--auto-tune-batch", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-auto-tune-batch", action="store_true", default=False)
    parser.add_argument("--auto-tune-max-batch", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--auto-tune-factor", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--auto-tune-min-batch", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--neg-warmup-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--neg-mix-start", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--neg-mix-end", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--neg-mix-ramp-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--neg-gate-margin", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--ff-neg-mix", default=argparse.SUPPRESS)
    parser.add_argument("--ff-neg-mix-weights", default=argparse.SUPPRESS)
    parser.add_argument("--ff-curriculum-epochs", default=argparse.SUPPRESS)
    parser.add_argument("--ff-rank-aux-weight", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--ff-rank-use-portfolio-targets",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--ff-hall-every-n-batches", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ff-hall-warmup-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ff-hall-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ff-concat-posneg", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--ff-layer-cache", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--ff-econ-eval-every", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--grad-clip", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-view-mode", default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-view-noise-std", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-ff-weight", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-ff-neg-mode", default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-ff-noise-std", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-ff-target", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--self-contrastive-energy-penalty-scale",
        type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--temp-sweep", default=argparse.SUPPRESS)
    parser.add_argument("--ff-layerwise", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--ff-blockwise", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--ff-block-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ff-multiscale", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--strict-component-split", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--freeze-encoder", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--freeze-critic", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--encoder-checkpoint-in", default=argparse.SUPPRESS)
    parser.add_argument("--critic-checkpoint-in", default=argparse.SUPPRESS)
    parser.add_argument("--save-encoder", default=argparse.SUPPRESS)
    parser.add_argument("--save-critic", default=argparse.SUPPRESS)
    parser.add_argument("--critic-hidden-dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--critic-num-layers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--critic-dropout", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--critic-positive-activation",
        choices=["softplus", "square"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--torch-compile", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-torch-compile", action="store_true", default=False)
    parser.add_argument("--torch-compile-mode", default=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    section = cfg.get("train", {})
    build_cfg = cfg.get("build_graphs", {})

    graphs_path = _get_setting(args, section, "graphs", None)
    if not graphs_path:
        raise ValueError("Provide --graphs (or set it in config).")

    epochs = _get_setting(args, section, "epochs", 10)
    batch_size = _get_setting(args, section, "batch_size", 8)
    lr = _get_setting(args, section, "lr", 1e-3)
    hidden_dim = _get_setting(args, section, "hidden_dim", 64)
    num_layers = _get_setting(args, section, "num_layers", 2)
    dropout = _get_setting(args, section, "dropout", 0.1)
    encoder_conv_type = str(_get_setting(args, section, "encoder_conv_type", "gcn")).strip().lower()
    encoder_gat_heads = int(_get_setting(args, section, "encoder_gat_heads", 2))
    goodness_target = _get_setting(args, section, "goodness_target", 1.0)
    goodness_temp = _get_setting(args, section, "goodness_temp", 1.0)
    ff_margin = float(_get_setting(args, section, "ff_margin", 0.0))
    ff_margin_weight = float(_get_setting(args, section, "ff_margin_weight", 1.0))
    temp_sweep = _get_setting(args, section, "temp_sweep", "")
    neg_mode = _get_setting(args, section, "neg_mode", "shuffle")
    noise_std = _get_setting(args, section, "noise_std", 0.05)
    device_choice = _get_setting(args, section, "device", "auto")
    seed = _get_setting(args, section, "seed", 7)
    loader_workers = _get_setting(args, section, "loader_workers", 0)
    dataloader_persistent = _get_setting(args, section, "dataloader_persistent_workers", True)
    dataloader_prefetch = _get_setting(args, section, "dataloader_prefetch_factor", 2)
    dataloader_pin_memory = _get_setting(args, section, "dataloader_pin_memory", False)
    dataloader_mp_context = _get_setting(args, section, "dataloader_mp_context", "")
    torch_num_threads = _get_setting(args, section, "torch_num_threads", None)
    torch_num_interop_threads = _get_setting(args, section, "torch_num_interop_threads", None)
    log_csv = _get_setting(args, section, "log_csv", "")
    plot_path = _get_setting(args, section, "plot_path", "")
    save_model = _get_setting(args, section, "save_model", "")
    save_encoder = _get_setting(args, section, "save_encoder", save_model)
    save_critic = _get_setting(args, section, "save_critic", "")
    encoder_checkpoint_in = _get_setting(args, section, "encoder_checkpoint_in", "")
    critic_checkpoint_in = _get_setting(args, section, "critic_checkpoint_in", "")
    strict_component_split = _to_bool(
        _get_setting(args, section, "strict_component_split", False)
    )
    freeze_encoder = _to_bool(_get_setting(args, section, "freeze_encoder", False))
    freeze_critic = _to_bool(_get_setting(args, section, "freeze_critic", False))
    critic_hidden_dim = int(_get_setting(args, section, "critic_hidden_dim", hidden_dim))
    critic_num_layers = int(_get_setting(args, section, "critic_num_layers", 2))
    critic_dropout = float(_get_setting(args, section, "critic_dropout", dropout))
    critic_positive_activation = str(
        _get_setting(args, section, "critic_positive_activation", "softplus")
    )
    critic_ensemble_size = int(_get_setting(args, section, "critic_ensemble_size", 1))
    critic_ensemble_seed_stride = int(
        _get_setting(args, section, "critic_ensemble_seed_stride", 1009)
    )
    sequence_critic_enabled = _to_bool(
        _get_setting(args, section, "sequence_critic_enabled", False)
    )
    sequence_critic_weight = float(_get_setting(args, section, "sequence_critic_weight", 0.0))
    sequence_critic_hidden_dim = int(
        _get_setting(args, section, "sequence_critic_hidden_dim", hidden_dim)
    )
    sequence_critic_num_layers = int(
        _get_setting(args, section, "sequence_critic_num_layers", 1)
    )
    sequence_critic_dropout = float(_get_setting(args, section, "sequence_critic_dropout", 0.0))
    sequence_critic_positive_activation = str(
        _get_setting(args, section, "sequence_critic_positive_activation", "softplus")
    )
    sequence_critic_force_chrono = _to_bool(
        _get_setting(args, section, "sequence_critic_force_chrono", True)
    )
    residual_edge_weight_enabled = _to_bool(
        _get_setting(args, section, "residual_edge_weight_enabled", False)
    )
    residual_edge_hidden_dim = int(_get_setting(args, section, "residual_edge_hidden_dim", 32))
    residual_edge_max_delta = float(_get_setting(args, section, "residual_edge_max_delta", 0.25))
    residual_edge_detach_features = _to_bool(
        _get_setting(args, section, "residual_edge_detach_features", True)
    )
    auto_tune = _get_setting(args, section, "auto_tune_batch", False)
    if bool(getattr(args, "no_auto_tune_batch", False)):
        auto_tune = False
    auto_tune_max = _get_setting(args, section, "auto_tune_max_batch", 64)
    auto_tune_factor = _get_setting(args, section, "auto_tune_factor", 2)
    auto_tune_min = _get_setting(args, section, "auto_tune_min_batch", 1)
    neg_warmup_epochs = _get_setting(args, section, "neg_warmup_epochs", 0)
    neg_mix_start = _get_setting(args, section, "neg_mix_start", 0.0)
    neg_mix_end = _get_setting(args, section, "neg_mix_end", 0.7)
    neg_mix_ramp_epochs = _get_setting(args, section, "neg_mix_ramp_epochs", 10)
    neg_gate_margin = _get_setting(args, section, "neg_gate_margin", 0.1)
    ff_neg_mix_raw = _get_setting(args, section, "ff_neg_mix", [])
    ff_neg_mix_weights_raw = _get_setting(args, section, "ff_neg_mix_weights", [])
    ff_curriculum_epochs_raw = _get_setting(args, section, "ff_curriculum_epochs", [])
    ff_rank_aux_weight = float(_get_setting(args, section, "ff_rank_aux_weight", 0.0))
    ff_rank_use_portfolio_targets = _to_bool(
        _get_setting(args, section, "ff_rank_use_portfolio_targets", True)
    )
    ff_hall_every_n_batches = int(_get_setting(args, section, "ff_hall_every_n_batches", 1))
    ff_hall_warmup_epochs = int(_get_setting(args, section, "ff_hall_warmup_epochs", 0))
    ff_hall_steps_override = _get_setting(args, section, "ff_hall_steps", None)
    ff_concat_posneg = _to_bool(_get_setting(args, section, "ff_concat_posneg", True))
    ff_layer_cache = _to_bool(_get_setting(args, section, "ff_layer_cache", True))
    ff_econ_eval_every = int(_get_setting(args, section, "ff_econ_eval_every", 1))
    grad_clip = _get_setting(args, section, "grad_clip", 1.0)
    self_contrastive_temp = float(_get_setting(args, section, "self_contrastive_temp", 0.2))
    self_contrastive_view_mode = str(
        _get_setting(args, section, "self_contrastive_view_mode", "shuffle+noise")
    )
    self_contrastive_view_noise_std = float(
        _get_setting(args, section, "self_contrastive_view_noise_std", noise_std)
    )
    self_contrastive_max_graphs = int(_get_setting(args, section, "self_contrastive_max_graphs", 0))
    self_contrastive_ff_weight = float(
        _get_setting(args, section, "self_contrastive_ff_weight", 0.0)
    )
    self_contrastive_ff_neg_mode = str(
        _get_setting(args, section, "self_contrastive_ff_neg_mode", "shuffle+noise")
    )
    self_contrastive_ff_noise_std = float(
        _get_setting(args, section, "self_contrastive_ff_noise_std", noise_std)
    )
    self_contrastive_ff_target = float(
        _get_setting(args, section, "self_contrastive_ff_target", goodness_target)
    )
    self_contrastive_energy_penalty_scale = float(
        _get_setting(args, section, "self_contrastive_energy_penalty_scale", 0.0)
    )
    distance_forward_weight = float(_get_setting(args, section, "distance_forward_weight", 0.0))
    distance_forward_margin = float(_get_setting(args, section, "distance_forward_margin", 0.15))
    distance_forward_max_graphs = int(_get_setting(args, section, "distance_forward_max_graphs", 0))
    distance_forward_interval = int(_get_setting(args, section, "distance_forward_interval", 1))
    amp_requested = _to_bool(_get_setting(args, section, "amp", True))
    amp_dtype_raw = _get_setting(args, section, "amp_dtype", "float16")
    fused_optimizer = _to_bool(_get_setting(args, section, "fused_optimizer", True))
    torch_compile_enabled = _to_bool(_get_setting(args, section, "torch_compile", False))
    if bool(getattr(args, "no_torch_compile", False)):
        torch_compile_enabled = False
    torch_compile_mode = str(_get_setting(args, section, "torch_compile_mode", "reduce-overhead"))
    ff_layerwise = _get_setting(args, section, "ff_layerwise", False) or getattr(
        args, "ff_layerwise", False
    )
    ff_blockwise = _to_bool(_get_setting(args, section, "ff_blockwise", False)) or getattr(
        args, "ff_blockwise", False
    )
    ff_block_size = int(_get_setting(args, section, "ff_block_size", 2))
    ff_multiscale = _get_setting(args, section, "ff_multiscale", False) or getattr(
        args, "ff_multiscale", False
    )
    if neg_mode == "self_contrastive" and ff_layerwise:
        print("self_contrastive mode requires end-to-end FF; disabling ff_layerwise.")
        ff_layerwise = False
    if ff_blockwise and not ff_layerwise:
        print("ff_blockwise requires ff_layerwise; disabling ff_blockwise.")
        ff_blockwise = False
    if ff_block_size < 1:
        ff_block_size = 1
    if ff_blockwise and ff_block_size <= 1:
        print("ff_block_size <= 1; disabling ff_blockwise.")
        ff_blockwise = False
    if ff_multiscale and ff_layerwise:
        print("ff_multiscale enabled; disabling ff_layerwise.")
        ff_layerwise = False
    if ff_multiscale and ff_blockwise:
        ff_blockwise = False

    hall_steps = _get_setting(args, section, "hallucinate_steps", 10)
    if ff_hall_steps_override not in (None, "", "none", "null"):
        hall_steps = ff_hall_steps_override
    hall_lr = _get_setting(args, section, "hallucinate_lr", 0.1)
    hall_l2 = _get_setting(args, section, "hallucinate_l2", 0.1)
    hall_mean = _get_setting(args, section, "hallucinate_mean", 0.05)
    hall_std = _get_setting(args, section, "hallucinate_std", 0.05)
    hall_corr = _get_setting(args, section, "hallucinate_corr", 1.0)
    hall_clamp = _get_setting(args, section, "hallucinate_clamp_std", 3.0)
    hall_node_fraction = _get_setting(args, section, "hallucinate_node_fraction", 1.0)
    hall_node_min = _get_setting(args, section, "hallucinate_node_min", 1)
    hall_init_noise = _get_setting(args, section, "hallucinate_init_noise", 0.0)
    hall_min_delta = _get_setting(args, section, "hallucinate_min_delta", 0.0)
    hall_fallback_noise = _get_setting(args, section, "hallucinate_fallback_noise", 1.0)
    hall_penalty_scope = str(
        _get_setting(args, section, "hallucinate_penalty_scope", "returns")
    ).strip().lower()
    hall_corr_scope = str(
        _get_setting(args, section, "hallucinate_corr_scope", "returns")
    ).strip().lower()
    hall_corr_every_n_steps = int(
        _get_setting(args, section, "hallucinate_corr_every_n_steps", 1)
    )
    hall_corr_edge_fraction = float(
        _get_setting(args, section, "hallucinate_corr_edge_fraction", 1.0)
    )
    hall_corr_edge_min = int(
        _get_setting(args, section, "hallucinate_corr_edge_min", 1)
    )
    hall_adaptive_lr = _to_bool(_get_setting(args, section, "hallucinate_adaptive_lr", False))
    hall_adaptive_lr_patience = int(
        _get_setting(args, section, "hallucinate_adaptive_lr_patience", 2)
    )
    hall_adaptive_lr_decay = float(
        _get_setting(args, section, "hallucinate_adaptive_lr_decay", 0.5)
    )
    hall_adaptive_lr_min = float(
        _get_setting(args, section, "hallucinate_adaptive_lr_min", 1e-4)
    )
    hall_early_stop_on_target_hit = _to_bool(
        _get_setting(args, section, "hallucinate_early_stop_on_target_hit", False)
    )
    hall_target_hit_patience = int(
        _get_setting(args, section, "hallucinate_target_hit_patience", 1)
    )
    hall_moment_mean = float(_get_setting(args, section, "hallucinate_moment_mean", 0.0))
    hall_moment_var = float(_get_setting(args, section, "hallucinate_moment_var", 0.0))
    hall_moment_skew = float(_get_setting(args, section, "hallucinate_moment_skew", 0.0))
    hall_moment_scope = str(_get_setting(args, section, "hallucinate_moment_scope", "returns"))
    hall_freeze_non_return = _to_bool(
        _get_setting(args, section, "hallucinate_freeze_non_return_features", True)
    )
    hall_attack_hub_fraction = float(_get_setting(args, section, "hall_attack_hub_fraction", 0.2))
    hall_attack_noise_mult = float(_get_setting(args, section, "hall_attack_noise_mult", 3.0))
    hall_attack_timeflip_prob = float(_get_setting(args, section, "hall_attack_timeflip_prob", 0.5))
    hall_attack_edge_drop_prob = float(
        _get_setting(args, section, "hall_attack_edge_drop_prob", 0.2)
    )
    hall_attack_sign_flip_prob = float(
        _get_setting(args, section, "hall_attack_sign_flip_prob", 0.2)
    )
    hall_attack_hub_weight_scale = float(
        _get_setting(args, section, "hall_attack_hub_weight_scale", 0.5)
    )
    if hall_penalty_scope not in {"all", "returns"}:
        hall_penalty_scope = "returns"
    if hall_corr_scope not in {"all", "returns"}:
        hall_corr_scope = "returns"
    if hall_moment_scope not in {"all", "returns"}:
        hall_moment_scope = "returns"
    hall_corr_every_n_steps = max(1, int(hall_corr_every_n_steps))
    hall_corr_edge_fraction = _clamp(float(hall_corr_edge_fraction), 0.0, 1.0)
    hall_corr_edge_min = max(1, int(hall_corr_edge_min))
    hall_curriculum = section.get("hallucinate_curriculum", {})
    hall_curr_enabled = bool(hall_curriculum.get("enabled", False))
    hall_curr_start = int(hall_curriculum.get("start_epoch", 1))
    hall_curr_ramp = int(hall_curriculum.get("ramp_epochs", 1))

    def _sync_hall_curriculum_end() -> None:
        if not hall_curr_enabled:
            return
        hall_curriculum["steps_end"] = int(hall_steps)
        hall_curriculum["lr_end"] = float(hall_lr)
        hall_curriculum["l2_end"] = float(hall_l2)
        hall_curriculum["mean_end"] = float(hall_mean)
        hall_curriculum["std_end"] = float(hall_std)
        hall_curriculum["corr_end"] = float(hall_corr)
        hall_curriculum["node_fraction_end"] = float(hall_node_fraction)
        hall_curriculum["node_min_end"] = int(hall_node_min)

    layerwise_neg_mode = _get_setting(args, section, "layerwise_neg_mode", "shuffle")
    layerwise_noise_std = _get_setting(args, section, "layerwise_noise_std", noise_std)
    layerwise_hall_corr = _get_setting(args, section, "layerwise_hall_corr", 0.0)
    layerwise_hall_mean = _get_setting(args, section, "layerwise_hall_mean", hall_mean)
    layerwise_hall_std = _get_setting(args, section, "layerwise_hall_std", hall_std)
    feature_mode = build_cfg.get("feature_mode", "window")
    window_len = int(build_cfg.get("window", 20))
    if feature_mode in ("window", "window_plus_summary", "window_plus_summary_fund"):
        returns_len = window_len
    else:
        returns_len = 1
    if feature_mode == "window_plus_summary":
        summary_dim = 5
    elif feature_mode == "window_plus_summary_fund":
        summary_dim = 10
    else:
        summary_dim = 0

    energy_penalty_weight = float(_get_setting(args, section, "energy_penalty_weight", 0.0))
    energy_penalty_mode = _get_setting(args, section, "energy_penalty_mode", "last")

    risk_head_enabled = bool(_get_setting(args, section, "risk_head_enabled", False))
    risk_ticker = _get_setting(args, section, "risk_ticker", "AUTO")
    risk_horizon = int(_get_setting(args, section, "risk_horizon", 21))
    risk_horizons = _parse_positive_int_list(
        _get_setting(args, section, "risk_horizons", section.get("risk_horizon", risk_horizon)),
        fallback=risk_horizon,
    )
    risk_loss_weight = float(_get_setting(args, section, "risk_loss_weight", 0.1))
    risk_loss_type = _get_setting(args, section, "risk_loss_type", "huber")
    risk_standardize = bool(_get_setting(args, section, "risk_standardize", True))
    risk_cache_dir = _get_setting(args, section, "risk_cache_dir", "runs/cache")
    risk_max_abs_logret = float(_get_setting(args, section, "risk_max_abs_logret", 0.5))
    portfolio_head_enabled = bool(_get_setting(args, section, "portfolio_head_enabled", False))
    portfolio_ticker = _get_setting(args, section, "portfolio_ticker", "AUTO")
    portfolio_horizon = int(_get_setting(args, section, "portfolio_horizon", 21))
    portfolio_loss_weight = float(_get_setting(args, section, "portfolio_loss_weight", 0.0))
    portfolio_loss_type = _get_setting(args, section, "portfolio_loss_type", "sharpe")
    portfolio_standardize = bool(_get_setting(args, section, "portfolio_standardize", True))
    portfolio_cache_dir = _get_setting(args, section, "portfolio_cache_dir", "runs/cache")
    portfolio_max_abs_logret = float(_get_setting(args, section, "portfolio_max_abs_logret", 0.5))

    adaptive_hall_enabled = _to_bool(_get_setting(args, section, "adaptive_hallucination", True))
    adaptive_hall_close_high = float(_get_setting(args, section, "adaptive_hall_close_high", 0.75))
    adaptive_hall_hardness_low = float(_get_setting(args, section, "adaptive_hall_hardness_low", -0.2))
    adaptive_hall_hardness_high = float(_get_setting(args, section, "adaptive_hall_hardness_high", 0.4))
    adaptive_hall_lr_mult = float(_get_setting(args, section, "adaptive_hall_lr_mult", 1.15))
    adaptive_hall_lr_max = float(_get_setting(args, section, "adaptive_hall_lr_max", 0.2))
    adaptive_hall_steps_inc = int(_get_setting(args, section, "adaptive_hall_steps_inc", 1))
    adaptive_hall_steps_max = int(_get_setting(args, section, "adaptive_hall_steps_max", 16))
    adaptive_hall_node_inc = float(_get_setting(args, section, "adaptive_hall_node_inc", 0.05))
    adaptive_hall_reg_mult = float(_get_setting(args, section, "adaptive_hall_reg_mult", 0.9))
    adaptive_hall_reg_min = float(_get_setting(args, section, "adaptive_hall_reg_min", 0.001))
    adaptive_hall_min_delta_mult = float(_get_setting(args, section, "adaptive_hall_min_delta_mult", 0.9))
    adaptive_hall_min_delta_min = float(_get_setting(args, section, "adaptive_hall_min_delta_min", 0.005))
    adaptive_hall_ratio_high = float(_get_setting(args, section, "adaptive_hall_ratio_high", 0.8))
    adaptive_mix_step = float(_get_setting(args, section, "adaptive_mix_step", 0.05))
    adaptive_mix_end_max = float(_get_setting(args, section, "adaptive_mix_end_max", 0.85))
    adaptive_gate_margin_step = float(_get_setting(args, section, "adaptive_gate_margin_step", 0.05))
    adaptive_gate_margin_min = float(_get_setting(args, section, "adaptive_gate_margin_min", 0.1))
    adaptive_gate_margin_max = float(_get_setting(args, section, "adaptive_gate_margin_max", 2.5))

    adaptive_target_enabled = _to_bool(_get_setting(args, section, "adaptive_goodness_target", True))
    adaptive_target_warmup = int(_get_setting(args, section, "adaptive_goodness_target_warmup", 5))
    adaptive_target_alpha = float(_get_setting(args, section, "adaptive_goodness_target_alpha", 0.15))
    adaptive_target_margin = float(_get_setting(args, section, "adaptive_goodness_target_margin", 0.0))
    adaptive_target_min = float(_get_setting(args, section, "adaptive_goodness_target_min", 0.1))
    adaptive_target_max = float(_get_setting(args, section, "adaptive_goodness_target_max", 10.0))
    self_contrastive_temp = _clamp(self_contrastive_temp, 1e-4, 10.0)
    self_contrastive_view_mode = str(self_contrastive_view_mode).strip().lower()
    if self_contrastive_view_mode in ("", "auto"):
        self_contrastive_view_mode = "shuffle+noise"
    self_contrastive_view_noise_std = max(0.0, float(self_contrastive_view_noise_std))
    self_contrastive_max_graphs = max(0, int(self_contrastive_max_graphs))
    self_contrastive_ff_weight = max(0.0, float(self_contrastive_ff_weight))
    self_contrastive_ff_neg_mode = str(self_contrastive_ff_neg_mode).strip().lower()
    if self_contrastive_ff_neg_mode not in _NEG_AUG_MODES:
        self_contrastive_ff_neg_mode = "shuffle+noise"
    self_contrastive_ff_noise_std = max(0.0, float(self_contrastive_ff_noise_std))
    self_contrastive_ff_target = float(self_contrastive_ff_target)
    self_contrastive_energy_penalty_scale = max(0.0, float(self_contrastive_energy_penalty_scale))
    distance_forward_weight = max(0.0, float(distance_forward_weight))
    distance_forward_margin = max(0.0, float(distance_forward_margin))
    distance_forward_max_graphs = max(0, int(distance_forward_max_graphs))
    distance_forward_interval = max(1, int(distance_forward_interval))
    ff_margin = max(0.0, float(ff_margin))
    ff_margin_weight = max(0.0, float(ff_margin_weight))
    critic_hidden_dim = max(1, int(critic_hidden_dim))
    critic_num_layers = max(1, int(critic_num_layers))
    critic_dropout = max(0.0, float(critic_dropout))
    critic_positive_activation = str(critic_positive_activation).strip().lower()
    if critic_positive_activation not in {"softplus", "square"}:
        critic_positive_activation = "softplus"
    critic_ensemble_size = max(1, int(critic_ensemble_size))
    critic_ensemble_seed_stride = max(1, int(critic_ensemble_seed_stride))
    sequence_critic_hidden_dim = max(1, int(sequence_critic_hidden_dim))
    sequence_critic_num_layers = max(1, int(sequence_critic_num_layers))
    sequence_critic_dropout = max(0.0, float(sequence_critic_dropout))
    sequence_critic_positive_activation = str(sequence_critic_positive_activation).strip().lower()
    if sequence_critic_positive_activation not in {"softplus", "square"}:
        sequence_critic_positive_activation = "softplus"
    residual_edge_hidden_dim = max(4, int(residual_edge_hidden_dim))
    residual_edge_max_delta = max(0.0, float(residual_edge_max_delta))
    portfolio_horizon = max(1, int(portfolio_horizon))
    portfolio_loss_weight = max(0.0, float(portfolio_loss_weight))
    portfolio_loss_type = str(portfolio_loss_type).strip().lower()
    if portfolio_loss_type not in {"sharpe", "mse"}:
        portfolio_loss_type = "sharpe"
    layerwise_neg_mode = str(layerwise_neg_mode).strip().lower()
    if layerwise_neg_mode not in _NEG_AUG_MODES:
        layerwise_neg_mode = "shuffle"
    if encoder_conv_type not in {"gcn", "sage", "gat"}:
        encoder_conv_type = "gcn"
    encoder_gat_heads = max(1, int(encoder_gat_heads))

    if strict_component_split and neg_mode == "self_contrastive":
        if "time_flip" in self_contrastive_view_mode:
            print(
                "strict_component_split: removing time_flip from self_contrastive views "
                "(arrow-of-time belongs to critic stage)."
            )
            self_contrastive_view_mode = "shuffle+noise"
        if self_contrastive_ff_weight > 0:
            print("strict_component_split: disabling self_contrastive_ff_weight for encoder stage.")
            self_contrastive_ff_weight = 0.0
        freeze_critic = True
    elif strict_component_split:
        freeze_encoder = True
        if not encoder_checkpoint_in:
            print(
                "strict_component_split: critic stage expects encoder_checkpoint_in; "
                "continuing with current encoder initialization."
            )

    if neg_mode == "self_contrastive" and adaptive_target_enabled:
        print("adaptive_goodness_target disabled for self_contrastive mode.")
        adaptive_target_enabled = False
    adaptive_mix_end_max = _clamp(adaptive_mix_end_max, 0.0, 0.99)
    ff_neg_mix = [m.strip().lower() for m in _parse_str_list(ff_neg_mix_raw)]
    if not ff_neg_mix:
        ff_neg_mix = ["time_flip", "sector_swap", "factor_hard", "noise"]
    ff_neg_mix = [m for m in ff_neg_mix if m in _NEG_AUG_MODES and m not in {"hallucinate"}]
    ff_neg_mix_weights = _normalize_mode_weights(ff_neg_mix, _parse_float_list(ff_neg_mix_weights_raw))
    ff_curriculum_epochs = _parse_float_list(ff_curriculum_epochs_raw)
    ff_rank_aux_weight = max(0.0, float(ff_rank_aux_weight))
    ff_rank_use_portfolio_targets = bool(ff_rank_use_portfolio_targets)
    ff_hall_every_n_batches = max(1, int(ff_hall_every_n_batches))
    ff_hall_warmup_epochs = max(0, int(ff_hall_warmup_epochs))
    ff_econ_eval_every = max(1, int(ff_econ_eval_every))
    torch_compile_mode = str(torch_compile_mode).strip() or "reduce-overhead"
    if neg_mode == "mix":
        neg_mix_end = _clamp(float(neg_mix_end), float(neg_mix_start), adaptive_mix_end_max)

    def _hall_cfg_for_epoch(epoch: int, corr_override: float | None = None, mean_override: float | None = None, std_override: float | None = None) -> HallucinationConfig:
        if not hall_curr_enabled:
            return HallucinationConfig(
                steps=hall_steps,
                lr=hall_lr,
                l2_weight=hall_l2,
                mean_weight=hall_mean if mean_override is None else mean_override,
                std_weight=hall_std if std_override is None else std_override,
                corr_weight=hall_corr if corr_override is None else corr_override,
                clamp_std=hall_clamp,
                goodness_temp=goodness_temp,
                node_fraction=hall_node_fraction,
                node_min=hall_node_min,
                init_noise=hall_init_noise,
                return_slice_len=returns_len,
                penalty_scope=hall_penalty_scope,
                corr_scope=hall_corr_scope,
                freeze_non_return_features=hall_freeze_non_return,
                corr_every_n_steps=hall_corr_every_n_steps,
                corr_edge_fraction=hall_corr_edge_fraction,
                corr_edge_min=hall_corr_edge_min,
                adaptive_lr=hall_adaptive_lr,
                adaptive_lr_patience=hall_adaptive_lr_patience,
                adaptive_lr_decay=hall_adaptive_lr_decay,
                adaptive_lr_min=hall_adaptive_lr_min,
                early_stop_on_target_hit=hall_early_stop_on_target_hit,
                target_hit_patience=hall_target_hit_patience,
                moment_mean_weight=hall_moment_mean,
                moment_var_weight=hall_moment_var,
                moment_skew_weight=hall_moment_skew,
                moment_scope=hall_moment_scope,
                adversarial_hub_fraction=hall_attack_hub_fraction,
                adversarial_feature_noise_mult=hall_attack_noise_mult,
                adversarial_timeflip_prob=hall_attack_timeflip_prob,
                adversarial_edge_drop_prob=hall_attack_edge_drop_prob,
                adversarial_sign_flip_prob=hall_attack_sign_flip_prob,
                adversarial_hub_weight_scale=hall_attack_hub_weight_scale,
            )

        if epoch < hall_curr_start:
            t = 0.0
        else:
            ramp = max(1, hall_curr_ramp)
            t = min(1.0, (epoch - hall_curr_start) / ramp)

        def _curr_val(name: str, base: float, cast=None):
            start = hall_curriculum.get(f"{name}_start", base)
            end = hall_curriculum.get(f"{name}_end", start)
            val = _lerp(float(start), float(end), t)
            if cast is not None:
                return cast(val)
            return val

        steps = max(1, _curr_val("steps", hall_steps, lambda v: int(round(v))))
        lr = _curr_val("lr", hall_lr)
        l2 = _curr_val("l2", hall_l2)
        mean_w = _curr_val("mean", hall_mean) if mean_override is None else mean_override
        std_w = _curr_val("std", hall_std) if std_override is None else std_override
        corr_w = _curr_val("corr", hall_corr) if corr_override is None else corr_override
        clamp_std = _curr_val("clamp_std", hall_clamp)
        node_fraction = _curr_val("node_fraction", hall_node_fraction)
        node_fraction = min(1.0, max(0.0, float(node_fraction)))
        node_min = max(1, _curr_val("node_min", hall_node_min, lambda v: int(round(v))))

        return HallucinationConfig(
            steps=steps,
            lr=lr,
            l2_weight=l2,
            mean_weight=mean_w,
            std_weight=std_w,
            corr_weight=corr_w,
            clamp_std=clamp_std,
            goodness_temp=goodness_temp,
            node_fraction=node_fraction,
            node_min=node_min,
            init_noise=hall_init_noise,
            return_slice_len=returns_len,
            penalty_scope=hall_penalty_scope,
            corr_scope=hall_corr_scope,
            freeze_non_return_features=hall_freeze_non_return,
            corr_every_n_steps=hall_corr_every_n_steps,
            corr_edge_fraction=hall_corr_edge_fraction,
            corr_edge_min=hall_corr_edge_min,
            adaptive_lr=hall_adaptive_lr,
            adaptive_lr_patience=hall_adaptive_lr_patience,
            adaptive_lr_decay=hall_adaptive_lr_decay,
            adaptive_lr_min=hall_adaptive_lr_min,
            early_stop_on_target_hit=hall_early_stop_on_target_hit,
            target_hit_patience=hall_target_hit_patience,
            moment_mean_weight=hall_moment_mean,
            moment_var_weight=hall_moment_var,
            moment_skew_weight=hall_moment_skew,
            moment_scope=hall_moment_scope,
            adversarial_hub_fraction=hall_attack_hub_fraction,
            adversarial_feature_noise_mult=hall_attack_noise_mult,
            adversarial_timeflip_prob=hall_attack_timeflip_prob,
            adversarial_edge_drop_prob=hall_attack_edge_drop_prob,
            adversarial_sign_flip_prob=hall_attack_sign_flip_prob,
            adversarial_hub_weight_scale=hall_attack_hub_weight_scale,
        )

    set_seed(seed)
    if torch_num_threads:
        torch.set_num_threads(int(torch_num_threads))
    if torch_num_interop_threads:
        torch.set_num_interop_threads(int(torch_num_interop_threads))
    device = resolve_device(device_choice)
    amp_dtype = _parse_amp_dtype(amp_dtype_raw)
    amp_enabled = bool(amp_requested and device.type == "cuda")
    if amp_enabled and amp_dtype == torch.bfloat16:
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if not bf16_ok:
            amp_dtype = torch.float16
            print("amp_dtype=bfloat16 not supported on this CUDA device; falling back to float16.")
    scaler = _make_scaler(amp_enabled and amp_dtype == torch.float16)

    try:
        payload = torch.load(Path(graphs_path), map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch versions don't support weights_only
        payload = torch.load(Path(graphs_path), map_location="cpu")
    graphs = payload["graphs"]
    dates = payload.get("dates", [])
    if not graphs:
        raise ValueError("No graphs found in the provided file.")

    for i, g in enumerate(graphs):
        setattr(g, "graph_idx", i)

    print(f"device request: {device_choice}")
    print(f"device: {device}")
    for key, value in collect_device_diagnostics().items():
        print(f"{key}: {value}")
    print(
        f"neg_mode: {neg_mode} | batch_size: {batch_size} | loader_workers: {loader_workers}"
    )
    print(
        "ff_mode: "
        f"layerwise={ff_layerwise}, blockwise={ff_blockwise}, "
        f"block_size={ff_block_size}, multiscale={ff_multiscale}"
    )
    if energy_penalty_weight > 0:
        print(
            f"energy_penalty: {energy_penalty_weight} (mode={energy_penalty_mode})"
        )
    if risk_head_enabled:
        print(
            f"risk_head: ticker={risk_ticker} horizons={risk_horizons} "
            f"weight={risk_loss_weight} type={risk_loss_type} std={risk_standardize} "
            f"max_abs_logret={risk_max_abs_logret}"
        )
    if portfolio_head_enabled:
        print(
            f"portfolio_head: ticker={portfolio_ticker} horizon={portfolio_horizon} "
            f"weight={portfolio_loss_weight} type={portfolio_loss_type} std={portfolio_standardize} "
            f"max_abs_logret={portfolio_max_abs_logret}"
        )
    if ff_rank_aux_weight > 0:
        print(
            f"ff_rank_aux: weight={ff_rank_aux_weight} "
            f"portfolio_targets={ff_rank_use_portfolio_targets and not ff_layerwise}"
        )
    print(
        "critic_arch: "
        f"ensemble={critic_ensemble_size}, seq_enabled={sequence_critic_enabled}, "
        f"seq_weight={sequence_critic_weight}"
    )
    if residual_edge_weight_enabled:
        print(
            "residual_edge_weight: "
            f"enabled hidden_dim={residual_edge_hidden_dim} max_delta={residual_edge_max_delta}"
        )
    if neg_mode == "self_contrastive":
        print(f"self_contrastive_temp: {self_contrastive_temp}")
        print(
            "self_contrastive_view: "
            f"mode={self_contrastive_view_mode}, noise_std={self_contrastive_view_noise_std}"
        )
        if self_contrastive_max_graphs > 0:
            print(f"self_contrastive_max_graphs: {self_contrastive_max_graphs}")
        if self_contrastive_ff_weight > 0:
            print(
                "self_contrastive_ff_aux: "
                f"weight={self_contrastive_ff_weight}, "
                f"neg_mode={self_contrastive_ff_neg_mode}, "
                f"noise_std={self_contrastive_ff_noise_std}, "
                f"target={self_contrastive_ff_target}"
            )
        if energy_penalty_weight > 0 and self_contrastive_energy_penalty_scale != 1.0:
            print(
                "self_contrastive_energy_penalty_scale: "
                f"{self_contrastive_energy_penalty_scale}"
            )
    if distance_forward_weight > 0:
        print(
            "distance_forward: "
            f"weight={distance_forward_weight}, margin={distance_forward_margin}"
        )
        if distance_forward_interval > 1:
            print(f"distance_forward_interval: every {distance_forward_interval} batches")
        if distance_forward_max_graphs > 0:
            print(f"distance_forward_max_graphs: {distance_forward_max_graphs}")
    if amp_enabled:
        dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
        print(f"amp: enabled ({dtype_name})")
    if torch_num_threads or torch_num_interop_threads:
        print(
            f"torch threads: {torch.get_num_threads()} | interop: {torch.get_num_interop_threads()}"
        )
    if hall_curr_enabled:
        steps_start = hall_curriculum.get("steps_start", hall_steps)
        steps_end = hall_curriculum.get("steps_end", hall_steps)
        lr_start = hall_curriculum.get("lr_start", hall_lr)
        lr_end = hall_curriculum.get("lr_end", hall_lr)
        frac_start = hall_curriculum.get("node_fraction_start", hall_node_fraction)
        frac_end = hall_curriculum.get("node_fraction_end", hall_node_fraction)
        print(
            "hallucination curriculum: "
            f"start={hall_curr_start}, ramp={hall_curr_ramp}, "
            f"steps {steps_start}->{steps_end}, "
            f"lr {lr_start}->{lr_end}, "
            f"node_fraction {frac_start}->{frac_end}"
        )
        _sync_hall_curriculum_end()
    print(
        "hallucination scope: "
        f"penalty={hall_penalty_scope}, corr={hall_corr_scope}, "
        f"freeze_non_return={hall_freeze_non_return}, return_slice_len={returns_len}"
    )
    if neg_mode in ("schedule", "mix"):
        print(f"neg_warmup_epochs: {neg_warmup_epochs}")
    if neg_mode == "mix":
        print(
            f"neg_mix_start: {neg_mix_start} | neg_mix_end: {neg_mix_end} | "
            f"neg_mix_ramp_epochs: {neg_mix_ramp_epochs}"
        )
    if ff_neg_mix:
        print(
            "ff_neg_mix: "
            + ",".join(f"{m}:{ff_neg_mix_weights.get(m, 0.0):.3f}" for m in ff_neg_mix)
        )
    print(
        "ff_runtime: "
        f"concat_posneg={ff_concat_posneg}, layer_cache={ff_layer_cache}, "
        f"rank_aux_weight={ff_rank_aux_weight}, hall_every_n_batches={ff_hall_every_n_batches}, "
        f"hall_warmup_epochs={ff_hall_warmup_epochs}, econ_eval_every={ff_econ_eval_every}"
    )
    if torch_compile_enabled:
        print(f"torch_compile: requested (mode={torch_compile_mode})")
    if adaptive_hall_enabled:
        print(
            "adaptive_hallucination: "
            f"close_high={adaptive_hall_close_high}, "
            f"hardness_low={adaptive_hall_hardness_low}, "
            f"hardness_high={adaptive_hall_hardness_high}, "
            f"ratio_high={adaptive_hall_ratio_high}, "
            f"mix_end_max={adaptive_mix_end_max}"
        )
    if adaptive_target_enabled:
        print(
            "adaptive_goodness_target: "
            f"warmup={adaptive_target_warmup}, alpha={adaptive_target_alpha}, "
            f"range=[{adaptive_target_min}, {adaptive_target_max}]"
        )
    print(
        "auto_tune_batch: "
        f"{auto_tune} (max={auto_tune_max}, factor={auto_tune_factor}, min={auto_tune_min})"
    )

    input_dim = graphs[0].x.shape[1]
    model = GCNEncoder(
        in_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=encoder_conv_type,
        gat_heads=encoder_gat_heads,
        residual_edge_enabled=bool(residual_edge_weight_enabled),
        residual_edge_hidden_dim=int(residual_edge_hidden_dim),
        residual_edge_max_delta=float(residual_edge_max_delta),
        residual_edge_detach_features=bool(residual_edge_detach_features),
    ).to(device)
    if torch_compile_enabled and hasattr(torch, "compile"):
        requested_mode = str(torch_compile_mode).strip() or "default"
        compile_candidates: list[str] = []
        if device.type == "cuda" and requested_mode == "reduce-overhead":
            # Multi-forward FF training is sensitive to CUDA graph buffer reuse.
            compile_candidates.append("max-autotune-no-cudagraphs")
        compile_candidates.append(requested_mode)
        if "default" not in compile_candidates:
            compile_candidates.append("default")
        seen_modes: set[str] = set()
        compile_candidates = [m for m in compile_candidates if not (m in seen_modes or seen_modes.add(m))]
        for mode in compile_candidates:
            try:
                model = torch.compile(model, mode=mode)
                torch_compile_mode = mode
                print(f"torch_compile: active (mode={mode})")
                break
            except Exception as exc:
                print(f"torch.compile failed (mode={mode}): {exc}")
        else:
            torch_compile_enabled = False
            print("torch_compile: disabled after fallback attempts.")
    elif torch_compile_enabled:
        torch_compile_enabled = False
        print("torch_compile: requested but torch.compile is unavailable in this runtime.")
    critic_cfg = {
        "dropout": float(dropout),
        "seed": int(seed),
        "critic_hidden_dim": int(critic_hidden_dim),
        "critic_num_layers": int(critic_num_layers),
        "critic_dropout": float(critic_dropout),
        "critic_positive_activation": str(critic_positive_activation),
        "critic_ensemble_size": int(critic_ensemble_size),
        "critic_ensemble_seed_stride": int(critic_ensemble_seed_stride),
        "sequence_critic_enabled": bool(sequence_critic_enabled),
        "sequence_critic_weight": float(sequence_critic_weight),
        "sequence_critic_hidden_dim": int(sequence_critic_hidden_dim),
        "sequence_critic_num_layers": int(sequence_critic_num_layers),
        "sequence_critic_dropout": float(sequence_critic_dropout),
        "sequence_critic_positive_activation": str(sequence_critic_positive_activation),
    }
    critic = _build_critic(critic_cfg, hidden_dim=hidden_dim, device=device)
    if encoder_checkpoint_in:
        ckpt = Path(str(encoder_checkpoint_in))
        if not ckpt.exists():
            raise FileNotFoundError(f"encoder_checkpoint_in not found: {ckpt}")
        model.load_state_dict(_load_state_dict_compat(str(ckpt)))
        print(f"loaded encoder checkpoint: {ckpt}")
    if critic_checkpoint_in:
        ckpt = Path(str(critic_checkpoint_in))
        if not ckpt.exists():
            raise FileNotFoundError(f"critic_checkpoint_in not found: {ckpt}")
        state = _load_state_dict_compat(str(ckpt))
        try:
            critic.load_state_dict(state, strict=True)
        except Exception:
            missing, unexpected = critic.load_state_dict(state, strict=False)
            print(
                f"warning: partial critic checkpoint load "
                f"(missing={len(missing)} unexpected={len(unexpected)})."
            )
        print(f"loaded critic checkpoint: {ckpt}")
    if freeze_encoder:
        for p in model.parameters():
            p.requires_grad_(False)
    if freeze_critic:
        for p in critic.parameters():
            p.requires_grad_(False)

    print(
        "component_split: "
        f"strict={strict_component_split}, freeze_encoder={freeze_encoder}, freeze_critic={freeze_critic}"
    )
    print(
        "critic: "
        f"layers={critic_num_layers}, hidden_dim={critic_hidden_dim}, "
        f"dropout={critic_dropout}, positive={critic_positive_activation}, "
        f"ensemble={critic_ensemble_size}, seq_enabled={sequence_critic_enabled}"
    )
    ff_block_endpoints = (
        _block_endpoint_indices(len(model.layers), ff_block_size) if ff_blockwise else []
    )
    if ff_blockwise:
        print(
            "ff_blockwise endpoints: "
            + ",".join(str(i + 1) for i in ff_block_endpoints)
            + f" (total_layers={len(model.layers)})"
        )

    risk_head = None
    risk_targets_by_horizon: list[list[float | None]] | None = None
    risk_horizons_effective: list[int] = []
    risk_ticker_effective = str(risk_ticker)
    if risk_head_enabled:
        if ff_layerwise:
            print("risk_head disabled when ff_layerwise is enabled.")
            risk_head_enabled = False
        elif not dates:
            print("risk_head disabled: graphs payload missing dates.")
            risk_head_enabled = False
        else:
            prices_path = build_cfg.get("prices", "data/processed/prices.csv")
            try:
                risk_ticker_effective, ticker_src, ticker_rows = resolve_price_ticker(
                    prices_path=prices_path,
                    requested_ticker=str(risk_ticker),
                    min_rows=max(64, max(risk_horizons)),
                )
                print(
                    "risk ticker: "
                    f"requested={risk_ticker} effective={risk_ticker_effective} "
                    f"source={ticker_src} rows={ticker_rows}"
                )
                risk_targets_by_horizon = []
                risk_horizons_effective = []
                for horizon in risk_horizons:
                    targets_h, _, _ = _compute_risk_targets(
                        prices_path=prices_path,
                        ticker=str(risk_ticker_effective),
                        dates=dates,
                        horizon=int(horizon),
                        standardize=risk_standardize,
                        max_abs_logret=risk_max_abs_logret,
                        cache_dir=str(risk_cache_dir) if risk_cache_dir else None,
                    )
                    risk_targets_by_horizon.append(targets_h)
                    risk_horizons_effective.append(int(horizon))
                if not risk_targets_by_horizon:
                    raise ValueError("no valid risk horizons configured")
            except Exception as exc:
                print(f"risk_head disabled: {exc}")
                risk_head_enabled = False

    if risk_head_enabled:
        print(f"risk_head output dim: {len(risk_horizons_effective)}")
        risk_head = torch.nn.Linear(hidden_dim, len(risk_horizons_effective)).to(device)

    portfolio_head = None
    portfolio_targets: list[float | None] | None = None
    portfolio_ticker_effective = str(portfolio_ticker)
    rank_needs_portfolio_targets = (
        ff_rank_aux_weight > 0 and ff_rank_use_portfolio_targets and not ff_layerwise
    )
    load_portfolio_targets = bool(portfolio_head_enabled or rank_needs_portfolio_targets)
    if load_portfolio_targets:
        if ff_layerwise and portfolio_head_enabled:
            print("portfolio_head disabled when ff_layerwise is enabled.")
            portfolio_head_enabled = False
        if not dates:
            if portfolio_head_enabled:
                print("portfolio_head disabled: graphs payload missing dates.")
                portfolio_head_enabled = False
            if rank_needs_portfolio_targets:
                print("rank_aux portfolio targets unavailable: graphs payload missing dates.")
                rank_needs_portfolio_targets = False
        else:
            prices_path = build_cfg.get("prices", "data/processed/prices.csv")
            try:
                portfolio_ticker_effective, ticker_src, ticker_rows = resolve_price_ticker(
                    prices_path=prices_path,
                    requested_ticker=str(portfolio_ticker),
                    min_rows=max(64, int(portfolio_horizon)),
                )
                print(
                    "portfolio ticker: "
                    f"requested={portfolio_ticker} effective={portfolio_ticker_effective} "
                    f"source={ticker_src} rows={ticker_rows}"
                )
                portfolio_targets, _, _ = _compute_forward_return_targets(
                    prices_path=prices_path,
                    ticker=str(portfolio_ticker_effective),
                    dates=dates,
                    horizon=int(portfolio_horizon),
                    standardize=bool(portfolio_standardize),
                    max_abs_logret=float(portfolio_max_abs_logret),
                    cache_dir=str(portfolio_cache_dir) if portfolio_cache_dir else None,
                )
                if not any(t is not None for t in portfolio_targets):
                    raise ValueError("no valid portfolio targets for configured horizon")
            except Exception as exc:
                if portfolio_head_enabled:
                    print(f"portfolio_head disabled: {exc}")
                    portfolio_head_enabled = False
                if rank_needs_portfolio_targets:
                    print(
                        "rank_aux portfolio targets unavailable; "
                        f"using spread fallback: {exc}"
                    )
                    rank_needs_portfolio_targets = False
                portfolio_targets = None

    if portfolio_head_enabled and portfolio_targets:
        portfolio_head = torch.nn.Linear(hidden_dim, 1).to(device)

    if temp_sweep:
        temps = [float(t.strip()) for t in str(temp_sweep).split(",") if t.strip()]
        if not temps:
            raise ValueError("temp_sweep provided but no valid values found.")
        print(f"Temp sweep: {temps}")
        loader = DataLoader(
            graphs,
            batch_size=min(batch_size, 32),
            shuffle=True,
            drop_last=False,
            num_workers=loader_workers,
        )
        batch = next(iter(loader)).to(device)
        x = batch.x
        edge_weight = getattr(batch, "edge_weight", None)
        h = _forward_encoder(model, x, batch.edge_index, edge_weight=edge_weight)
        for t in temps:
            g = goodness(h, batch.batch, temperature=t, critic=critic).mean().item()
            print(f"goodness_temp={t} -> mean_goodness={g:.4f}")
        return 0

    hall_cfg = _hall_cfg_for_epoch(hall_curr_start if hall_curr_enabled else 1)
    train_shuffle = True
    if bool(sequence_critic_enabled) and bool(sequence_critic_force_chrono):
        train_shuffle = False

    if auto_tune and device.type in ("cuda", "mps"):
        print(f"Auto-tuning batch size for {device.type.upper()}...")
        test_bs = batch_size
        best_bs = None
        while test_bs <= auto_tune_max:
            try:
                model.train()
                _try_batch_size(
                    graphs,
                    model,
                    critic,
                    device,
                    test_bs,
                    loader_workers,
                    neg_mode,
                    noise_std,
                    goodness_target,
                    goodness_temp,
                    hall_cfg,
                    returns_len,
                    summary_dim,
                    ff_multiscale,
                    self_contrastive_temp,
                    self_contrastive_max_graphs,
                    self_contrastive_view_mode,
                    self_contrastive_view_noise_std,
                    self_contrastive_ff_weight,
                    self_contrastive_ff_neg_mode,
                    self_contrastive_ff_noise_std,
                    self_contrastive_ff_target,
                    distance_forward_weight,
                    distance_forward_margin,
                    distance_forward_max_graphs,
                    ff_margin,
                    ff_margin_weight,
                    loader_shuffle=train_shuffle,
                )
                best_bs = test_bs
                test_bs = int(test_bs * auto_tune_factor)
                if test_bs == best_bs:
                    break
            except RuntimeError as exc:
                if _is_oom(exc):
                    break
                raise
            finally:
                empty_device_cache(device)

        if best_bs is None:
            test_bs = max(auto_tune_min, int(batch_size / auto_tune_factor))
            while test_bs >= auto_tune_min:
                try:
                    model.train()
                    _try_batch_size(
                        graphs,
                        model,
                        critic,
                        device,
                        test_bs,
                        loader_workers,
                        neg_mode,
                        noise_std,
                        goodness_target,
                        goodness_temp,
                        hall_cfg,
                        returns_len,
                        summary_dim,
                        ff_multiscale,
                        self_contrastive_temp,
                        self_contrastive_max_graphs,
                        self_contrastive_view_mode,
                        self_contrastive_view_noise_std,
                        self_contrastive_ff_weight,
                        self_contrastive_ff_neg_mode,
                        self_contrastive_ff_noise_std,
                        self_contrastive_ff_target,
                        distance_forward_weight,
                        distance_forward_margin,
                        distance_forward_max_graphs,
                        ff_margin,
                        ff_margin_weight,
                        loader_shuffle=train_shuffle,
                    )
                    best_bs = test_bs
                    break
                except RuntimeError as exc:
                    if _is_oom(exc):
                        test_bs = int(test_bs / auto_tune_factor)
                        continue
                    raise
                finally:
                    empty_device_cache(device)

        if best_bs is not None and best_bs != batch_size:
            print(f"Auto-tune selected batch_size={best_bs}")
            batch_size = best_bs

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train_shuffle,
        "drop_last": False,
        "num_workers": loader_workers,
        "pin_memory": bool(dataloader_pin_memory) if device.type == "cuda" else False,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = bool(dataloader_persistent)
        loader_kwargs["prefetch_factor"] = int(dataloader_prefetch)
        if dataloader_mp_context:
            loader_kwargs["multiprocessing_context"] = dataloader_mp_context
    loader = DataLoader(graphs, **loader_kwargs)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optim_params.extend(p for p in critic.parameters() if p.requires_grad)
    if risk_head is not None:
        optim_params.extend(p for p in risk_head.parameters() if p.requires_grad)
    if portfolio_head is not None:
        optim_params.extend(p for p in portfolio_head.parameters() if p.requires_grad)
    if not optim_params:
        raise ValueError("No trainable parameters. Check freeze_encoder/freeze_critic settings.")
    optim = _build_optimizer(optim_params, lr=lr, device=device, use_fused=fused_optimizer)

    if log_csv:
        log_path = Path(log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            f.write(
                "epoch,loss,g_pos,g_neg,hallucinate_ratio,gate_ratio,hall_hardness,"
                "hall_close_ratio,energy_penalty,risk_loss,portfolio_loss,dist_forward_loss,goodness_target_used,"
                "neg_mix_end_used,neg_gate_margin_used,hall_lr_used,hall_steps_used,"
                "hall_node_fraction_used,rank_aux_loss,"
                "time_neg_gen_s,time_hallucinate_s,time_forward_pos_s,time_forward_neg_s,"
                "time_loss_terms_s,time_optimizer_s,time_econ_eval_s\n"
            )

    epoch_iter = tqdm(
        range(1, epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for epoch in epoch_iter:
        model.train()
        critic_train_mode = any(p.requires_grad for p in critic.parameters()) or bool(
            sequence_critic_enabled
        )
        # cuDNN-backed RNN critics require training mode for backward, even when
        # critic params are frozen and only encoder grads flow through critic outputs.
        critic.train(critic_train_mode)
        epoch_goodness_target = float(goodness_target)
        epoch_neg_mix_end = float(neg_mix_end)
        epoch_neg_gate_margin = float(neg_gate_margin)
        curriculum_phase = _curriculum_phase(epoch, epochs, ff_curriculum_epochs)
        if curriculum_phase == 0:
            epoch_sc_ff_weight = 0.0
        elif curriculum_phase == 1:
            epoch_sc_ff_weight = float(self_contrastive_ff_weight)
        else:
            epoch_sc_ff_weight = 0.5 * float(self_contrastive_ff_weight)
        hall_cfg = _hall_cfg_for_epoch(epoch)
        hall_cfg_layer = _hall_cfg_for_epoch(
            epoch,
            corr_override=layerwise_hall_corr,
            mean_override=layerwise_hall_mean,
            std_override=layerwise_hall_std,
        )
        epoch_hall_lr = float(hall_cfg.lr)
        epoch_hall_steps = int(hall_cfg.steps)
        epoch_hall_node_fraction = float(hall_cfg.node_fraction)
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        batches = 0
        energy_penalty_sum = 0.0
        risk_loss_sum = 0.0
        risk_batches = 0
        portfolio_loss_sum = 0.0
        portfolio_batches = 0
        dist_forward_sum = 0.0
        rank_aux_sum = 0.0
        timing_totals = {
            "neg_gen": 0.0,
            "hallucinate": 0.0,
            "forward_pos": 0.0,
            "forward_neg": 0.0,
            "loss_terms": 0.0,
            "optimizer": 0.0,
            "econ_eval": 0.0,
        }

        hall_used = 0
        total_used = 0
        hall_gated = 0
        hall_close_count = 0
        hall_close_total = 0
        hall_hardness_sum = 0.0
        hall_hardness_count = 0

        for batch in loader:
            try:
                batch = batch.to(device)
            except Exception as exc:
                if device.type == "mps":
                    raise RuntimeError(
                        "MPS device placement failed for PyG tensors. "
                        "If you hit unsupported ops, rerun with --device cpu."
                    ) from exc
                raise
            x = batch.x
            edge_weight = getattr(batch, "edge_weight", None)

            step_idx = batches + 1
            if neg_mode == "schedule":
                use_mode = "shuffle" if epoch <= neg_warmup_epochs else "hallucinate"
            elif neg_mode == "mix":
                if epoch <= neg_warmup_epochs:
                    use_mode = "shuffle"
                else:
                    ramp = max(1, neg_mix_ramp_epochs)
                    progress = min(1.0, (epoch - neg_warmup_epochs) / ramp)
                    p_hall = neg_mix_start + progress * (neg_mix_end - neg_mix_start)
                    use_mode = "hallucinate" if random.random() < p_hall else "shuffle"
            else:
                use_mode = neg_mode

            use_mode = _pick_curriculum_neg_mode(
                use_mode,
                epoch=epoch,
                epochs=epochs,
                mix_modes=ff_neg_mix,
                mix_weights=ff_neg_mix_weights,
                phase_ratios=ff_curriculum_epochs,
            )
            if use_mode == "hallucinate" and not _should_use_hallucination(
                epoch=epoch,
                step_idx=step_idx,
                warmup_epochs=ff_hall_warmup_epochs,
                every_n_batches=ff_hall_every_n_batches,
            ):
                use_mode = "shuffle+noise"
            apply_distance = (
                distance_forward_weight > 0
                and (step_idx % distance_forward_interval == 0)
            )
            step_scaler = scaler if (amp_enabled and (use_mode == "self_contrastive" or ff_layerwise)) else None
            batch_t0 = time.perf_counter()
            timing_before = timing_totals.copy()

            if ff_multiscale:
                hall_active = False
                dist_loss_val = 0.0

                if use_mode == "self_contrastive":
                    total_used += 1
                    t_neg_gen = time.perf_counter()
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        x_view = _make_self_contrastive_view(
                            x,
                            batch.batch,
                            view_mode=self_contrastive_view_mode,
                            view_noise_std=self_contrastive_view_noise_std,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                    timing_totals["neg_gen"] += time.perf_counter() - t_neg_gen

                    if ff_concat_posneg:
                        t_fwd_cat = time.perf_counter()
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            layers_pos, layers_view = _concat_forward_pos_neg(
                                model=model,
                                x_pos=x,
                                x_neg=x_view,
                                edge_index=batch.edge_index,
                                edge_weight=edge_weight,
                                batch_nodes=batch.batch,
                                return_all=True,
                            )
                        dt_cat = time.perf_counter() - t_fwd_cat
                        timing_totals["forward_pos"] += 0.5 * dt_cat
                        timing_totals["forward_neg"] += 0.5 * dt_cat
                    else:
                        t_fwd_pos = time.perf_counter()
                        if step_scaler is not None:
                            with _autocast_if_needed(True, amp_dtype):
                                layers_pos = _forward_encoder(
                                    model,
                                    x,
                                    batch.edge_index,
                                    edge_weight=edge_weight,
                                    return_all=True,
                                )
                        else:
                            layers_pos = _forward_encoder(
                                model,
                                x,
                                batch.edge_index,
                                edge_weight=edge_weight,
                                return_all=True,
                            )
                        timing_totals["forward_pos"] += time.perf_counter() - t_fwd_pos
                        t_fwd_neg = time.perf_counter()
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            layers_view = _forward_encoder(
                                model,
                                x_view,
                                batch.edge_index,
                                edge_weight=edge_weight,
                                return_all=True,
                            )
                        timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg

                    t_loss_terms = time.perf_counter()
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        batch_loss = 0.0
                        g_pos_last = 0.0
                        g_neg_last = 0.0
                        z_pos_last = None
                        z_neg_last = None
                        for h_p, h_v in zip(layers_pos, layers_view):
                            sc_loss, pos_score, neg_score, z_pos, z_view = _self_contrastive_batch_loss(
                                h_p,
                                h_v,
                                batch.batch,
                                temperature=self_contrastive_temp,
                                max_graphs=self_contrastive_max_graphs,
                            )
                            batch_loss = batch_loss + sc_loss
                            g_pos_last = float(pos_score.detach())
                            g_neg_last = float(neg_score.detach())
                            z_pos_last = z_pos
                            z_neg_last = permute_graph_embeddings(z_view)
                        batch_loss = batch_loss / max(1, len(layers_pos))
                        if apply_distance and z_pos_last is not None and z_neg_last is not None:
                            dist_loss_val = pairwise_distance_forward_loss(
                                z_pos_last,
                                z_neg_last,
                                margin=distance_forward_margin,
                                max_graphs=distance_forward_max_graphs,
                            )
                            batch_loss = batch_loss + distance_forward_weight * dist_loss_val
                        if epoch_sc_ff_weight > 0:
                            t_neg_aux = time.perf_counter()
                            x_neg_aux = _make_negatives(
                                model,
                                x,
                                batch.batch,
                                batch.edge_index,
                                getattr(batch, "edge_attr", None),
                                edge_weight,
                                self_contrastive_ff_neg_mode,
                                self_contrastive_ff_noise_std,
                                hall_cfg,
                                critic=critic,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            timing_totals["neg_gen"] += time.perf_counter() - t_neg_aux
                            t_fwd_neg_aux = time.perf_counter()
                            layers_neg_aux = _forward_encoder(
                                model,
                                x_neg_aux,
                                batch.edge_index,
                                edge_weight=edge_weight,
                                return_all=True,
                            )
                            timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg_aux
                            g_pos_aux = goodness(
                                layers_pos[-1], batch.batch, temperature=goodness_temp, critic=critic
                            )
                            g_neg_aux = goodness(
                                layers_neg_aux[-1], batch.batch, temperature=goodness_temp, critic=critic
                            )
                            ff_aux = ff_loss(
                                g_pos_aux,
                                g_neg_aux,
                                target=self_contrastive_ff_target,
                                margin=ff_margin,
                                margin_weight=ff_margin_weight,
                            )
                            batch_loss = batch_loss + epoch_sc_ff_weight * ff_aux
                    timing_totals["loss_terms"] += time.perf_counter() - t_loss_terms
                else:
                    t_fwd_pos = time.perf_counter()
                    if step_scaler is not None:
                        with _autocast_if_needed(True, amp_dtype):
                            layers_pos = _forward_encoder(
                                model,
                                x,
                                batch.edge_index,
                                edge_weight=edge_weight,
                                return_all=True,
                            )
                    else:
                        layers_pos = _forward_encoder(
                            model,
                            x,
                            batch.edge_index,
                            edge_weight=edge_weight,
                            return_all=True,
                        )
                    timing_totals["forward_pos"] += time.perf_counter() - t_fwd_pos
                    hall_active = use_mode == "hallucinate"
                    if use_mode == "hallucinate":
                        t_neg_gen = time.perf_counter()
                        x_neg_hall = _make_negatives(
                            model,
                            x,
                            batch.batch,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            edge_weight,
                            use_mode,
                            noise_std,
                            hall_cfg,
                            critic=critic,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        dt_neg = time.perf_counter() - t_neg_gen
                        timing_totals["neg_gen"] += dt_neg
                        timing_totals["hallucinate"] += dt_neg
                        if hall_min_delta and hall_min_delta > 0:
                            delta = (
                                x_neg_hall[:, :returns_len] - x[:, :returns_len]
                            ).abs().mean()
                            hall_close_total += 1
                            if float(delta) < hall_min_delta:
                                hall_close_count += 1
                                x_neg_hall = make_negative(
                                    x,
                                    batch.batch,
                                    mode="shuffle+noise",
                                    noise_std=max(float(noise_std), float(hall_fallback_noise)),
                                    window_len=returns_len,
                                    summary_dim=summary_dim,
                                )
                        hall_used += 1
                    else:
                        t_neg_gen = time.perf_counter()
                        x_neg_hall = _make_negatives(
                            model,
                            x,
                            batch.batch,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            edge_weight,
                            use_mode,
                            noise_std,
                            hall_cfg,
                            critic=critic,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        timing_totals["neg_gen"] += time.perf_counter() - t_neg_gen
                    total_used += 1

                    t_neg_time = time.perf_counter()
                    x_neg_time = make_negative(
                        x,
                        batch.batch,
                        mode="time_flip",
                        noise_std=noise_std,
                        window_len=returns_len,
                        summary_dim=summary_dim,
                    )
                    timing_totals["neg_gen"] += time.perf_counter() - t_neg_time

                    t_fwd_neg = time.perf_counter()
                    layers_neg_h = _forward_encoder(model, 
                        x_neg_hall, batch.edge_index, edge_weight=edge_weight, return_all=True
                    )
                    layers_neg_t = _forward_encoder(model, 
                        x_neg_time, batch.edge_index, edge_weight=edge_weight, return_all=True
                    )
                    timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg

                    if use_mode == "hallucinate":
                        g_pos_probe = goodness(
                            layers_pos[-1], batch.batch, temperature=goodness_temp, critic=critic
                        ).mean().item()
                        g_neg_probe = goodness(
                            layers_neg_h[-1], batch.batch, temperature=goodness_temp, critic=critic
                        ).mean().item()
                        if g_neg_probe > g_pos_probe + neg_gate_margin:
                            x_neg_hall = make_negative(
                                x,
                                batch.batch,
                                mode="shuffle",
                                noise_std=noise_std,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            hall_used -= 1
                            hall_gated += 1
                            hall_active = False
                            t_fwd_neg_gate = time.perf_counter()
                            layers_neg_h = _forward_encoder(model, 
                                x_neg_hall,
                                batch.edge_index,
                                edge_weight=edge_weight,
                                return_all=True,
                            )
                            timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg_gate

                    t_loss_terms = time.perf_counter()
                    batch_loss = 0.0
                    g_pos_last = 0.0
                    g_neg_h_last = 0.0
                    g_neg_t_last = 0.0
                    for h_p, h_n_h, h_n_t in zip(layers_pos, layers_neg_h, layers_neg_t):
                        g_p = goodness(h_p, batch.batch, temperature=goodness_temp, critic=critic)
                        g_n_h = goodness(h_n_h, batch.batch, temperature=goodness_temp, critic=critic)
                        g_n_t = goodness(h_n_t, batch.batch, temperature=goodness_temp, critic=critic)
                        g_pos_last = g_p.mean().item()
                        g_neg_h_last = g_n_h.mean().item()
                        g_neg_t_last = g_n_t.mean().item()
                        batch_loss += ff_loss(
                            g_p,
                            g_n_h,
                            target=goodness_target,
                            margin=ff_margin,
                            margin_weight=ff_margin_weight,
                        )
                        batch_loss += ff_loss(
                            g_p,
                            g_n_t,
                            target=goodness_target,
                            margin=ff_margin,
                            margin_weight=ff_margin_weight,
                        )
                    batch_loss = batch_loss / max(1, len(layers_pos))

                    if apply_distance:
                        z_pos = global_mean_pool(layers_pos[-1], batch.batch)
                        z_neg_h = global_mean_pool(layers_neg_h[-1], batch.batch)
                        z_neg_t = global_mean_pool(layers_neg_t[-1], batch.batch)
                        dist_loss_h = pairwise_distance_forward_loss(
                            z_pos,
                            z_neg_h,
                            margin=distance_forward_margin,
                            max_graphs=distance_forward_max_graphs,
                        )
                        dist_loss_t = pairwise_distance_forward_loss(
                            z_pos,
                            z_neg_t,
                            margin=distance_forward_margin,
                            max_graphs=distance_forward_max_graphs,
                        )
                        dist_loss_val = 0.5 * (dist_loss_h + dist_loss_t)
                        batch_loss = batch_loss + distance_forward_weight * dist_loss_val

                    g_neg_last = (g_neg_h_last + g_neg_t_last) / 2.0
                    timing_totals["loss_terms"] += time.perf_counter() - t_loss_terms

                energy_penalty_val = 0.0
                energy_penalty_weight_eff = (
                    energy_penalty_weight * self_contrastive_energy_penalty_scale
                    if use_mode == "self_contrastive"
                    else energy_penalty_weight
                )
                if energy_penalty_weight_eff > 0:
                    if energy_penalty_mode == "all":
                        energy_penalty_val = sum(
                            h.pow(2).mean() for h in layers_pos
                        ) / max(1, len(layers_pos))
                    else:
                        energy_penalty_val = layers_pos[-1].pow(2).mean()
                    batch_loss = batch_loss + energy_penalty_weight_eff * energy_penalty_val

                risk_loss_val = None
                if risk_head is not None and risk_targets_by_horizon is not None:
                    embed = global_mean_pool(layers_pos[-1], batch.batch)
                    risk_loss_val = _compute_multi_horizon_risk_loss(
                        risk_head=risk_head,
                        embeddings=embed,
                        graph_idx=batch.graph_idx,
                        risk_targets_by_horizon=risk_targets_by_horizon,
                        device=device,
                        risk_loss_type=str(risk_loss_type).strip().lower(),
                    )
                    if risk_loss_val is not None:
                        batch_loss = batch_loss + risk_loss_weight * risk_loss_val
                portfolio_loss_val = None
                if portfolio_head is not None and portfolio_targets is not None:
                    embed = global_mean_pool(layers_pos[-1], batch.batch)
                    portfolio_loss_val = _compute_portfolio_head_loss(
                        portfolio_head=portfolio_head,
                        embeddings=embed,
                        graph_idx=batch.graph_idx,
                        portfolio_targets=portfolio_targets,
                        device=device,
                        loss_type=portfolio_loss_type,
                    )
                    if portfolio_loss_val is not None:
                        batch_loss = batch_loss + portfolio_loss_weight * portfolio_loss_val
                rank_aux_val = None
                if ff_rank_aux_weight > 0:
                    g_rank = goodness(
                        layers_pos[-1], batch.batch, temperature=goodness_temp, critic=critic
                    )
                    rank_aux_val = _goodness_rank_alignment_loss(
                        g_rank,
                        graph_idx=batch.graph_idx,
                        portfolio_targets=portfolio_targets,
                        device=device,
                    )
                    if rank_aux_val is None:
                        rank_aux_val = rank_spread_loss(g_rank)
                    batch_loss = batch_loss + ff_rank_aux_weight * rank_aux_val

                t_opt = time.perf_counter()
                _optimizer_step(
                    optim=optim,
                    loss=batch_loss,
                    grad_clip=grad_clip,
                    clip_params=optim_params,
                    scaler=step_scaler,
                )
                timing_totals["optimizer"] += time.perf_counter() - t_opt

                total_loss += batch_loss.item()
                total_pos += g_pos_last
                total_neg += g_neg_last
                if hall_active:
                    hall_hardness_sum += (g_neg_last - g_pos_last)
                    hall_hardness_count += 1
                if energy_penalty_weight_eff > 0:
                    energy_penalty_sum += float(energy_penalty_val.detach())
                if risk_loss_val is not None:
                    risk_loss_sum += float(risk_loss_val.detach())
                    risk_batches += 1
                if portfolio_loss_val is not None:
                    portfolio_loss_sum += float(portfolio_loss_val.detach())
                    portfolio_batches += 1
                if rank_aux_val is not None:
                    rank_aux_sum += float(rank_aux_val.detach())
                if distance_forward_weight > 0 and isinstance(dist_loss_val, torch.Tensor):
                    dist_forward_sum += float(dist_loss_val.detach())
            elif ff_layerwise:
                if ff_blockwise:
                    block_mode = "shuffle" if use_mode == "self_contrastive" else use_mode
                    hall_active = block_mode == "hallucinate"

                    if hall_active:
                        t_neg_gen = time.perf_counter()
                        x_neg = _make_negatives(
                            model,
                            x,
                            batch.batch,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            edge_weight,
                            "hallucinate",
                            layerwise_noise_std,
                            hall_cfg_layer,
                            critic=critic,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        dt_neg = time.perf_counter() - t_neg_gen
                        timing_totals["neg_gen"] += dt_neg
                        timing_totals["hallucinate"] += dt_neg
                        if hall_min_delta and hall_min_delta > 0:
                            delta = (x_neg[:, :returns_len] - x[:, :returns_len]).abs().mean()
                            hall_close_total += 1
                            if float(delta) < hall_min_delta:
                                hall_close_count += 1
                                x_neg = make_negative(
                                    x,
                                    batch.batch,
                                    mode="shuffle+noise",
                                    noise_std=max(float(layerwise_noise_std), float(hall_fallback_noise)),
                                    window_len=returns_len,
                                    summary_dim=summary_dim,
                                )
                        hall_used += 1
                    else:
                        t_neg_gen = time.perf_counter()
                        x_neg = _make_negatives(
                            model,
                            x,
                            batch.batch,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            edge_weight,
                            layerwise_neg_mode,
                            layerwise_noise_std,
                            hall_cfg_layer,
                            critic=critic,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        timing_totals["neg_gen"] += time.perf_counter() - t_neg_gen
                    total_used += 1

                    if ff_layer_cache and ff_concat_posneg and not hall_active:
                        t_cat = time.perf_counter()
                        layers_pos, layers_neg = _concat_forward_pos_neg(
                            model=model,
                            x_pos=x,
                            x_neg=x_neg,
                            edge_index=batch.edge_index,
                            edge_weight=edge_weight,
                            batch_nodes=batch.batch,
                            return_all=True,
                        )
                        dt_cat = time.perf_counter() - t_cat
                        timing_totals["forward_pos"] += 0.5 * dt_cat
                        timing_totals["forward_neg"] += 0.5 * dt_cat
                    else:
                        t_fwd_pos = time.perf_counter()
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            layers_pos = _forward_encoder(model, x, batch.edge_index, edge_weight=edge_weight, return_all=True)
                        timing_totals["forward_pos"] += time.perf_counter() - t_fwd_pos
                        t_fwd_neg = time.perf_counter()
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            layers_neg = _forward_encoder(model, 
                                x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True
                            )
                        timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg
                    if hall_active:
                        last_idx = ff_block_endpoints[-1]
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            g_pos_probe = goodness(
                                layers_pos[last_idx], batch.batch, temperature=goodness_temp, critic=critic
                            ).mean().item()
                            g_neg_probe = goodness(
                                layers_neg[last_idx], batch.batch, temperature=goodness_temp, critic=critic
                            ).mean().item()
                        if g_neg_probe > g_pos_probe + neg_gate_margin:
                            x_neg = make_negative(
                                x,
                                batch.batch,
                                mode="shuffle",
                                noise_std=noise_std,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            hall_used -= 1
                            hall_gated += 1
                            hall_active = False
                            t_fwd_neg = time.perf_counter()
                            with _autocast_if_needed(step_scaler is not None, amp_dtype):
                                layers_neg = _forward_encoder(model, 
                                    x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True
                                )
                            timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg

                    block_loss = 0.0
                    block_gpos = 0.0
                    block_gneg = 0.0
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        for li in ff_block_endpoints:
                            g_pos = goodness(
                                layers_pos[li],
                                batch.batch,
                                temperature=goodness_temp,
                                critic=critic,
                            )
                            g_neg = goodness(layers_neg[li], batch.batch, temperature=goodness_temp, critic=critic)
                            block_loss = block_loss + ff_loss(
                                g_pos,
                                g_neg,
                                target=goodness_target,
                                margin=ff_margin,
                                margin_weight=ff_margin_weight,
                            )
                            block_gpos += g_pos.mean().item()
                            block_gneg += g_neg.mean().item()
                    block_loss = block_loss / max(1, len(ff_block_endpoints))
                    t_opt = time.perf_counter()
                    _optimizer_step(
                        optim=optim,
                        loss=block_loss,
                        grad_clip=grad_clip,
                        clip_params=optim_params,
                        scaler=step_scaler,
                    )
                    timing_totals["optimizer"] += time.perf_counter() - t_opt

                    avg_g_pos = block_gpos / max(1, len(ff_block_endpoints))
                    avg_g_neg = block_gneg / max(1, len(ff_block_endpoints))
                    total_loss += block_loss.item()
                    total_pos += avg_g_pos
                    total_neg += avg_g_neg
                    if hall_active:
                        hall_hardness_sum += (avg_g_neg - avg_g_pos)
                        hall_hardness_count += 1
                else:
                    x_in = x
                    layer_losses = 0.0
                    layer_gpos = 0.0
                    layer_gneg = 0.0
                    for li in range(len(model.layers)):
                        layer_mode = use_mode
                        if use_mode == "hallucinate" and li > 0:
                            layer_mode = "shuffle"
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            t_fwd_pos = time.perf_counter()
                            h_pos = model.forward_layer(x_in, batch.edge_index, edge_weight, li)
                            timing_totals["forward_pos"] += time.perf_counter() - t_fwd_pos
                            g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)

                        hall_active = layer_mode == "hallucinate"
                        h_neg = None
                        g_neg = None
                        if layer_mode == "hallucinate":
                            forward_fn = lambda x_var, li=li: model.forward_layer(
                                x_var, batch.edge_index, edge_weight, li
                            )
                            t_neg_gen = time.perf_counter()
                            x_neg = _make_negatives(
                                model,
                                x_in,
                                batch.batch,
                                batch.edge_index,
                                getattr(batch, "edge_attr", None),
                                edge_weight,
                                "hallucinate",
                                layerwise_noise_std,
                                hall_cfg_layer,
                                critic=critic,
                                forward_fn=forward_fn,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            dt_neg = time.perf_counter() - t_neg_gen
                            timing_totals["neg_gen"] += dt_neg
                            timing_totals["hallucinate"] += dt_neg
                            if hall_min_delta and hall_min_delta > 0:
                                delta = (
                                    x_neg[:, :returns_len] - x_in[:, :returns_len]
                                ).abs().mean()
                                hall_close_total += 1
                                if float(delta) < hall_min_delta:
                                    hall_close_count += 1
                                    x_neg = make_negative(
                                        x_in,
                                        batch.batch,
                                        mode="shuffle+noise",
                                        noise_std=max(float(layerwise_noise_std), float(hall_fallback_noise)),
                                        window_len=returns_len,
                                        summary_dim=summary_dim,
                                    )
                            hall_used += 1
                        else:
                            t_neg_gen = time.perf_counter()
                            x_neg = _make_negatives(
                                model,
                                x_in,
                                batch.batch,
                                batch.edge_index,
                                getattr(batch, "edge_attr", None),
                                edge_weight,
                                layerwise_neg_mode,
                                layerwise_noise_std,
                                hall_cfg_layer,
                                critic=critic,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            timing_totals["neg_gen"] += time.perf_counter() - t_neg_gen
                        total_used += 1

                        if layer_mode == "hallucinate":
                            with _autocast_if_needed(step_scaler is not None, amp_dtype):
                                h_neg_probe = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                                g_neg_probe_t = goodness(
                                    h_neg_probe,
                                    batch.batch,
                                    temperature=goodness_temp,
                                    critic=critic,
                                )
                                g_neg_probe = g_neg_probe_t.mean().item()
                            g_pos_probe = g_pos.mean().item()
                            if g_neg_probe > g_pos_probe + neg_gate_margin:
                                x_neg = make_negative(
                                    x_in,
                                    batch.batch,
                                    mode="shuffle",
                                    noise_std=noise_std,
                                    window_len=returns_len,
                                    summary_dim=summary_dim,
                                )
                                hall_used -= 1
                                hall_gated += 1
                                hall_active = False
                            else:
                                h_neg = h_neg_probe
                                g_neg = g_neg_probe_t

                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            if h_neg is None:
                                t_fwd_neg = time.perf_counter()
                                h_neg = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                                timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg
                                g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp, critic=critic)
                            loss = ff_loss(
                                g_pos,
                                g_neg,
                                target=goodness_target,
                                margin=ff_margin,
                                margin_weight=ff_margin_weight,
                            )
                        t_opt = time.perf_counter()
                        _optimizer_step(
                            optim=optim,
                            loss=loss,
                            grad_clip=grad_clip,
                            clip_params=optim_params,
                            scaler=step_scaler,
                        )
                        timing_totals["optimizer"] += time.perf_counter() - t_opt

                        layer_losses += loss.item()
                        layer_gpos += g_pos.mean().item()
                        layer_gneg += g_neg.mean().item()
                        if hall_active:
                            hall_hardness_sum += (g_neg.mean().item() - g_pos.mean().item())
                            hall_hardness_count += 1
                        x_in = h_pos.detach()

                    total_loss += layer_losses / len(model.layers)
                    total_pos += layer_gpos / len(model.layers)
                    total_neg += layer_gneg / len(model.layers)
            else:
                hall_active = False
                dist_loss_val = 0.0

                if use_mode == "self_contrastive":
                    total_used += 1
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        t_neg_gen = time.perf_counter()
                        x_view = _make_self_contrastive_view(
                            x,
                            batch.batch,
                            view_mode=self_contrastive_view_mode,
                            view_noise_std=self_contrastive_view_noise_std,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        timing_totals["neg_gen"] += time.perf_counter() - t_neg_gen
                        if ff_concat_posneg:
                            t_fwd_cat = time.perf_counter()
                            h_pos, h_view = _concat_forward_pos_neg(
                                model=model,
                                x_pos=x,
                                x_neg=x_view,
                                edge_index=batch.edge_index,
                                edge_weight=edge_weight,
                                batch_nodes=batch.batch,
                                return_all=False,
                            )
                            dt_cat = time.perf_counter() - t_fwd_cat
                            timing_totals["forward_pos"] += 0.5 * dt_cat
                            timing_totals["forward_neg"] += 0.5 * dt_cat
                        else:
                            t_fwd_pos = time.perf_counter()
                            h_pos = _forward_encoder(model, x, batch.edge_index, edge_weight=edge_weight)
                            timing_totals["forward_pos"] += time.perf_counter() - t_fwd_pos
                            t_fwd_neg = time.perf_counter()
                            h_view = _forward_encoder(model, x_view, batch.edge_index, edge_weight=edge_weight)
                            timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg
                        t_loss_terms = time.perf_counter()
                        loss, pos_score, neg_score, z_pos, z_view = _self_contrastive_batch_loss(
                            h_pos,
                            h_view,
                            batch.batch,
                            temperature=self_contrastive_temp,
                            max_graphs=self_contrastive_max_graphs,
                        )
                        g_pos_val = float(pos_score.detach())
                        g_neg_val = float(neg_score.detach())
                        if apply_distance:
                            z_neg_dist = permute_graph_embeddings(z_view)
                            dist_loss_val = pairwise_distance_forward_loss(
                                z_pos,
                                z_neg_dist,
                                margin=distance_forward_margin,
                                max_graphs=distance_forward_max_graphs,
                            )
                            loss = loss + distance_forward_weight * dist_loss_val
                        if epoch_sc_ff_weight > 0:
                            t_neg_aux = time.perf_counter()
                            x_neg_aux = _make_negatives(
                                model,
                                x,
                                batch.batch,
                                batch.edge_index,
                                getattr(batch, "edge_attr", None),
                                edge_weight,
                                self_contrastive_ff_neg_mode,
                                self_contrastive_ff_noise_std,
                                hall_cfg,
                                critic=critic,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            timing_totals["neg_gen"] += time.perf_counter() - t_neg_aux
                            t_fwd_neg_aux = time.perf_counter()
                            h_neg_aux = _forward_encoder(model, x_neg_aux, batch.edge_index, edge_weight=edge_weight)
                            timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg_aux
                            g_pos_aux = goodness(
                                h_pos,
                                batch.batch,
                                temperature=goodness_temp,
                                critic=critic,
                            )
                            g_neg_aux = goodness(h_neg_aux, batch.batch, temperature=goodness_temp, critic=critic)
                            ff_aux = ff_loss(
                                g_pos_aux,
                                g_neg_aux,
                                target=self_contrastive_ff_target,
                                margin=ff_margin,
                                margin_weight=ff_margin_weight,
                            )
                            loss = loss + epoch_sc_ff_weight * ff_aux
                        timing_totals["loss_terms"] += time.perf_counter() - t_loss_terms
                else:
                    t_loss_terms = time.perf_counter()
                    hall_active = use_mode == "hallucinate"
                    h_pos = None
                    g_pos = None
                    h_neg = None
                    g_neg = None
                    if hall_active:
                        t_fwd_pos = time.perf_counter()
                        h_pos = _forward_encoder(model, x, batch.edge_index, edge_weight=edge_weight)
                        timing_totals["forward_pos"] += time.perf_counter() - t_fwd_pos
                        g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)
                    if use_mode == "hallucinate":
                        t_neg_gen = time.perf_counter()
                        x_neg = _make_negatives(
                            model,
                            x,
                            batch.batch,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            edge_weight,
                            use_mode,
                            noise_std,
                            hall_cfg,
                            critic=critic,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        dt_neg = time.perf_counter() - t_neg_gen
                        timing_totals["neg_gen"] += dt_neg
                        timing_totals["hallucinate"] += dt_neg
                        if hall_min_delta and hall_min_delta > 0:
                            delta = (x_neg[:, :returns_len] - x[:, :returns_len]).abs().mean()
                            hall_close_total += 1
                            if float(delta) < hall_min_delta:
                                hall_close_count += 1
                                x_neg = make_negative(
                                    x,
                                    batch.batch,
                                    mode="shuffle+noise",
                                    noise_std=max(float(noise_std), float(hall_fallback_noise)),
                                    window_len=returns_len,
                                    summary_dim=summary_dim,
                                )
                            hall_used += 1
                    else:
                        t_neg_gen = time.perf_counter()
                        x_neg = _make_negatives(
                            model,
                            x,
                            batch.batch,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            edge_weight,
                            use_mode,
                            noise_std,
                            hall_cfg,
                            critic=critic,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        timing_totals["neg_gen"] += time.perf_counter() - t_neg_gen
                    total_used += 1

                    if use_mode == "hallucinate":
                        t_fwd_neg_probe = time.perf_counter()
                        h_neg_probe = _forward_encoder(model, x_neg, batch.edge_index, edge_weight=edge_weight)
                        timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg_probe
                        g_neg_probe_t = goodness(
                            h_neg_probe, batch.batch, temperature=goodness_temp, critic=critic
                        )
                        g_neg_probe = g_neg_probe_t.mean().item()
                        g_pos_probe = g_pos.mean().item()
                        if g_neg_probe > g_pos_probe + neg_gate_margin:
                            x_neg = make_negative(
                                x,
                                batch.batch,
                                mode="shuffle",
                                noise_std=noise_std,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            hall_used -= 1
                            hall_gated += 1
                            hall_active = False
                        else:
                            h_neg = h_neg_probe
                            g_neg = g_neg_probe_t
                    if hall_active or not ff_concat_posneg:
                        if h_pos is None:
                            t_fwd_pos = time.perf_counter()
                            h_pos = _forward_encoder(model, x, batch.edge_index, edge_weight=edge_weight)
                            timing_totals["forward_pos"] += time.perf_counter() - t_fwd_pos
                        if h_neg is None:
                            t_fwd_neg = time.perf_counter()
                            h_neg = _forward_encoder(model, x_neg, batch.edge_index, edge_weight=edge_weight)
                            timing_totals["forward_neg"] += time.perf_counter() - t_fwd_neg
                    else:
                        t_fwd_cat = time.perf_counter()
                        h_pos, h_neg = _concat_forward_pos_neg(
                            model=model,
                            x_pos=x,
                            x_neg=x_neg,
                            edge_index=batch.edge_index,
                            edge_weight=edge_weight,
                            batch_nodes=batch.batch,
                            return_all=False,
                        )
                        dt_cat = time.perf_counter() - t_fwd_cat
                        timing_totals["forward_pos"] += 0.5 * dt_cat
                        timing_totals["forward_neg"] += 0.5 * dt_cat
                    if g_pos is None:
                        g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)
                    if g_neg is None:
                        g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp, critic=critic)

                    loss = ff_loss(
                        g_pos,
                        g_neg,
                        target=goodness_target,
                        margin=ff_margin,
                        margin_weight=ff_margin_weight,
                    )
                    g_pos_val = g_pos.mean().item()
                    g_neg_val = g_neg.mean().item()
                    if apply_distance:
                        z_pos = global_mean_pool(h_pos, batch.batch)
                        z_neg = global_mean_pool(h_neg, batch.batch)
                        dist_loss_val = pairwise_distance_forward_loss(
                            z_pos,
                            z_neg,
                            margin=distance_forward_margin,
                            max_graphs=distance_forward_max_graphs,
                        )
                        loss = loss + distance_forward_weight * dist_loss_val
                    timing_totals["loss_terms"] += time.perf_counter() - t_loss_terms

                energy_penalty_val = 0.0
                energy_penalty_weight_eff = (
                    energy_penalty_weight * self_contrastive_energy_penalty_scale
                    if use_mode == "self_contrastive"
                    else energy_penalty_weight
                )
                if energy_penalty_weight_eff > 0:
                    energy_penalty_val = h_pos.pow(2).mean()
                    loss = loss + energy_penalty_weight_eff * energy_penalty_val

                risk_loss_val = None
                if risk_head is not None and risk_targets_by_horizon is not None:
                    embed = global_mean_pool(h_pos, batch.batch)
                    risk_loss_val = _compute_multi_horizon_risk_loss(
                        risk_head=risk_head,
                        embeddings=embed,
                        graph_idx=batch.graph_idx,
                        risk_targets_by_horizon=risk_targets_by_horizon,
                        device=device,
                        risk_loss_type=str(risk_loss_type).strip().lower(),
                    )
                    if risk_loss_val is not None:
                        loss = loss + risk_loss_weight * risk_loss_val
                portfolio_loss_val = None
                if portfolio_head is not None and portfolio_targets is not None:
                    embed = global_mean_pool(h_pos, batch.batch)
                    portfolio_loss_val = _compute_portfolio_head_loss(
                        portfolio_head=portfolio_head,
                        embeddings=embed,
                        graph_idx=batch.graph_idx,
                        portfolio_targets=portfolio_targets,
                        device=device,
                        loss_type=portfolio_loss_type,
                    )
                    if portfolio_loss_val is not None:
                        loss = loss + portfolio_loss_weight * portfolio_loss_val
                rank_aux_val = None
                if ff_rank_aux_weight > 0:
                    g_rank = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)
                    rank_aux_val = _goodness_rank_alignment_loss(
                        g_rank,
                        graph_idx=batch.graph_idx,
                        portfolio_targets=portfolio_targets,
                        device=device,
                    )
                    if rank_aux_val is None:
                        rank_aux_val = rank_spread_loss(g_rank)
                    loss = loss + ff_rank_aux_weight * rank_aux_val
                t_opt = time.perf_counter()
                _optimizer_step(
                    optim=optim,
                    loss=loss,
                    grad_clip=grad_clip,
                    clip_params=optim_params,
                    scaler=step_scaler,
                )
                timing_totals["optimizer"] += time.perf_counter() - t_opt

                total_loss += loss.item()
                total_pos += g_pos_val
                total_neg += g_neg_val
                if hall_active:
                    hall_hardness_sum += (g_neg_val - g_pos_val)
                    hall_hardness_count += 1
                if energy_penalty_weight_eff > 0:
                    energy_penalty_sum += float(energy_penalty_val.detach())
                if risk_loss_val is not None:
                    risk_loss_sum += float(risk_loss_val.detach())
                    risk_batches += 1
                if portfolio_loss_val is not None:
                    portfolio_loss_sum += float(portfolio_loss_val.detach())
                    portfolio_batches += 1
                if rank_aux_val is not None:
                    rank_aux_sum += float(rank_aux_val.detach())
                if distance_forward_weight > 0 and isinstance(dist_loss_val, torch.Tensor):
                    dist_forward_sum += float(dist_loss_val.detach())
            batch_elapsed = time.perf_counter() - batch_t0
            known_elapsed = sum(
                max(0.0, timing_totals[k] - timing_before.get(k, 0.0))
                for k in ("neg_gen", "hallucinate", "forward_pos", "forward_neg", "loss_terms", "optimizer")
            )
            if batch_elapsed > known_elapsed:
                timing_totals["loss_terms"] += (batch_elapsed - known_elapsed)
            batches += 1

        hall_ratio = hall_used / total_used if total_used else 0.0
        gate_ratio = hall_gated / total_used if total_used else 0.0
        hall_close_ratio = hall_close_count / hall_close_total if hall_close_total else 0.0
        hall_hardness = hall_hardness_sum / hall_hardness_count if hall_hardness_count else 0.0
        energy_penalty_epoch = energy_penalty_sum / batches if batches else 0.0
        risk_loss_epoch = risk_loss_sum / risk_batches if risk_batches else 0.0
        portfolio_loss_epoch = portfolio_loss_sum / portfolio_batches if portfolio_batches else 0.0
        dist_forward_epoch = dist_forward_sum / batches if batches else 0.0
        rank_aux_epoch = rank_aux_sum / batches if batches else 0.0
        time_neg_gen_epoch = timing_totals["neg_gen"] / batches if batches else 0.0
        time_hall_epoch = timing_totals["hallucinate"] / batches if batches else 0.0
        time_fwd_pos_epoch = timing_totals["forward_pos"] / batches if batches else 0.0
        time_fwd_neg_epoch = timing_totals["forward_neg"] / batches if batches else 0.0
        time_loss_epoch = timing_totals["loss_terms"] / batches if batches else 0.0
        time_opt_epoch = timing_totals["optimizer"] / batches if batches else 0.0
        time_econ_epoch = timing_totals["econ_eval"] / batches if batches else 0.0
        epoch_loss = total_loss / batches if batches else 0.0
        epoch_pos = total_pos / batches if batches else 0.0
        epoch_neg = total_neg / batches if batches else 0.0

        target_updated = False
        if adaptive_target_enabled and batches and epoch >= adaptive_target_warmup:
            midpoint = 0.5 * (epoch_pos + epoch_neg) + adaptive_target_margin
            new_target = _clamp(
                (1.0 - adaptive_target_alpha) * goodness_target
                + adaptive_target_alpha * midpoint,
                adaptive_target_min,
                adaptive_target_max,
            )
            if abs(new_target - goodness_target) > 1e-8:
                goodness_target = new_target
                target_updated = True

        adapt_event = ""
        if (
            adaptive_hall_enabled
            and hall_used > 0
            and neg_mode in ("hallucinate", "schedule", "mix")
        ):
            hall_overused = hall_ratio >= adaptive_hall_ratio_high
            needs_harder = (
                hall_close_ratio >= adaptive_hall_close_high
                or hall_hardness <= adaptive_hall_hardness_low
            )
            too_hard = (
                hall_close_ratio < adaptive_hall_close_high
                and hall_hardness >= adaptive_hall_hardness_high
            )

            if hall_overused and hall_hardness <= adaptive_hall_hardness_low:
                hall_node_fraction = _clamp(
                    hall_node_fraction - adaptive_hall_node_inc,
                    0.0,
                    1.0,
                )
                neg_gate_margin = _clamp(
                    neg_gate_margin + adaptive_gate_margin_step,
                    adaptive_gate_margin_min,
                    adaptive_gate_margin_max,
                )
                if neg_mode == "mix":
                    neg_mix_end = _clamp(
                        neg_mix_end - adaptive_mix_step,
                        neg_mix_start,
                        adaptive_mix_end_max,
                    )
                _sync_hall_curriculum_end()
                adapt_event = "rebalance_mix"
            elif needs_harder:
                hall_steps = int(_clamp(float(hall_steps + adaptive_hall_steps_inc), 1.0, float(adaptive_hall_steps_max)))
                hall_lr = _clamp(hall_lr * adaptive_hall_lr_mult, 1e-4, adaptive_hall_lr_max)
                hall_mean = _clamp(hall_mean * adaptive_hall_reg_mult, adaptive_hall_reg_min, 5.0)
                hall_std = _clamp(hall_std * adaptive_hall_reg_mult, adaptive_hall_reg_min, 5.0)
                hall_corr = _clamp(hall_corr * adaptive_hall_reg_mult, adaptive_hall_reg_min, 5.0)
                hall_node_fraction = _clamp(hall_node_fraction + adaptive_hall_node_inc, 0.0, 1.0)
                if hall_min_delta > 0:
                    hall_min_delta = _clamp(
                        hall_min_delta * adaptive_hall_min_delta_mult,
                        adaptive_hall_min_delta_min,
                        hall_min_delta,
                    )
                neg_gate_margin = _clamp(
                    neg_gate_margin - adaptive_gate_margin_step,
                    adaptive_gate_margin_min,
                    adaptive_gate_margin_max,
                )
                if neg_mode == "mix":
                    neg_mix_end = _clamp(
                        neg_mix_end + adaptive_mix_step,
                        neg_mix_start,
                        adaptive_mix_end_max,
                    )
                _sync_hall_curriculum_end()
                adapt_event = "harder_neg"
            elif too_hard:
                hall_lr = _clamp(hall_lr / max(adaptive_hall_lr_mult, 1e-6), 1e-4, adaptive_hall_lr_max)
                hall_mean = _clamp(
                    hall_mean / max(adaptive_hall_reg_mult, 1e-6),
                    adaptive_hall_reg_min,
                    5.0,
                )
                hall_std = _clamp(
                    hall_std / max(adaptive_hall_reg_mult, 1e-6),
                    adaptive_hall_reg_min,
                    5.0,
                )
                hall_corr = _clamp(
                    hall_corr / max(adaptive_hall_reg_mult, 1e-6),
                    adaptive_hall_reg_min,
                    5.0,
                )
                hall_node_fraction = _clamp(hall_node_fraction - adaptive_hall_node_inc, 0.0, 1.0)
                neg_gate_margin = _clamp(
                    neg_gate_margin + adaptive_gate_margin_step,
                    adaptive_gate_margin_min,
                    adaptive_gate_margin_max,
                )
                if neg_mode == "mix":
                    neg_mix_end = _clamp(
                        neg_mix_end - adaptive_mix_step,
                        neg_mix_start,
                        adaptive_mix_end_max,
                    )
                _sync_hall_curriculum_end()
                adapt_event = "easier_neg"

        if adapt_event or target_updated:
            print(
                f"epoch {epoch}: adapt={adapt_event or 'target_only'} "
                f"target={goodness_target:.3f} mix_end={neg_mix_end:.3f} "
                f"gate_margin={neg_gate_margin:.3f} hall_steps={hall_steps} "
                f"hall_lr={hall_lr:.4f} hall_node_fraction={hall_node_fraction:.2f}"
            )

        if log_csv:
            with Path(log_csv).open("a") as f:
                f.write(
                    f"{epoch},{epoch_loss:.6f},"
                    f"{epoch_pos:.6f},{epoch_neg:.6f},"
                    f"{hall_ratio:.4f},{gate_ratio:.4f},{hall_hardness:.6f},"
                    f"{hall_close_ratio:.4f},{energy_penalty_epoch:.6f},{risk_loss_epoch:.6f},"
                    f"{portfolio_loss_epoch:.6f},"
                    f"{dist_forward_epoch:.6f},"
                    f"{epoch_goodness_target:.6f},{epoch_neg_mix_end:.6f},{epoch_neg_gate_margin:.6f},"
                    f"{epoch_hall_lr:.6f},{epoch_hall_steps},{epoch_hall_node_fraction:.6f},"
                    f"{rank_aux_epoch:.6f},"
                    f"{time_neg_gen_epoch:.6f},{time_hall_epoch:.6f},{time_fwd_pos_epoch:.6f},{time_fwd_neg_epoch:.6f},"
                    f"{time_loss_epoch:.6f},{time_opt_epoch:.6f},{time_econ_epoch:.6f}\n"
                )

    if save_encoder:
        save_path = Path(str(save_encoder))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    elif save_model:
        # Backward-compat: if only save_model is provided, treat it as encoder checkpoint output.
        save_path = Path(str(save_model))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    if save_critic:
        save_path = Path(str(save_critic))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(critic.state_dict(), save_path)

    if log_csv and plot_path:
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.read_csv(log_csv)
            plt.figure(figsize=(8, 5))
            plt.plot(df["epoch"], df["loss"], label="loss")
            plt.plot(df["epoch"], df["g_pos"], label="g_pos")
            plt.plot(df["epoch"], df["g_neg"], label="g_neg")
            if "hallucinate_ratio" in df.columns:
                plt.plot(df["epoch"], df["hallucinate_ratio"], label="hall_ratio")
            if "gate_ratio" in df.columns:
                plt.plot(df["epoch"], df["gate_ratio"], label="gate_ratio")
            if "hall_hardness" in df.columns:
                plt.plot(df["epoch"], df["hall_hardness"], label="hall_hardness")
            if "energy_penalty" in df.columns:
                plt.plot(df["epoch"], df["energy_penalty"], label="energy_penalty")
            if "risk_loss" in df.columns:
                plt.plot(df["epoch"], df["risk_loss"], label="risk_loss")
            if "portfolio_loss" in df.columns:
                plt.plot(df["epoch"], df["portfolio_loss"], label="portfolio_loss")
            if "dist_forward_loss" in df.columns:
                plt.plot(df["epoch"], df["dist_forward_loss"], label="dist_forward")
            if "goodness_target_used" in df.columns:
                plt.plot(df["epoch"], df["goodness_target_used"], label="goodness_target")
            if "neg_mix_end_used" in df.columns:
                plt.plot(df["epoch"], df["neg_mix_end_used"], label="neg_mix_end")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plot_path = Path(plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
        except Exception as exc:
            print(f"Plotting failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
