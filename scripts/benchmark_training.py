#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
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
    infer_graph_goodness_with_uncertainty,
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
    "edge_attack",
    "sector_swap",
    "factor_hard",
}
_RISK_TARGET_MEM_CACHE: dict[str, tuple[list[float | None], float, float]] = {}
_PORT_TARGET_MEM_CACHE: dict[str, tuple[list[float | None], float, float]] = {}


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
        result = (targets, 0.0, 1.0)
        _RISK_TARGET_MEM_CACHE[cache_key] = result
        return result
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
        return torch.nn.functional.mse_loss(pred[mask], target[mask])

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
        return torch.nn.functional.mse_loss(pred[mask], target[mask])
    return torch.nn.functional.smooth_l1_loss(pred[mask], target[mask])


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
            out[f"{key}_min"] = float(np.min(vals))
            out[f"{key}_max"] = float(np.max(vals))
    first = fold_rows[0]
    for key in (
        "eval_objective",
        "objective_track",
        "primary_eval_metric_name",
        "primary_eval_metric_robust_name",
        "neg_mode_effective",
        "eval_neg_mode_effective",
        "risk_head_enabled_effective",
        "portfolio_head_enabled_effective",
    ):
        if key in first:
            out[key] = first[key]
    out["walk_forward_num_folds"] = len(fold_rows)
    out["split_mode_effective"] = "walk_forward"
    return out


def _compute_econ_metrics_for_eval(
    model,
    critic,
    eval_graphs,
    eval_dates,
    config: dict,
):
    t0 = time.perf_counter()
    meta = {
        "econ_ticker_requested": str(config.get("econ_ticker", "")),
        "econ_ticker_effective": str(config.get("econ_ticker_effective", "")),
        "econ_ticker_source": str(config.get("econ_ticker_source", "")),
        "econ_ticker_rows": float(config.get("econ_ticker_rows", 0) or 0),
        "econ_eval_s": 0.0,
    }
    if not bool(config.get("econ_enabled", False)):
        return meta
    if not eval_graphs or not eval_dates:
        return meta
    fwd_ret_1 = config.get("econ_fwd_ret_1")
    if fwd_ret_1 is None:
        return meta
    g, g_unc = infer_graph_goodness_with_uncertainty(
        model,
        eval_graphs,
        goodness_temp=float(config.get("goodness_temp", 1.0)),
        batch_size=int(config.get("econ_loader_batch_size", config.get("batch_size", 64))),
        critic=critic,
    )
    if g.size == 0:
        return meta
    risk_signal = None
    if bool(config.get("econ_regime_gate_enabled", False)):
        rgw = max(10, int(config.get("econ_regime_gate_window", 63)))
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
        signal_window=int(config.get("econ_signal_window", 126)),
        signal_quantile=float(config.get("econ_signal_quantile", 0.5)),
        turnover_cost_bps=float(config.get("econ_turnover_cost_bps", 0.0)),
        slippage_bps=float(config.get("econ_slippage_bps", 0.0)),
        slippage_vol_scale=float(config.get("econ_slippage_vol_scale", 0.0)),
        slippage_vol_lookback=int(config.get("econ_slippage_vol_lookback", 21)),
        trading_days=int(config.get("econ_trading_days", 252)),
        regime_gate_enabled=bool(config.get("econ_regime_gate_enabled", False)),
        regime_gate_window=int(config.get("econ_regime_gate_window", 63)),
        regime_confidence_temp=float(config.get("econ_regime_confidence_temp", 1.0)),
        regime_neutral_exposure=float(config.get("econ_regime_neutral_exposure", 0.0)),
        regime_min_confidence=float(config.get("econ_regime_min_confidence", 0.0)),
        goodness_uncertainty=g_unc,
        regime_uncertainty_scale=float(config.get("econ_regime_uncertainty_scale", 0.0)),
        risk_signal=risk_signal,
        regime_risk_scale=float(config.get("econ_regime_risk_scale", 0.0)),
        regime_thresholding_enabled=bool(config.get("econ_regime_thresholding_enabled", True)),
        regime_threshold_window=int(
            config.get("econ_regime_threshold_window", config.get("econ_signal_window", 126))
        ),
        regime_threshold_quantile=float(
            config.get("econ_regime_threshold_quantile", config.get("econ_signal_quantile", 0.5))
        ),
        regime_vol_window=int(config.get("econ_regime_vol_window", 21)),
        regime_low_quantile=float(config.get("econ_regime_low_quantile", 0.33)),
        regime_high_quantile=float(config.get("econ_regime_high_quantile", 0.67)),
    )
    out.update(meta)
    out["econ_eval_s"] = float(time.perf_counter() - t0)
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


def _infer_neg_feature_slices(config: dict) -> tuple[int | None, int | None, int]:
    window_len = int(config.get("window_len", 0) or 0)
    summary_dim = int(config.get("summary_dim", 0) or 0)
    if window_len <= 0:
        return None, None, 0
    if summary_dim >= 10:
        # window_plus_summary_fund: [returns | summary(5) | fund(5)].
        sector_idx = window_len + 5
        factor_start = sector_idx + 1
        factor_dim = 4
        return sector_idx, factor_start, factor_dim
    if summary_dim >= 2:
        # Fallback: use summary tail as factor proxies.
        factor_start = window_len
        factor_dim = min(4, summary_dim)
        return None, factor_start, factor_dim
    return None, None, 0


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
    window_len: int | None = None,
    summary_dim: int = 0,
    sector_idx: int | None = None,
    factor_start_idx: int | None = None,
    factor_dim: int = 0,
    timing: dict[str, float] | None = None,
):
    t0 = time.perf_counter()
    mode_raw = str(use_mode).strip().lower()
    def _finish(out_tensor: torch.Tensor) -> torch.Tensor:
        if timing is not None:
            dt = time.perf_counter() - t0
            timing["neg_gen"] = timing.get("neg_gen", 0.0) + dt
            if mode_raw == "hallucinate":
                timing["hallucinate"] = timing.get("hallucinate", 0.0) + dt
        return out_tensor
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
            return _finish(out)
        hub_frac = float(getattr(hall_cfg, "adversarial_hub_fraction", 0.2))
        noise_mult = float(getattr(hall_cfg, "adversarial_feature_noise_mult", 3.0))
        flip_prob = float(getattr(hall_cfg, "adversarial_timeflip_prob", 0.5))
        hub_frac = max(0.0, min(1.0, hub_frac))
        noise_mult = max(1.0, noise_mult)
        flip_prob = max(0.0, min(1.0, flip_prob))
        if hub_frac <= 0:
            return _finish(out)
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
        return _finish(out_adv)
    if use_mode == "hallucinate":
        out_h = hallucinate_negative(
            model,
            x,
            edge_index,
            edge_attr,
            batch,
            hall_cfg,
            edge_weight=edge_weight,
            critic=critic,
        )
        return _finish(out_h)
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
    out = make_negative(
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
    return _finish(out)


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
            g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)

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


def _baseline_context(config: dict, device: torch.device, num_graphs: int) -> dict[str, float | str]:
    return {
        "baseline_seed": int(config.get("seed", 0)),
        "baseline_split_mode": str(config.get("split_mode", "")),
        "baseline_device": str(device),
        "baseline_graphs_total": int(num_graphs),
        "baseline_batch_size": int(config.get("batch_size", 0)),
        "baseline_eval_frac": float(config.get("eval_frac", 0.0)),
    }


def _calibrate_goodness_target(
    model,
    critic,
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
            g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp, critic=critic)

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
                    neg_mode,
                    noise_std,
                    hall_cfg,
                    critic=critic,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )

        with torch.no_grad():
            h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
            g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp, critic=critic)

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
    train_shuffle = True
    if bool(config.get("sequence_critic_enabled", False)) and bool(
        config.get("sequence_critic_force_chrono", True)
    ):
        train_shuffle = False
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": train_shuffle,
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
        conv_type=str(config.get("encoder_conv_type", "gcn")).strip().lower(),
        gat_heads=int(config.get("encoder_gat_heads", 2)),
        residual_edge_enabled=bool(config.get("residual_edge_weight_enabled", False)),
        residual_edge_hidden_dim=int(config.get("residual_edge_hidden_dim", 32)),
        residual_edge_max_delta=float(config.get("residual_edge_max_delta", 0.25)),
        residual_edge_detach_features=bool(config.get("residual_edge_detach_features", True)),
    ).to(device)
    if bool(config.get("torch_compile", False)) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=str(config.get("torch_compile_mode", "reduce-overhead")))
        except Exception as exc:
            print(f"warning: benchmark ff torch.compile disabled: {exc}")
    critic = _build_critic(config, hidden_dim=int(config["hidden_dim"]), device=device)
    risk_targets_by_horizon = config.get("risk_targets_by_horizon")
    risk_horizons_effective = config.get("risk_horizons_effective", [])
    risk_head_active = bool(config.get("risk_head_enabled", False)) and bool(risk_targets_by_horizon)
    if layerwise and risk_head_active:
        print("risk_head disabled for ff_layerwise benchmarking.")
        risk_head_active = False
    risk_head = None
    if risk_head_active:
        risk_head = torch.nn.Linear(
            config["hidden_dim"],
            len(risk_horizons_effective),
        ).to(device)
    portfolio_targets = config.get("portfolio_targets")
    portfolio_head_active = bool(config.get("portfolio_head_enabled", False)) and bool(portfolio_targets)
    if layerwise and portfolio_head_active:
        print("portfolio_head disabled for ff_layerwise benchmarking.")
        portfolio_head_active = False
    portfolio_head = None
    if portfolio_head_active:
        portfolio_head = torch.nn.Linear(config["hidden_dim"], 1).to(device)
    optim_params = list(model.parameters())
    optim_params.extend(list(critic.parameters()))
    if risk_head is not None:
        optim_params.extend(list(risk_head.parameters()))
    if portfolio_head is not None:
        optim_params.extend(list(portfolio_head.parameters()))
    optim = _build_optimizer(
        optim_params,
        lr=config["lr"],
        device=device,
        use_fused=bool(config.get("fused_optimizer", True)),
    )
    clip_params = list(optim_params)

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
        adaptive_lr=bool(config.get("hall_adaptive_lr", False)),
        adaptive_lr_patience=int(config.get("hall_adaptive_lr_patience", 2)),
        adaptive_lr_decay=float(config.get("hall_adaptive_lr_decay", 0.5)),
        adaptive_lr_min=float(config.get("hall_adaptive_lr_min", 1e-4)),
        early_stop_on_target_hit=bool(config.get("hall_early_stop_on_target_hit", False)),
        target_hit_patience=int(config.get("hall_target_hit_patience", 1)),
        moment_mean_weight=float(config.get("hall_moment_mean", 0.0)),
        moment_var_weight=float(config.get("hall_moment_var", 0.0)),
        moment_skew_weight=float(config.get("hall_moment_skew", 0.0)),
        moment_scope=str(config.get("hall_moment_scope", "returns")),
        adversarial_hub_fraction=float(config.get("hall_attack_hub_fraction", 0.2)),
        adversarial_feature_noise_mult=float(config.get("hall_attack_noise_mult", 3.0)),
        adversarial_timeflip_prob=float(config.get("hall_attack_timeflip_prob", 0.5)),
        adversarial_edge_drop_prob=float(config.get("hall_attack_edge_drop_prob", 0.2)),
        adversarial_sign_flip_prob=float(config.get("hall_attack_sign_flip_prob", 0.2)),
        adversarial_hub_weight_scale=float(config.get("hall_attack_hub_weight_scale", 0.5)),
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
    neg_sector_idx, neg_factor_start_idx, neg_factor_dim = _infer_neg_feature_slices(config)
    hall_every_n_batches = max(1, int(config.get("ff_hall_every_n_batches", 1)))
    hall_warmup_epochs = max(0, int(config.get("ff_hall_warmup_epochs", 0)))

    epoch_times = []
    component_times = {
        "neg_gen": 0.0,
        "hallucinate": 0.0,
        "forward_pos": 0.0,
        "forward_neg": 0.0,
        "loss_terms": 0.0,
        "optimizer": 0.0,
    }
    steps_total = 0
    risk_loss_sum = 0.0
    risk_batches = 0
    portfolio_loss_sum = 0.0
    portfolio_batches = 0
    for epoch in tqdm(
        range(1, config["epochs"] + 1),
        desc="Benchmark",
        unit="epoch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        model.train()
        critic.train()
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
            if use_mode == "hallucinate":
                if epoch <= hall_warmup_epochs or (batch_idx % hall_every_n_batches) != 0:
                    use_mode = "shuffle+noise"
            apply_distance = dist_weight > 0 and (batch_idx % dist_interval == 0)
            step_scaler = scaler if (amp_enabled and (use_mode == "self_contrastive" or layerwise)) else None
            batch_t0 = time.perf_counter()
            comp_before = component_times.copy()

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
                        critic=critic,
                        window_len=config.get("window_len"),
                        summary_dim=config.get("summary_dim", 0),
                        sector_idx=neg_sector_idx,
                        factor_start_idx=neg_factor_start_idx,
                        factor_dim=neg_factor_dim,
                        timing=component_times,
                    )
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        t_fwd_pos = time.perf_counter()
                        layers_pos = model(x, batch.edge_index, edge_weight=edge_weight, return_all=True)
                        component_times["forward_pos"] += time.perf_counter() - t_fwd_pos
                        t_fwd_neg = time.perf_counter()
                        layers_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True)
                        component_times["forward_neg"] += time.perf_counter() - t_fwd_neg
                    if layer_mode == "hallucinate":
                        last_idx = ff_block_endpoints[-1]
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            g_pos_probe = goodness(
                                layers_pos[last_idx],
                                batch.batch,
                                temperature=config["goodness_temp"],
                                critic=critic,
                            ).mean().item()
                            g_neg_probe = goodness(
                                layers_neg[last_idx],
                                batch.batch,
                                temperature=config["goodness_temp"],
                                critic=critic,
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
                                t_fwd_neg = time.perf_counter()
                                layers_neg = model(
                                    x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True
                                )
                                component_times["forward_neg"] += time.perf_counter() - t_fwd_neg
                    loss = 0.0
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        for li in ff_block_endpoints:
                            g_pos = goodness(
                                layers_pos[li],
                                batch.batch,
                                temperature=config["goodness_temp"],
                                critic=critic,
                            )
                            g_neg = goodness(
                                layers_neg[li],
                                batch.batch,
                                temperature=config["goodness_temp"],
                                critic=critic,
                            )
                            loss = loss + ff_loss(
                                g_pos,
                                g_neg,
                                target=config["goodness_target"],
                                margin=float(config.get("ff_margin", 0.0)),
                                margin_weight=float(config.get("ff_margin_weight", 1.0)),
                            )
                    loss = loss / max(1, len(ff_block_endpoints))
                    t_opt = time.perf_counter()
                    _optimizer_step(
                        optim=optim,
                        loss=loss,
                        grad_clip=float(config["grad_clip"]),
                        clip_params=clip_params,
                        scaler=step_scaler,
                    )
                    component_times["optimizer"] += time.perf_counter() - t_opt
                    total_loss += loss.item()
                else:
                    x_in = x
                    for li in range(len(model.layers)):
                        layer_mode = use_mode
                        if layer_mode == "self_contrastive":
                            layer_mode = "shuffle"
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            t_fwd_pos = time.perf_counter()
                            h_pos = model.forward_layer(x_in, batch.edge_index, edge_weight, li)
                            component_times["forward_pos"] += time.perf_counter() - t_fwd_pos
                            g_pos = goodness(
                                h_pos,
                                batch.batch,
                                temperature=config["goodness_temp"],
                                critic=critic,
                            )
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
                            critic=critic,
                            window_len=config.get("window_len"),
                            summary_dim=config.get("summary_dim", 0),
                            sector_idx=neg_sector_idx,
                            factor_start_idx=neg_factor_start_idx,
                            factor_dim=neg_factor_dim,
                            timing=component_times,
                        )
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            t_fwd_neg = time.perf_counter()
                            h_neg = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                            component_times["forward_neg"] += time.perf_counter() - t_fwd_neg
                            g_neg = goodness(
                                h_neg,
                                batch.batch,
                                temperature=config["goodness_temp"],
                                critic=critic,
                            )
                            loss = ff_loss(
                                g_pos,
                                g_neg,
                                target=config["goodness_target"],
                                margin=float(config.get("ff_margin", 0.0)),
                                margin_weight=float(config.get("ff_margin_weight", 1.0)),
                            )
                        t_opt = time.perf_counter()
                        _optimizer_step(
                            optim=optim,
                            loss=loss,
                            grad_clip=float(config["grad_clip"]),
                            clip_params=clip_params,
                            scaler=step_scaler,
                        )
                        component_times["optimizer"] += time.perf_counter() - t_opt
                        total_loss += loss.item()
                        x_in = h_pos.detach()
            else:
                if use_mode == "self_contrastive":
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        t_fwd_pos = time.perf_counter()
                        h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                        component_times["forward_pos"] += time.perf_counter() - t_fwd_pos
                        t_neg_gen = time.perf_counter()
                        x_view = _make_self_contrastive_view(
                            x,
                            batch.batch,
                            view_mode=config["self_contrastive_view_mode"],
                            view_noise_std=config["self_contrastive_view_noise_std"],
                            window_len=config.get("window_len"),
                            summary_dim=config.get("summary_dim", 0),
                        )
                        component_times["neg_gen"] += time.perf_counter() - t_neg_gen
                        t_fwd_neg = time.perf_counter()
                        h_view = model(x_view, batch.edge_index, edge_weight=edge_weight)
                        component_times["forward_neg"] += time.perf_counter() - t_fwd_neg
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
                            t_neg_gen = time.perf_counter()
                            x_neg_aux = make_negative(
                                x,
                                batch.batch,
                                mode=sc_ff_neg_mode,
                                noise_std=sc_ff_noise_std,
                                window_len=config.get("window_len"),
                                summary_dim=config.get("summary_dim", 0),
                            )
                            component_times["neg_gen"] += time.perf_counter() - t_neg_gen
                            t_fwd_neg_aux = time.perf_counter()
                            h_neg_aux = model(x_neg_aux, batch.edge_index, edge_weight=edge_weight)
                            component_times["forward_neg"] += time.perf_counter() - t_fwd_neg_aux
                            g_pos_aux = goodness(
                                h_pos,
                                batch.batch,
                                temperature=config["goodness_temp"],
                                critic=critic,
                            )
                            g_neg_aux = goodness(
                                h_neg_aux,
                                batch.batch,
                                temperature=config["goodness_temp"],
                                critic=critic,
                            )
                            loss = loss + sc_ff_weight * ff_loss(
                                g_pos_aux,
                                g_neg_aux,
                                target=sc_ff_target,
                                margin=float(config.get("ff_margin", 0.0)),
                                margin_weight=float(config.get("ff_margin_weight", 1.0)),
                            )
                else:
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
                        critic=critic,
                        window_len=config.get("window_len"),
                        summary_dim=config.get("summary_dim", 0),
                        sector_idx=neg_sector_idx,
                        factor_start_idx=neg_factor_start_idx,
                        factor_dim=neg_factor_dim,
                        timing=component_times,
                    )
                    if bool(config.get("ff_concat_posneg", True)) and use_mode != "hallucinate":
                        n_nodes = int(x.size(0))
                        num_graphs = int(batch.batch.max().item()) + 1 if batch.batch.numel() else 0
                        x_cat = torch.cat([x, x_neg], dim=0)
                        edge_index_cat = torch.cat(
                            [batch.edge_index, batch.edge_index + n_nodes], dim=1
                        )
                        edge_weight_cat = (
                            torch.cat([edge_weight, edge_weight], dim=0)
                            if edge_weight is not None
                            else None
                        )
                        batch_cat = torch.cat([batch.batch, batch.batch + num_graphs], dim=0)
                        t_fwd = time.perf_counter()
                        h_cat = model(x_cat, edge_index_cat, edge_weight=edge_weight_cat)
                        dt_fwd = time.perf_counter() - t_fwd
                        component_times["forward_pos"] += 0.5 * dt_fwd
                        component_times["forward_neg"] += 0.5 * dt_fwd
                        h_pos = h_cat[:n_nodes]
                        h_neg = h_cat[n_nodes:]
                    else:
                        t_fwd_pos = time.perf_counter()
                        h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                        component_times["forward_pos"] += time.perf_counter() - t_fwd_pos
                        t_fwd_neg = time.perf_counter()
                        h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                        component_times["forward_neg"] += time.perf_counter() - t_fwd_neg
                    g_pos = goodness(
                        h_pos,
                        batch.batch,
                        temperature=config["goodness_temp"],
                        critic=critic,
                    )
                    g_neg = goodness(
                        h_neg,
                        batch.batch,
                        temperature=config["goodness_temp"],
                        critic=critic,
                    )
                    loss = ff_loss(
                        g_pos,
                        g_neg,
                        target=config["goodness_target"],
                        margin=float(config.get("ff_margin", 0.0)),
                        margin_weight=float(config.get("ff_margin_weight", 1.0)),
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
                risk_loss_val = None
                if risk_head is not None and risk_targets_by_horizon is not None:
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        z_risk = global_mean_pool(h_pos, batch.batch)
                        risk_loss_val = _compute_multi_horizon_risk_loss(
                            risk_head=risk_head,
                            embeddings=z_risk,
                            graph_idx=batch.graph_idx,
                            risk_targets_by_horizon=risk_targets_by_horizon,
                            device=device,
                            risk_loss_type=str(config.get("risk_loss_type", "huber")).strip().lower(),
                        )
                        if risk_loss_val is not None:
                            loss = loss + float(config.get("risk_loss_weight", 0.0)) * risk_loss_val
                portfolio_loss_val = None
                if portfolio_head is not None and portfolio_targets is not None:
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        z_port = global_mean_pool(h_pos, batch.batch)
                        portfolio_loss_val = _compute_portfolio_head_loss(
                            portfolio_head=portfolio_head,
                            embeddings=z_port,
                            graph_idx=batch.graph_idx,
                            portfolio_targets=portfolio_targets,
                            device=device,
                            loss_type=str(config.get("portfolio_loss_type", "sharpe")).strip().lower(),
                        )
                        if portfolio_loss_val is not None:
                            loss = loss + float(config.get("portfolio_loss_weight", 0.0)) * portfolio_loss_val
                t_opt = time.perf_counter()
                _optimizer_step(
                    optim=optim,
                    loss=loss,
                    grad_clip=float(config["grad_clip"]),
                    clip_params=clip_params,
                    scaler=step_scaler,
                )
                component_times["optimizer"] += time.perf_counter() - t_opt
                total_loss += loss.item()
                if risk_loss_val is not None:
                    risk_loss_sum += float(risk_loss_val.detach())
                    risk_batches += 1
                if portfolio_loss_val is not None:
                    portfolio_loss_sum += float(portfolio_loss_val.detach())
                    portfolio_batches += 1

            batch_elapsed = time.perf_counter() - batch_t0
            known_elapsed = sum(
                max(0.0, component_times[k] - comp_before.get(k, 0.0))
                for k in ("neg_gen", "hallucinate", "forward_pos", "forward_neg", "loss_terms", "optimizer")
            )
            if batch_elapsed > known_elapsed:
                component_times["loss_terms"] += (batch_elapsed - known_elapsed)
            steps_total += 1
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
            critic,
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
        critic,
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
    risk_loss_train = risk_loss_sum / risk_batches if risk_batches else 0.0
    portfolio_loss_train = portfolio_loss_sum / portfolio_batches if portfolio_batches else 0.0
    steps_denom = max(1, int(steps_total))
    out = {
        "avg_epoch_s": avg_time,
        "graphs_per_s": avg_gps,
        "time_neg_gen_s": float(component_times["neg_gen"] / steps_denom),
        "time_hallucinate_s": float(component_times["hallucinate"] / steps_denom),
        "time_forward_pos_s": float(component_times["forward_pos"] / steps_denom),
        "time_forward_neg_s": float(component_times["forward_neg"] / steps_denom),
        "time_loss_terms_s": float(component_times["loss_terms"] / steps_denom),
        "time_optimizer_s": float(component_times["optimizer"] / steps_denom),
        "time_econ_eval_s": 0.0,
        "goodness_target_eval": target_eval,
        "target_cal_acc": target_cal_acc,
        "neg_mode_effective": train_neg_mode,
        "eval_neg_mode_effective": eval_mode,
        "risk_head_enabled_effective": bool(risk_head is not None),
        "risk_loss_train": risk_loss_train,
        "portfolio_head_enabled_effective": bool(portfolio_head is not None),
        "portfolio_loss_train": portfolio_loss_train,
    }
    if risk_head is not None:
        out["risk_horizons_effective"] = ",".join(str(h) for h in risk_horizons_effective)
        out["risk_ticker_effective"] = str(config.get("risk_ticker_effective", ""))
    if portfolio_head is not None:
        out["portfolio_ticker_effective"] = str(config.get("portfolio_ticker_effective", ""))
    out.update(eval_metrics)
    econ = _compute_econ_metrics_for_eval(
        model,
        critic,
        eval_graphs,
        eval_dates or [],
        config,
    )
    if econ:
        out.update(econ)
        out["time_econ_eval_s"] = float(econ.get("econ_eval_s", 0.0))

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
                critic,
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
        conv_type=str(config.get("encoder_conv_type", "gcn")).strip().lower(),
        gat_heads=int(config.get("encoder_gat_heads", 2)),
        residual_edge_enabled=bool(config.get("residual_edge_weight_enabled", False)),
        residual_edge_hidden_dim=int(config.get("residual_edge_hidden_dim", 32)),
        residual_edge_max_delta=float(config.get("residual_edge_max_delta", 0.25)),
        residual_edge_detach_features=bool(config.get("residual_edge_detach_features", True)),
    ).to(device)
    if bool(config.get("torch_compile", False)) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=str(config.get("torch_compile_mode", "reduce-overhead")))
        except Exception as exc:
            print(f"warning: benchmark backprop torch.compile disabled: {exc}")
    head = torch.nn.Linear(config["hidden_dim"], 1).to(device)
    risk_targets_by_horizon = config.get("risk_targets_by_horizon")
    risk_horizons_effective = config.get("risk_horizons_effective", [])
    risk_head = None
    if bool(config.get("risk_head_enabled", False)) and risk_targets_by_horizon:
        risk_head = torch.nn.Linear(config["hidden_dim"], len(risk_horizons_effective)).to(device)
    portfolio_targets = config.get("portfolio_targets")
    portfolio_head = None
    if bool(config.get("portfolio_head_enabled", False)) and portfolio_targets:
        portfolio_head = torch.nn.Linear(config["hidden_dim"], 1).to(device)
    optim_params = list(model.parameters()) + list(head.parameters())
    if risk_head is not None:
        optim_params.extend(list(risk_head.parameters()))
    if portfolio_head is not None:
        optim_params.extend(list(portfolio_head.parameters()))
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
        adaptive_lr=bool(config.get("hall_adaptive_lr", False)),
        adaptive_lr_patience=int(config.get("hall_adaptive_lr_patience", 2)),
        adaptive_lr_decay=float(config.get("hall_adaptive_lr_decay", 0.5)),
        adaptive_lr_min=float(config.get("hall_adaptive_lr_min", 1e-4)),
        early_stop_on_target_hit=bool(config.get("hall_early_stop_on_target_hit", False)),
        target_hit_patience=int(config.get("hall_target_hit_patience", 1)),
        moment_mean_weight=float(config.get("hall_moment_mean", 0.0)),
        moment_var_weight=float(config.get("hall_moment_var", 0.0)),
        moment_skew_weight=float(config.get("hall_moment_skew", 0.0)),
        moment_scope=str(config.get("hall_moment_scope", "returns")),
        adversarial_hub_fraction=float(config.get("hall_attack_hub_fraction", 0.2)),
        adversarial_feature_noise_mult=float(config.get("hall_attack_noise_mult", 3.0)),
        adversarial_timeflip_prob=float(config.get("hall_attack_timeflip_prob", 0.5)),
        adversarial_edge_drop_prob=float(config.get("hall_attack_edge_drop_prob", 0.2)),
        adversarial_sign_flip_prob=float(config.get("hall_attack_sign_flip_prob", 0.2)),
        adversarial_hub_weight_scale=float(config.get("hall_attack_hub_weight_scale", 0.5)),
    )
    train_neg_mode = str(config["neg_mode"]).strip().lower()
    if train_neg_mode == "self_contrastive":
        print("backprop mode does not use self_contrastive negatives; using shuffle negatives.")
        train_neg_mode = "shuffle"
    eval_mode = _resolve_mode(config.get("eval_neg_mode", "auto"), train_neg_mode)
    if eval_mode == "self_contrastive":
        eval_mode = "shuffle"
    neg_sector_idx, neg_factor_start_idx, neg_factor_dim = _infer_neg_feature_slices(config)
    hall_every_n_batches = max(1, int(config.get("ff_hall_every_n_batches", 1)))
    hall_warmup_epochs = max(0, int(config.get("ff_hall_warmup_epochs", 0)))

    epoch_times = []
    component_times = {
        "neg_gen": 0.0,
        "hallucinate": 0.0,
        "forward_pos": 0.0,
        "forward_neg": 0.0,
        "loss_terms": 0.0,
        "optimizer": 0.0,
    }
    steps_total = 0
    risk_loss_sum = 0.0
    risk_batches = 0
    portfolio_loss_sum = 0.0
    portfolio_batches = 0
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
            batch_t0 = time.perf_counter()
            comp_before = component_times.copy()
            with _autocast_if_needed(bp_amp_enabled, bp_amp_dtype):
                t_fwd_pos = time.perf_counter()
                h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                component_times["forward_pos"] += time.perf_counter() - t_fwd_pos
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
            if use_mode == "hallucinate":
                batch_step = steps_total + 1
                if epoch <= hall_warmup_epochs or (batch_step % hall_every_n_batches) != 0:
                    use_mode = "shuffle+noise"
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
                sector_idx=neg_sector_idx,
                factor_start_idx=neg_factor_start_idx,
                factor_dim=neg_factor_dim,
                timing=component_times,
            )
            with _autocast_if_needed(bp_amp_enabled, bp_amp_dtype):
                t_fwd_neg = time.perf_counter()
                h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                component_times["forward_neg"] += time.perf_counter() - t_fwd_neg
                z_neg = global_mean_pool(h_neg, batch.batch)
                y_neg = torch.zeros(z_neg.size(0), device=device)

                z = torch.cat([z_pos, z_neg], dim=0)
                y = torch.cat([y_pos, y_neg], dim=0)
                logits = head(z).squeeze(1)
                loss = bce(logits, y)
                risk_loss_val = None
                if risk_head is not None and risk_targets_by_horizon is not None:
                    risk_loss_val = _compute_multi_horizon_risk_loss(
                        risk_head=risk_head,
                        embeddings=z_pos,
                        graph_idx=batch.graph_idx,
                        risk_targets_by_horizon=risk_targets_by_horizon,
                        device=device,
                        risk_loss_type=str(config.get("risk_loss_type", "huber")).strip().lower(),
                    )
                    if risk_loss_val is not None:
                        loss = loss + float(config.get("risk_loss_weight", 0.0)) * risk_loss_val
                portfolio_loss_val = None
                if portfolio_head is not None and portfolio_targets is not None:
                    portfolio_loss_val = _compute_portfolio_head_loss(
                        portfolio_head=portfolio_head,
                        embeddings=z_pos,
                        graph_idx=batch.graph_idx,
                        portfolio_targets=portfolio_targets,
                        device=device,
                        loss_type=str(config.get("portfolio_loss_type", "sharpe")).strip().lower(),
                    )
                    if portfolio_loss_val is not None:
                        loss = loss + float(config.get("portfolio_loss_weight", 0.0)) * portfolio_loss_val

            t_opt = time.perf_counter()
            _optimizer_step(
                optim=optim,
                loss=loss,
                grad_clip=float(config["grad_clip"]),
                clip_params=optim_params,
                scaler=bp_scaler,
            )
            component_times["optimizer"] += time.perf_counter() - t_opt
            if risk_loss_val is not None:
                risk_loss_sum += float(risk_loss_val.detach())
                risk_batches += 1
            if portfolio_loss_val is not None:
                portfolio_loss_sum += float(portfolio_loss_val.detach())
                portfolio_batches += 1
            batch_elapsed = time.perf_counter() - batch_t0
            known_elapsed = sum(
                max(0.0, component_times[k] - comp_before.get(k, 0.0))
                for k in ("neg_gen", "hallucinate", "forward_pos", "forward_neg", "loss_terms", "optimizer")
            )
            if batch_elapsed > known_elapsed:
                component_times["loss_terms"] += (batch_elapsed - known_elapsed)
            steps_total += 1
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
    risk_loss_train = risk_loss_sum / risk_batches if risk_batches else 0.0
    portfolio_loss_train = portfolio_loss_sum / portfolio_batches if portfolio_batches else 0.0
    steps_denom = max(1, int(steps_total))
    out = {
        "avg_epoch_s": avg_time,
        "graphs_per_s": avg_gps,
        "time_neg_gen_s": float(component_times["neg_gen"] / steps_denom),
        "time_hallucinate_s": float(component_times["hallucinate"] / steps_denom),
        "time_forward_pos_s": float(component_times["forward_pos"] / steps_denom),
        "time_forward_neg_s": float(component_times["forward_neg"] / steps_denom),
        "time_loss_terms_s": float(component_times["loss_terms"] / steps_denom),
        "time_optimizer_s": float(component_times["optimizer"] / steps_denom),
        "time_econ_eval_s": 0.0,
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
        "risk_head_enabled_effective": bool(risk_head is not None),
        "risk_loss_train": risk_loss_train,
        "portfolio_head_enabled_effective": bool(portfolio_head is not None),
        "portfolio_loss_train": portfolio_loss_train,
    }
    if risk_head is not None:
        out["risk_horizons_effective"] = ",".join(str(h) for h in risk_horizons_effective)
        out["risk_ticker_effective"] = str(config.get("risk_ticker_effective", ""))
    if portfolio_head is not None:
        out["portfolio_ticker_effective"] = str(config.get("portfolio_ticker_effective", ""))
    econ = _compute_econ_metrics_for_eval(
        model,
        None,
        eval_graphs,
        eval_dates or [],
        config,
    )
    if econ:
        out.update(econ)
        out["time_econ_eval_s"] = float(econ.get("econ_eval_s", 0.0))
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
    for i, g in enumerate(graphs):
        setattr(g, "graph_idx", i)

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
        "neg_mode": str(bench_cfg.get("neg_mode", train_cfg.get("neg_mode", "shuffle"))),
        "eval_neg_mode": str(bench_cfg.get("eval_neg_mode", "auto")),
        "ff_neg_mix": train_cfg.get("ff_neg_mix", []),
        "ff_neg_mix_weights": train_cfg.get("ff_neg_mix_weights", []),
        "ff_curriculum_epochs": train_cfg.get("ff_curriculum_epochs", []),
        "ff_rank_aux_weight": float(train_cfg.get("ff_rank_aux_weight", 0.0)),
        "ff_hall_every_n_batches": int(train_cfg.get("ff_hall_every_n_batches", 1)),
        "ff_hall_warmup_epochs": int(train_cfg.get("ff_hall_warmup_epochs", 0)),
        "ff_hall_steps": int(train_cfg.get("ff_hall_steps", train_cfg.get("hallucinate_steps", 3))),
        "ff_concat_posneg": bool(train_cfg.get("ff_concat_posneg", True)),
        "ff_layer_cache": bool(train_cfg.get("ff_layer_cache", True)),
        "ff_econ_eval_every": int(train_cfg.get("ff_econ_eval_every", 1)),
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
        "torch_compile": bool(train_cfg.get("torch_compile", False)),
        "torch_compile_mode": str(train_cfg.get("torch_compile_mode", "reduce-overhead")),
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
        "hall_steps": int(train_cfg.get("ff_hall_steps", train_cfg.get("hallucinate_steps", 3))),
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
        "risk_head_enabled": bool(train_cfg.get("risk_head_enabled", False)),
        "risk_ticker": str(train_cfg.get("risk_ticker", "AUTO")),
        "risk_horizon": int(train_cfg.get("risk_horizon", 21)),
        "risk_horizons": _parse_positive_int_list(
            train_cfg.get("risk_horizons", train_cfg.get("risk_horizon", 21)),
            fallback=int(train_cfg.get("risk_horizon", 21)),
        ),
        "risk_loss_weight": float(train_cfg.get("risk_loss_weight", 0.1)),
        "risk_loss_type": str(train_cfg.get("risk_loss_type", "huber")),
        "risk_standardize": bool(train_cfg.get("risk_standardize", True)),
        "risk_cache_dir": str(train_cfg.get("risk_cache_dir", "runs/cache")),
        "risk_max_abs_logret": float(train_cfg.get("risk_max_abs_logret", 0.5)),
        "portfolio_head_enabled": bool(train_cfg.get("portfolio_head_enabled", False)),
        "portfolio_ticker": str(train_cfg.get("portfolio_ticker", "AUTO")),
        "portfolio_horizon": int(train_cfg.get("portfolio_horizon", 5)),
        "portfolio_loss_weight": float(train_cfg.get("portfolio_loss_weight", 0.0)),
        "portfolio_loss_type": str(train_cfg.get("portfolio_loss_type", "sharpe")),
        "portfolio_standardize": bool(train_cfg.get("portfolio_standardize", True)),
        "portfolio_cache_dir": str(train_cfg.get("portfolio_cache_dir", "runs/cache")),
        "portfolio_max_abs_logret": float(train_cfg.get("portfolio_max_abs_logret", 0.5)),
        "timing_warmup_epochs": int(bench_cfg.get("timing_warmup_epochs", 1)),
        "baseline_ref_csv": str(bench_cfg.get("baseline_ref_csv", "")),
        "require_baseline_match": bool(bench_cfg.get("require_baseline_match", False)),
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
        "econ_slippage_bps": float(bench_cfg.get("econ_slippage_bps", 0.0)),
        "econ_slippage_vol_scale": float(bench_cfg.get("econ_slippage_vol_scale", 0.0)),
        "econ_slippage_vol_lookback": int(bench_cfg.get("econ_slippage_vol_lookback", 21)),
        "econ_regime_gate_enabled": bool(bench_cfg.get("econ_regime_gate_enabled", False)),
        "econ_regime_gate_window": int(bench_cfg.get("econ_regime_gate_window", 63)),
        "econ_regime_confidence_temp": float(bench_cfg.get("econ_regime_confidence_temp", 1.0)),
        "econ_regime_neutral_exposure": float(bench_cfg.get("econ_regime_neutral_exposure", 0.0)),
        "econ_regime_min_confidence": float(bench_cfg.get("econ_regime_min_confidence", 0.0)),
        "econ_regime_uncertainty_scale": float(bench_cfg.get("econ_regime_uncertainty_scale", 0.0)),
        "econ_regime_risk_scale": float(bench_cfg.get("econ_regime_risk_scale", 0.0)),
        "econ_regime_thresholding_enabled": bool(
            bench_cfg.get("econ_regime_thresholding_enabled", True)
        ),
        "econ_regime_threshold_window": int(
            bench_cfg.get("econ_regime_threshold_window", bench_cfg.get("econ_signal_window", 126))
        ),
        "econ_regime_threshold_quantile": float(
            bench_cfg.get("econ_regime_threshold_quantile", bench_cfg.get("econ_signal_quantile", 0.5))
        ),
        "econ_regime_vol_window": int(bench_cfg.get("econ_regime_vol_window", 21)),
        "econ_regime_low_quantile": float(bench_cfg.get("econ_regime_low_quantile", 0.33)),
        "econ_regime_high_quantile": float(bench_cfg.get("econ_regime_high_quantile", 0.67)),
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

    config["risk_targets_by_horizon"] = None
    config["risk_horizons_effective"] = []
    config["risk_ticker_effective"] = str(config.get("risk_ticker", "AUTO"))
    if bool(config.get("risk_head_enabled", False)):
        if not graph_dates:
            config["risk_head_enabled"] = False
            print("warning: disabled risk_head in benchmark: graphs payload missing dates.")
        else:
            prices_path = str(build_cfg.get("prices", "data/processed/prices.csv"))
            try:
                risk_horizons = _parse_positive_int_list(
                    config.get("risk_horizons", config.get("risk_horizon", 21)),
                    fallback=int(config.get("risk_horizon", 21)),
                )
                ticker_eff, ticker_src, ticker_rows = resolve_price_ticker(
                    prices_path=prices_path,
                    requested_ticker=str(config.get("risk_ticker", "AUTO")),
                    min_rows=max(64, max(risk_horizons)),
                )
                risk_targets_by_horizon = []
                for horizon in risk_horizons:
                    targets_h, _, _ = _compute_risk_targets(
                        prices_path=prices_path,
                        ticker=str(ticker_eff),
                        dates=list(graph_dates),
                        horizon=int(horizon),
                        standardize=bool(config.get("risk_standardize", True)),
                        max_abs_logret=float(config.get("risk_max_abs_logret", 0.5)),
                        cache_dir=str(config.get("risk_cache_dir", "runs/cache")),
                    )
                    risk_targets_by_horizon.append(targets_h)
                if not risk_targets_by_horizon:
                    raise ValueError("no valid risk targets resolved")
                config["risk_targets_by_horizon"] = risk_targets_by_horizon
                config["risk_horizons_effective"] = risk_horizons
                config["risk_ticker_effective"] = str(ticker_eff)
                print(
                    "risk ticker: "
                    f"requested={config.get('risk_ticker')} "
                    f"effective={ticker_eff} source={ticker_src} rows={ticker_rows} "
                    f"horizons={risk_horizons}"
                )
            except Exception as exc:
                config["risk_head_enabled"] = False
                print(f"warning: disabled risk_head in benchmark: {exc}")

    config["portfolio_targets"] = None
    config["portfolio_ticker_effective"] = str(config.get("portfolio_ticker", "AUTO"))
    if bool(config.get("portfolio_head_enabled", False)):
        if not graph_dates:
            config["portfolio_head_enabled"] = False
            print("warning: disabled portfolio_head in benchmark: graphs payload missing dates.")
        else:
            prices_path = str(build_cfg.get("prices", "data/processed/prices.csv"))
            try:
                horizon = max(1, int(config.get("portfolio_horizon", 5)))
                ticker_eff, ticker_src, ticker_rows = resolve_price_ticker(
                    prices_path=prices_path,
                    requested_ticker=str(config.get("portfolio_ticker", "AUTO")),
                    min_rows=max(64, horizon),
                )
                portfolio_targets, _, _ = _compute_forward_return_targets(
                    prices_path=prices_path,
                    ticker=str(ticker_eff),
                    dates=list(graph_dates),
                    horizon=horizon,
                    standardize=bool(config.get("portfolio_standardize", True)),
                    max_abs_logret=float(config.get("portfolio_max_abs_logret", 0.5)),
                    cache_dir=str(config.get("portfolio_cache_dir", "runs/cache")),
                )
                config["portfolio_targets"] = portfolio_targets
                config["portfolio_ticker_effective"] = str(ticker_eff)
                print(
                    "portfolio ticker: "
                    f"requested={config.get('portfolio_ticker')} "
                    f"effective={ticker_eff} source={ticker_src} rows={ticker_rows} "
                    f"horizon={horizon}"
                )
            except Exception as exc:
                config["portfolio_head_enabled"] = False
                print(f"warning: disabled portfolio_head in benchmark: {exc}")

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
    base_ctx = _baseline_context(config, device, len(graphs))

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
                res.update(base_ctx)
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
            agg.update(base_ctx)
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
            res.update(base_ctx)
            if isinstance(mode_override, dict):
                for key, value in mode_override.items():
                    if key not in res:
                        res[key] = value
            results.append(res)

    out_path = Path(bench_cfg.get("out_csv", "runs/experiments/manual/metrics/benchmark.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    if bool(config.get("require_baseline_match", False)):
        ref_raw = str(config.get("baseline_ref_csv", "")).strip()
        ref_path = Path(ref_raw) if ref_raw else out_path
        if ref_path.exists():
            try:
                ref_df = pd.read_csv(ref_path)
                if not ref_df.empty:
                    ref = ref_df.iloc[0].to_dict()
                    compare_keys = [
                        "baseline_seed",
                        "baseline_split_mode",
                        "baseline_device",
                        "baseline_graphs_total",
                        "baseline_batch_size",
                    ]
                    mismatches = []
                    for k in compare_keys:
                        cur_v = base_ctx.get(k)
                        ref_v = ref.get(k)
                        if str(cur_v) != str(ref_v):
                            mismatches.append(f"{k}: current={cur_v} ref={ref_v}")
                    if mismatches:
                        raise RuntimeError(
                            "Baseline context mismatch; refusing comparison run.\n"
                            + "\n".join(mismatches)
                        )
            except Exception as exc:
                raise RuntimeError(f"failed baseline context check against {ref_path}: {exc}") from exc

    keys = sorted({k for r in results for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"Wrote {out_path}")
    baseline_out = str(bench_cfg.get("baseline_out_csv", "")).strip()
    baseline_path = (
        Path(baseline_out)
        if baseline_out
        else out_path.with_name(f"{out_path.stem}_baseline.csv")
    )
    baseline_rows = []
    for r in results:
        baseline_rows.append(
            {
                "mode": r.get("mode"),
                "avg_epoch_s": r.get("avg_epoch_s"),
                "graphs_per_s": r.get("graphs_per_s"),
                "primary_eval_metric_name": r.get("primary_eval_metric_name"),
                "primary_eval_metric": r.get("primary_eval_metric"),
                "econ_sharpe_uplift": r.get("econ_sharpe_uplift"),
                "baseline_seed": r.get("baseline_seed"),
                "baseline_split_mode": r.get("baseline_split_mode"),
                "baseline_device": r.get("baseline_device"),
                "baseline_graphs_total": r.get("baseline_graphs_total"),
                "baseline_batch_size": r.get("baseline_batch_size"),
            }
        )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()) if baseline_rows else ["mode"])
        w.writeheader()
        for row in baseline_rows:
            w.writerow(row)
    print(f"Wrote {baseline_path}")
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
