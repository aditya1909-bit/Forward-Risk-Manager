from __future__ import annotations

from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from frisk.ff import goodness


def max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(dd.min())


def strategy_stats(
    name: str,
    rets: np.ndarray,
    trading_days: int = 252,
) -> dict[str, float | str]:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {
            "strategy": name,
            "num_days": 0,
            "total_return": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "calmar": float("nan"),
            "max_drawdown": float("nan"),
            "max_drawdown_duration_days": float("nan"),
            "var_95_daily": float("nan"),
            "cvar_95_daily": float("nan"),
            "hit_rate_daily": float("nan"),
        }
    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    ann_return = float(equity[-1] ** (trading_days / max(1, r.size)) - 1.0)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(trading_days)) if r.size > 1 else 0.0
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-12 else float("nan")
    downside = np.minimum(r, 0.0)
    downside_vol = (
        float(np.sqrt(np.mean(downside**2)) * np.sqrt(trading_days))
        if r.size > 0
        else float("nan")
    )
    sortino = float(ann_return / downside_vol) if downside_vol > 1e-12 else float("nan")
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    underwater = drawdown < 0.0
    max_dd_duration = 0
    run = 0
    for is_under in underwater:
        if bool(is_under):
            run += 1
            if run > max_dd_duration:
                max_dd_duration = run
        else:
            run = 0
    mdd = max_drawdown(equity)
    calmar = float(ann_return / abs(mdd)) if abs(mdd) > 1e-12 else float("nan")
    var_95 = float(np.quantile(r, 0.05))
    cvar_95 = float(r[r <= var_95].mean()) if np.any(r <= var_95) else float(var_95)
    return {
        "strategy": name,
        "num_days": int(r.size),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "max_drawdown_duration_days": float(max_dd_duration),
        "var_95_daily": var_95,
        "cvar_95_daily": cvar_95,
        "hit_rate_daily": float(np.mean(r > 0)),
    }


def load_forward_returns_from_prices(
    prices_path: str | Path,
    ticker: str,
    max_abs_logret: float = 0.5,
) -> pd.Series:
    prices = pd.read_csv(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    target = str(ticker).strip().upper()
    prices = prices[prices["ticker"].astype(str).str.upper() == target].sort_values("date")
    if prices.empty:
        raise ValueError(f"Ticker {target} not found in prices: {prices_path}")

    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px = prices.set_index("date")[price_col].astype(float)
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    px = px[px > 0]
    if px.index.has_duplicates:
        px = px.groupby(level=0).median()
    logret = np.log(px).diff().dropna()
    if max_abs_logret and float(max_abs_logret) > 0:
        logret = logret[np.abs(logret) <= float(max_abs_logret)]

    fwd_ret_1 = np.exp(logret.shift(-1)) - 1.0
    fwd_ret_1 = fwd_ret_1.dropna()
    fwd_ret_1.name = "fwd_ret_1"
    return fwd_ret_1.astype(float)


def load_forward_returns_panel_from_prices(
    prices_path: str | Path,
    max_abs_logret: float = 0.5,
) -> pd.DataFrame:
    prices = pd.read_csv(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    if "ticker" not in prices.columns:
        raise ValueError(f"Ticker column not found in prices file: {prices_path}")
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    prices = prices[["date", "ticker", price_col]].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices[price_col] = pd.to_numeric(prices[price_col], errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "ticker", price_col])
    prices = prices[prices[price_col] > 0]
    if prices.empty:
        raise ValueError(f"No valid prices found in {prices_path}")

    panel = prices.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="median")
    panel = panel.sort_index()
    logret = np.log(panel).diff()
    if max_abs_logret and float(max_abs_logret) > 0:
        logret = logret.clip(lower=-float(max_abs_logret), upper=float(max_abs_logret))
    fwd = np.exp(logret.shift(-1)) - 1.0
    fwd = fwd.dropna(how="all")
    return fwd.astype(float)


def _ticker_row_counts(prices_path: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with Path(prices_path).open() as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "ticker" not in r.fieldnames:
            raise ValueError(f"prices csv missing ticker column: {prices_path}")
        for row in r:
            t = str(row.get("ticker", "")).upper().strip()
            if not t:
                continue
            counts[t] = counts.get(t, 0) + 1
    return counts


def resolve_price_ticker(
    prices_path: str | Path,
    requested_ticker: str | None,
    min_rows: int = 64,
    prefer_tickers: list[str] | None = None,
) -> tuple[str, str, int]:
    counts = _ticker_row_counts(prices_path)
    if not counts:
        raise ValueError(f"no tickers found in prices file: {prices_path}")

    req_raw = str(requested_ticker or "").upper().strip()
    req_tokens = [t.strip() for t in re.split(r"[,;|]+", req_raw) if t.strip()] if req_raw else []
    if not req_tokens:
        req_tokens = ["AUTO"]

    min_rows = max(1, int(min_rows))
    auto_tokens = {"", "AUTO", "AUTO_DETECT", "AUTO-DETECT"}
    explicit_tokens = [t for t in req_tokens if t not in auto_tokens]
    has_auto_fallback = any(t in auto_tokens for t in req_tokens)

    best_explicit = None
    for token in explicit_tokens:
        n = int(counts.get(token, 0))
        if n <= 0:
            continue
        if n >= min_rows:
            return token, "requested_priority" if len(req_tokens) > 1 else "requested", n
        if best_explicit is None or n > best_explicit[1]:
            best_explicit = (token, n)

    if explicit_tokens and not has_auto_fallback:
        if best_explicit is not None:
            token, n = best_explicit
            if len(explicit_tokens) == 1:
                return token, "requested", n
            return token, "requested_priority_no_min_match", n
        if len(explicit_tokens) == 1:
            raise ValueError(f"requested ticker {explicit_tokens[0]} not found in {prices_path}")
        raise ValueError(
            f"none of requested tickers {explicit_tokens} found in {prices_path}"
        )

    if prefer_tickers:
        for t in prefer_tickers:
            tt = str(t).upper().strip()
            n = int(counts.get(tt, 0))
            if n >= min_rows:
                return tt, "preferred_auto", n

    eligible = [(t, n) for t, n in counts.items() if int(n) >= min_rows]
    if not eligible:
        t, n = max(counts.items(), key=lambda kv: kv[1])
        return str(t), "auto_max_rows_no_min_match", int(n)

    t, n = max(eligible, key=lambda kv: kv[1])
    return str(t), "auto_max_rows", int(n)


def infer_graph_goodness(
    model,
    graphs,
    goodness_temp: float,
    batch_size: int = 128,
    critic=None,
    norm: str = "none",
    reducer: str = "logsumexp",
) -> np.ndarray:
    if not graphs:
        return np.asarray([], dtype=float)
    model.eval()
    loader = DataLoader(
        graphs,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
    )
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    vals = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_weight = getattr(batch, "edge_weight", None)
            edge_type = getattr(batch, "edge_type", None)
            h = model(batch.x, batch.edge_index, edge_weight=edge_weight, edge_type=edge_type)
            g = goodness(
                h,
                batch.batch,
                temperature=float(goodness_temp),
                critic=critic,
                norm=str(norm).strip().lower(),
                reducer=str(reducer).strip().lower(),
            )
            vals.extend(g.detach().cpu().tolist())
    return np.asarray(vals, dtype=float)


def infer_graph_goodness_with_uncertainty(
    model,
    graphs,
    goodness_temp: float,
    batch_size: int = 128,
    critic=None,
    norm: str = "none",
    reducer: str = "logsumexp",
) -> tuple[np.ndarray, np.ndarray | None]:
    if not graphs:
        return np.asarray([], dtype=float), None
    model.eval()
    loader = DataLoader(
        graphs,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
    )
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    vals = []
    unc = []
    member_graph_energy = getattr(critic, "member_graph_energy", None) if critic is not None else None
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_weight = getattr(batch, "edge_weight", None)
            edge_type = getattr(batch, "edge_type", None)
            h = model(batch.x, batch.edge_index, edge_weight=edge_weight, edge_type=edge_type)
            g = goodness(
                h,
                batch.batch,
                temperature=float(goodness_temp),
                critic=critic,
                norm=str(norm).strip().lower(),
                reducer=str(reducer).strip().lower(),
            )
            vals.extend(g.detach().cpu().tolist())
            if callable(member_graph_energy):
                ge = member_graph_energy(h, batch.batch, temperature=float(goodness_temp))
                if ge.ndim == 2 and ge.size(0) > 1:
                    std = ge.std(dim=0, unbiased=False)
                    unc.extend(std.detach().cpu().tolist())
    g_np = np.asarray(vals, dtype=float)
    if unc:
        unc_np = np.asarray(unc, dtype=float)
        if unc_np.shape[0] == g_np.shape[0]:
            return g_np, unc_np
    return g_np, None


def infer_node_goodness_with_uncertainty(
    model,
    graphs,
    goodness_temp: float,
    batch_size: int = 32,
    critic=None,
    norm: str = "none",
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    if not graphs:
        return [], []
    model.eval()
    loader = DataLoader(
        graphs,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
    )
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    norm_mode = str(norm).strip().lower()
    out_scores: list[np.ndarray] = []
    out_unc: list[np.ndarray | None] = []
    member_node_energy = getattr(critic, "member_node_energy", None) if critic is not None else None
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_weight = getattr(batch, "edge_weight", None)
            edge_type = getattr(batch, "edge_type", None)
            h = model(batch.x, batch.edge_index, edge_weight=edge_weight, edge_type=edge_type)
            if norm_mode == "layernorm":
                h = F.layer_norm(h, (h.size(-1),))
            if critic is not None:
                node_scores = critic.node_energy(h)
                member_scores = member_node_energy(h) if callable(member_node_energy) else None
            else:
                node_scores = (h * h).mean(dim=1)
                member_scores = None

            _, segment_ids = torch.unique(batch.batch, sorted=True, return_inverse=True)
            num_segments = int(segment_ids.max().item()) + 1 if segment_ids.numel() else 0
            for gid in range(num_segments):
                mask = segment_ids == gid
                out_scores.append(np.asarray(node_scores[mask].detach().cpu().tolist(), dtype=float))
                if member_scores is not None and member_scores.ndim == 2 and member_scores.size(0) > 1:
                    std = member_scores[:, mask].std(dim=0, unbiased=False)
                    out_unc.append(np.asarray(std.detach().cpu().tolist(), dtype=float))
                else:
                    out_unc.append(None)
    return out_scores, out_unc


def _nan_sub(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(a - b)


def _weighted_turnover(
    prev_weights: pd.Series | None,
    next_weights: pd.Series,
) -> float:
    if prev_weights is None or prev_weights.empty:
        return 0.0
    aligned = pd.concat(
        [
            prev_weights.rename("prev"),
            next_weights.rename("next"),
        ],
        axis=1,
    ).fillna(0.0)
    return float(0.5 * np.abs(aligned["next"] - aligned["prev"]).sum())


def _rolling_regime_quantile(
    goodness_vals: np.ndarray,
    regimes: np.ndarray,
    window: int,
    quantile: float,
    min_periods: int,
) -> np.ndarray:
    n = int(goodness_vals.shape[0])
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    w = max(2, int(window))
    q = float(np.clip(quantile, 0.01, 0.99))
    minp = max(2, int(min_periods))
    for i in range(n):
        start = max(0, i - w + 1)
        g_slice = goodness_vals[start : i + 1]
        r_slice = regimes[start : i + 1]
        mask = np.isfinite(g_slice) & (r_slice == regimes[i])
        vals = g_slice[mask]
        if vals.size >= minp:
            out[i] = float(np.quantile(vals, q))
    return out


def evaluate_goodness_strategy(
    dates,
    goodness_scores,
    fwd_ret_1: pd.Series,
    signal_window: int = 126,
    signal_quantile: float = 0.5,
    signal_polarity: str = "high",
    oos_folds: int = 4,
    oos_min_fold_days: int = 63,
    turnover_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    slippage_vol_scale: float = 0.0,
    slippage_vol_lookback: int = 21,
    short_borrow_bps: float = 0.0,
    max_abs_exposure: float = 1.0,
    trading_days: int = 252,
    regime_gate_enabled: bool = False,
    regime_gate_window: int = 63,
    regime_confidence_temp: float = 1.0,
    regime_neutral_exposure: float = 0.0,
    regime_min_confidence: float = 0.0,
    goodness_uncertainty: np.ndarray | None = None,
    regime_uncertainty_scale: float = 0.0,
    risk_signal: np.ndarray | None = None,
    regime_risk_scale: float = 0.0,
    regime_thresholding_enabled: bool = True,
    regime_threshold_window: int | None = None,
    regime_threshold_quantile: float | None = None,
    regime_vol_window: int = 21,
    regime_low_quantile: float = 0.33,
    regime_high_quantile: float = 0.67,
) -> dict[str, float | str]:
    polarity_requested = str(signal_polarity or "high").strip().lower()
    if polarity_requested not in {"high", "low", "auto"}:
        polarity_requested = "high"
    out = {
        "econ_num_days": 0.0,
        "econ_bh_total_return": float("nan"),
        "econ_bh_ann_return": float("nan"),
        "econ_bh_ann_vol": float("nan"),
        "econ_bh_sharpe": float("nan"),
        "econ_bh_sortino": float("nan"),
        "econ_bh_calmar": float("nan"),
        "econ_bh_max_drawdown": float("nan"),
        "econ_bh_max_drawdown_duration_days": float("nan"),
        "econ_bh_var_95_daily": float("nan"),
        "econ_bh_cvar_95_daily": float("nan"),
        "econ_bh_hit_rate_daily": float("nan"),
        "econ_strategy_total_return": float("nan"),
        "econ_strategy_ann_return": float("nan"),
        "econ_strategy_ann_vol": float("nan"),
        "econ_strategy_sharpe": float("nan"),
        "econ_strategy_sortino": float("nan"),
        "econ_strategy_calmar": float("nan"),
        "econ_strategy_max_drawdown": float("nan"),
        "econ_strategy_max_drawdown_duration_days": float("nan"),
        "econ_strategy_var_95_daily": float("nan"),
        "econ_strategy_cvar_95_daily": float("nan"),
        "econ_strategy_hit_rate_daily": float("nan"),
        "econ_exposure_benchmark_exposure": float("nan"),
        "econ_exposure_benchmark_total_return": float("nan"),
        "econ_exposure_benchmark_ann_return": float("nan"),
        "econ_exposure_benchmark_ann_vol": float("nan"),
        "econ_exposure_benchmark_sharpe": float("nan"),
        "econ_exposure_benchmark_sortino": float("nan"),
        "econ_exposure_benchmark_calmar": float("nan"),
        "econ_exposure_benchmark_max_drawdown": float("nan"),
        "econ_exposure_benchmark_max_drawdown_duration_days": float("nan"),
        "econ_exposure_benchmark_var_95_daily": float("nan"),
        "econ_exposure_benchmark_cvar_95_daily": float("nan"),
        "econ_exposure_benchmark_hit_rate_daily": float("nan"),
        "econ_exposure_adjusted_total_return_uplift": float("nan"),
        "econ_exposure_adjusted_ann_return_uplift": float("nan"),
        "econ_exposure_adjusted_sharpe_uplift": float("nan"),
        "econ_exposure_adjusted_sortino_uplift": float("nan"),
        "econ_exposure_adjusted_calmar_uplift": float("nan"),
        "econ_exposure_adjusted_max_drawdown_delta": float("nan"),
        "econ_exposure_adjusted_max_drawdown_duration_delta": float("nan"),
        "econ_exposure_adjusted_var_95_daily_delta": float("nan"),
        "econ_exposure_adjusted_cvar_95_daily_delta": float("nan"),
        "econ_exposure_adjusted_hit_rate_delta": float("nan"),
        "econ_ann_return_uplift": float("nan"),
        "econ_sharpe_uplift": float("nan"),
        "econ_sortino_uplift": float("nan"),
        "econ_calmar_uplift": float("nan"),
        "econ_max_drawdown_delta": float("nan"),
        "econ_max_drawdown_duration_delta": float("nan"),
        "econ_var_95_daily_delta": float("nan"),
        "econ_cvar_95_daily_delta": float("nan"),
        "econ_hit_rate_delta": float("nan"),
        "econ_turnover_mean_daily": float("nan"),
        "econ_avg_cost_bps_applied": float("nan"),
        "econ_avg_borrow_bps_applied": float("nan"),
        "econ_signal_window": float(signal_window),
        "econ_signal_quantile": float(signal_quantile),
        "econ_signal_polarity_requested": polarity_requested,
        "econ_signal_polarity_effective": "high",
        "econ_oos_folds_requested": float(max(1, int(oos_folds))),
        "econ_oos_folds_used": 0.0,
        "econ_oos_min_fold_days": float(max(10, int(oos_min_fold_days))),
        "econ_oos_sharpe_uplift_mean": float("nan"),
        "econ_oos_sharpe_uplift_min": float("nan"),
        "econ_oos_ann_return_uplift_mean": float("nan"),
        "econ_oos_ann_return_uplift_min": float("nan"),
        "econ_turnover_cost_bps": float(turnover_cost_bps),
        "econ_slippage_bps": float(slippage_bps),
        "econ_slippage_vol_scale": float(slippage_vol_scale),
        "econ_slippage_vol_lookback": float(slippage_vol_lookback),
        "econ_short_borrow_bps": float(short_borrow_bps),
        "econ_max_abs_exposure": float(max_abs_exposure),
        "econ_regime_gate_enabled": float(bool(regime_gate_enabled)),
        "econ_regime_confidence_mean": float("nan"),
        "econ_regime_exposure_mean": float("nan"),
        "econ_regime_confidence_temp": float(regime_confidence_temp),
        "econ_regime_neutral_exposure": float(regime_neutral_exposure),
        "econ_regime_min_confidence": float(regime_min_confidence),
        "econ_regime_thresholding_enabled": float(bool(regime_thresholding_enabled)),
        "econ_regime_threshold_window": float(
            signal_window if regime_threshold_window is None else regime_threshold_window
        ),
        "econ_regime_threshold_quantile": float(
            signal_quantile if regime_threshold_quantile is None else regime_threshold_quantile
        ),
        "econ_regime_low_count": float("nan"),
        "econ_regime_mid_count": float("nan"),
        "econ_regime_high_count": float("nan"),
        "econ_regime_vol_window": float(regime_vol_window),
    }

    if fwd_ret_1 is None or len(fwd_ret_1) == 0:
        return out
    if len(dates) == 0:
        return out

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(list(dates)),
            "goodness": np.asarray(goodness_scores, dtype=float),
        }
    )
    df = df[np.isfinite(df["goodness"].to_numpy(dtype=float))]
    if df.empty:
        return out

    bench = fwd_ret_1.copy()
    bench.index = pd.to_datetime(bench.index)
    if bench.index.has_duplicates:
        bench = bench.groupby(level=0).mean()

    df["fwd_ret_1"] = bench.reindex(df["date"]).to_numpy(dtype=float)
    df = df.dropna(subset=["fwd_ret_1"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return out

    sw = max(20, int(signal_window))
    sq = float(np.clip(signal_quantile, 0.05, 0.95))
    roll_q_global = df["goodness"].rolling(
        sw,
        min_periods=max(10, sw // 3),
    ).quantile(sq)
    roll_q_eff = roll_q_global.to_numpy(dtype=float)

    if bool(regime_thresholding_enabled):
        rw = max(2, int(regime_vol_window))
        vol = (
            df["fwd_ret_1"]
            .rolling(rw, min_periods=max(2, rw // 2))
            .std()
            .bfill()
            .ffill()
            .fillna(0.0)
        )
        vol_vals = vol.to_numpy(dtype=float)
        lo_q = float(np.clip(regime_low_quantile, 0.01, 0.49))
        hi_q = float(np.clip(regime_high_quantile, 0.51, 0.99))
        lo_thr = float(np.nanquantile(vol_vals, lo_q)) if np.isfinite(vol_vals).any() else 0.0
        hi_thr = float(np.nanquantile(vol_vals, hi_q)) if np.isfinite(vol_vals).any() else 0.0
        regimes = np.where(vol_vals <= lo_thr, 0, np.where(vol_vals >= hi_thr, 2, 1)).astype(int)

        rt_window = sw if regime_threshold_window is None else max(10, int(regime_threshold_window))
        rt_q = sq if regime_threshold_quantile is None else float(np.clip(regime_threshold_quantile, 0.05, 0.95))
        roll_q_regime = _rolling_regime_quantile(
            df["goodness"].to_numpy(dtype=float),
            regimes,
            window=rt_window,
            quantile=rt_q,
            min_periods=max(10, rt_window // 3),
        )
        roll_q_eff = np.where(np.isfinite(roll_q_regime), roll_q_regime, roll_q_eff)
        out["econ_regime_threshold_window"] = float(rt_window)
        out["econ_regime_threshold_quantile"] = float(rt_q)
        out["econ_regime_low_count"] = float(np.sum(regimes == 0))
        out["econ_regime_mid_count"] = float(np.sum(regimes == 1))
        out["econ_regime_high_count"] = float(np.sum(regimes == 2))

    signal_high = (df["goodness"].to_numpy(dtype=float) >= roll_q_eff).astype(float)
    signal_high = pd.Series(signal_high).fillna(1.0).to_numpy(dtype=float)
    signal_low = 1.0 - signal_high
    exposure_high = np.asarray(signal_high, dtype=float)
    exposure_low = np.asarray(signal_low, dtype=float)

    if bool(regime_gate_enabled):
        rgw = max(10, int(regime_gate_window))
        g_roll_mean = df["goodness"].rolling(
            rgw,
            min_periods=max(5, rgw // 3),
        ).mean()
        g_roll_std = df["goodness"].rolling(
            rgw,
            min_periods=max(5, rgw // 3),
        ).std()
        g_z = (df["goodness"] - g_roll_mean) / (g_roll_std + 1e-8)
        temp = max(1e-6, float(regime_confidence_temp))
        conf = 1.0 / (1.0 + np.exp(-(g_z.to_numpy(dtype=float) / temp)))
        conf = np.nan_to_num(conf, nan=0.5, posinf=1.0, neginf=0.0)

        if goodness_uncertainty is not None:
            unc = np.asarray(goodness_uncertainty, dtype=float)
            if unc.shape[0] == conf.shape[0]:
                unc = np.nan_to_num(unc, nan=float(np.nanmean(unc) if np.isfinite(unc).any() else 0.0))
                conf = conf * np.exp(-max(0.0, float(regime_uncertainty_scale)) * np.maximum(0.0, unc))

        if risk_signal is not None:
            rs = np.asarray(risk_signal, dtype=float)
            if rs.shape[0] == conf.shape[0]:
                rs = np.nan_to_num(rs, nan=0.0)
                conf = conf * (1.0 / (1.0 + np.exp(max(0.0, float(regime_risk_scale)) * rs)))

        conf = np.clip(conf, 0.0, 1.0)
        min_conf = float(np.clip(regime_min_confidence, 0.0, 1.0))
        if min_conf > 0:
            conf = np.where(conf >= min_conf, conf, 0.0)
        neutral = float(np.clip(regime_neutral_exposure, -1.0, 1.0))
        exposure_high = conf * exposure_high + (1.0 - conf) * neutral
        exposure_low = conf * exposure_low + (1.0 - conf) * neutral
        out["econ_regime_confidence_mean"] = float(np.nanmean(conf)) if conf.size else float("nan")

    bench_ret_1 = df["fwd_ret_1"].to_numpy(dtype=float)
    base_cost_bps = max(0.0, float(turnover_cost_bps))
    slip_bps = max(0.0, float(slippage_bps))
    slip_scale = max(0.0, float(slippage_vol_scale))
    borrow_bps = max(0.0, float(short_borrow_bps))
    max_exposure = max(0.0, float(max_abs_exposure))
    vol_lb = max(2, int(slippage_vol_lookback))
    roll_vol = (
        pd.Series(bench_ret_1)
        .rolling(vol_lb, min_periods=max(2, vol_lb // 3))
        .std()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    # slippage_vol_scale is interpreted as additional bps per 1% daily rolling vol.
    slip_curve_bps = slip_bps + (slip_scale * (100.0 * np.maximum(0.0, roll_vol)))
    total_cost_rate = (base_cost_bps + slip_curve_bps) * 1e-4

    def _strategy_for_exposure(
        exposure: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        exposure = np.asarray(exposure, dtype=float)
        if max_exposure > 0:
            exposure = np.clip(exposure, -max_exposure, max_exposure)
        else:
            exposure = np.zeros_like(exposure)
        strat_ret_1 = exposure * bench_ret_1
        turnover = pd.Series(exposure).diff().abs().fillna(0.0).to_numpy(dtype=float)
        borrow_cost_rate = borrow_bps * 1e-4 * np.maximum(0.0, -exposure)
        if np.any(borrow_cost_rate > 0):
            strat_ret_1 = strat_ret_1 - borrow_cost_rate
        if np.any(total_cost_rate > 0):
            strat_ret_1 = strat_ret_1 - total_cost_rate * turnover
        st = strategy_stats("goodness_risk_on_off", strat_ret_1, trading_days=trading_days)
        return exposure, strat_ret_1, turnover, borrow_cost_rate, st

    bh = strategy_stats("benchmark_buy_and_hold", bench_ret_1, trading_days=trading_days)
    candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]] = {
        "high": _strategy_for_exposure(exposure_high),
        "low": _strategy_for_exposure(exposure_low),
    }
    effective = polarity_requested
    if polarity_requested == "auto":
        high_sharpe = float(candidates["high"][4].get("sharpe", float("nan")))
        low_sharpe = float(candidates["low"][4].get("sharpe", float("nan")))
        if np.isfinite(low_sharpe) and (not np.isfinite(high_sharpe) or low_sharpe > high_sharpe):
            effective = "low"
        else:
            effective = "high"
    elif polarity_requested not in candidates:
        effective = "high"

    exposure, strat_ret_1, turnover, borrow_cost_rate, st = candidates[effective]
    exposure_benchmark_weight = (
        float(np.nanmean(np.abs(exposure))) if exposure.size and np.isfinite(exposure).any() else 0.6
    )
    exposure_benchmark_weight = float(np.clip(exposure_benchmark_weight, 0.0, max(0.0, max_exposure)))
    exposure_benchmark_ret_1 = exposure_benchmark_weight * bench_ret_1
    exposure_benchmark = strategy_stats(
        "exposure_adjusted_buy_and_hold",
        exposure_benchmark_ret_1,
        trading_days=trading_days,
    )
    out["econ_signal_polarity_effective"] = effective
    out["econ_regime_exposure_mean"] = float(np.nanmean(exposure)) if exposure.size else float("nan")
    out.update(
        {
            "econ_num_days": float(st["num_days"]),
            "econ_bh_total_return": float(bh["total_return"]),
            "econ_bh_ann_return": float(bh["ann_return"]),
            "econ_bh_ann_vol": float(bh["ann_vol"]),
            "econ_bh_sharpe": float(bh["sharpe"]),
            "econ_bh_sortino": float(bh["sortino"]),
            "econ_bh_calmar": float(bh["calmar"]),
            "econ_bh_max_drawdown": float(bh["max_drawdown"]),
            "econ_bh_max_drawdown_duration_days": float(bh["max_drawdown_duration_days"]),
            "econ_bh_var_95_daily": float(bh["var_95_daily"]),
            "econ_bh_cvar_95_daily": float(bh["cvar_95_daily"]),
            "econ_bh_hit_rate_daily": float(bh["hit_rate_daily"]),
            "econ_strategy_total_return": float(st["total_return"]),
            "econ_strategy_ann_return": float(st["ann_return"]),
            "econ_strategy_ann_vol": float(st["ann_vol"]),
            "econ_strategy_sharpe": float(st["sharpe"]),
            "econ_strategy_sortino": float(st["sortino"]),
            "econ_strategy_calmar": float(st["calmar"]),
            "econ_strategy_max_drawdown": float(st["max_drawdown"]),
            "econ_strategy_max_drawdown_duration_days": float(st["max_drawdown_duration_days"]),
            "econ_strategy_var_95_daily": float(st["var_95_daily"]),
            "econ_strategy_cvar_95_daily": float(st["cvar_95_daily"]),
            "econ_strategy_hit_rate_daily": float(st["hit_rate_daily"]),
            "econ_exposure_benchmark_exposure": exposure_benchmark_weight,
            "econ_exposure_benchmark_total_return": float(exposure_benchmark["total_return"]),
            "econ_exposure_benchmark_ann_return": float(exposure_benchmark["ann_return"]),
            "econ_exposure_benchmark_ann_vol": float(exposure_benchmark["ann_vol"]),
            "econ_exposure_benchmark_sharpe": float(exposure_benchmark["sharpe"]),
            "econ_exposure_benchmark_sortino": float(exposure_benchmark["sortino"]),
            "econ_exposure_benchmark_calmar": float(exposure_benchmark["calmar"]),
            "econ_exposure_benchmark_max_drawdown": float(exposure_benchmark["max_drawdown"]),
            "econ_exposure_benchmark_max_drawdown_duration_days": float(
                exposure_benchmark["max_drawdown_duration_days"]
            ),
            "econ_exposure_benchmark_var_95_daily": float(exposure_benchmark["var_95_daily"]),
            "econ_exposure_benchmark_cvar_95_daily": float(exposure_benchmark["cvar_95_daily"]),
            "econ_exposure_benchmark_hit_rate_daily": float(exposure_benchmark["hit_rate_daily"]),
            "econ_exposure_adjusted_total_return_uplift": _nan_sub(
                float(st["total_return"]),
                float(exposure_benchmark["total_return"]),
            ),
            "econ_exposure_adjusted_ann_return_uplift": _nan_sub(
                float(st["ann_return"]),
                float(exposure_benchmark["ann_return"]),
            ),
            "econ_exposure_adjusted_sharpe_uplift": _nan_sub(
                float(st["sharpe"]),
                float(exposure_benchmark["sharpe"]),
            ),
            "econ_exposure_adjusted_sortino_uplift": _nan_sub(
                float(st["sortino"]),
                float(exposure_benchmark["sortino"]),
            ),
            "econ_exposure_adjusted_calmar_uplift": _nan_sub(
                float(st["calmar"]),
                float(exposure_benchmark["calmar"]),
            ),
            "econ_exposure_adjusted_max_drawdown_delta": _nan_sub(
                float(st["max_drawdown"]),
                float(exposure_benchmark["max_drawdown"]),
            ),
            "econ_exposure_adjusted_max_drawdown_duration_delta": _nan_sub(
                float(st["max_drawdown_duration_days"]),
                float(exposure_benchmark["max_drawdown_duration_days"]),
            ),
            "econ_exposure_adjusted_var_95_daily_delta": _nan_sub(
                float(st["var_95_daily"]),
                float(exposure_benchmark["var_95_daily"]),
            ),
            "econ_exposure_adjusted_cvar_95_daily_delta": _nan_sub(
                float(st["cvar_95_daily"]),
                float(exposure_benchmark["cvar_95_daily"]),
            ),
            "econ_exposure_adjusted_hit_rate_delta": _nan_sub(
                float(st["hit_rate_daily"]),
                float(exposure_benchmark["hit_rate_daily"]),
            ),
            "econ_ann_return_uplift": _nan_sub(float(st["ann_return"]), float(bh["ann_return"])),
            "econ_sharpe_uplift": _nan_sub(float(st["sharpe"]), float(bh["sharpe"])),
            "econ_sortino_uplift": _nan_sub(float(st["sortino"]), float(bh["sortino"])),
            "econ_calmar_uplift": _nan_sub(float(st["calmar"]), float(bh["calmar"])),
            "econ_max_drawdown_delta": _nan_sub(float(st["max_drawdown"]), float(bh["max_drawdown"])),
            "econ_max_drawdown_duration_delta": _nan_sub(
                float(st["max_drawdown_duration_days"]),
                float(bh["max_drawdown_duration_days"]),
            ),
            "econ_var_95_daily_delta": _nan_sub(float(st["var_95_daily"]), float(bh["var_95_daily"])),
            "econ_cvar_95_daily_delta": _nan_sub(float(st["cvar_95_daily"]), float(bh["cvar_95_daily"])),
            "econ_hit_rate_delta": _nan_sub(float(st["hit_rate_daily"]), float(bh["hit_rate_daily"])),
            "econ_turnover_mean_daily": float(np.nanmean(turnover)) if turnover.size else 0.0,
            "econ_avg_cost_bps_applied": float(np.nanmean((base_cost_bps + slip_curve_bps) * turnover))
            if turnover.size
            else 0.0,
            "econ_avg_borrow_bps_applied": float(np.nanmean(1e4 * borrow_cost_rate))
            if borrow_cost_rate.size
            else 0.0,
            "econ_signal_window": float(sw),
            "econ_signal_quantile": float(sq),
            "econ_turnover_cost_bps": float(turnover_cost_bps),
            "econ_slippage_bps": float(slippage_bps),
            "econ_slippage_vol_scale": float(slippage_vol_scale),
            "econ_slippage_vol_lookback": float(vol_lb),
            "econ_short_borrow_bps": float(short_borrow_bps),
            "econ_max_abs_exposure": float(max_exposure),
        }
    )

    valid_mask = np.isfinite(strat_ret_1) & np.isfinite(bench_ret_1)
    strat_valid = np.asarray(strat_ret_1[valid_mask], dtype=float)
    bench_valid = np.asarray(bench_ret_1[valid_mask], dtype=float)
    requested_folds = max(1, int(oos_folds))
    min_fold_days = max(10, int(oos_min_fold_days))
    out["econ_oos_folds_requested"] = float(requested_folds)
    out["econ_oos_min_fold_days"] = float(min_fold_days)
    if requested_folds >= 2 and strat_valid.size >= 2 * min_fold_days:
        max_folds = int(strat_valid.size // min_fold_days)
        use_folds = max(2, min(requested_folds, max_folds))
        base_fold_size = int(strat_valid.size // use_folds)
        remainder = int(strat_valid.size % use_folds)
        start = 0
        used_folds = 0
        sharpe_uplifts: list[float] = []
        ann_uplifts: list[float] = []
        for fold_idx in range(use_folds):
            fold_size = base_fold_size + (1 if fold_idx < remainder else 0)
            end = start + fold_size
            if fold_size < min_fold_days:
                start = end
                continue
            st_fold = strategy_stats(
                f"goodness_risk_on_off_fold_{fold_idx}",
                strat_valid[start:end],
                trading_days=trading_days,
            )
            bh_fold = strategy_stats(
                f"benchmark_buy_and_hold_fold_{fold_idx}",
                bench_valid[start:end],
                trading_days=trading_days,
            )
            sharpe_u = _nan_sub(float(st_fold["sharpe"]), float(bh_fold["sharpe"]))
            ann_u = _nan_sub(float(st_fold["ann_return"]), float(bh_fold["ann_return"]))
            if np.isfinite(sharpe_u):
                sharpe_uplifts.append(float(sharpe_u))
            if np.isfinite(ann_u):
                ann_uplifts.append(float(ann_u))
            used_folds += 1
            start = end
        out["econ_oos_folds_used"] = float(used_folds)
        if sharpe_uplifts:
            out["econ_oos_sharpe_uplift_mean"] = float(np.mean(sharpe_uplifts))
            out["econ_oos_sharpe_uplift_min"] = float(np.min(sharpe_uplifts))
        if ann_uplifts:
            out["econ_oos_ann_return_uplift_mean"] = float(np.mean(ann_uplifts))
            out["econ_oos_ann_return_uplift_min"] = float(np.min(ann_uplifts))
    return out


def evaluate_cross_sectional_goodness_strategy(
    dates,
    node_tickers,
    node_scores,
    forward_returns: pd.DataFrame,
    top_k: int | None = None,
    bottom_k: int | None = None,
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
    signal_polarity: str = "high",
    turnover_cost_bps: float = 0.0,
    short_borrow_bps: float = 0.0,
    max_gross_exposure: float = 1.0,
    trading_days: int = 252,
    min_names: int = 4,
    oos_folds: int = 4,
    oos_min_fold_days: int = 63,
    uncertainty_scores: list[np.ndarray | None] | None = None,
    uncertainty_scale: float = 0.0,
) -> dict[str, float | str]:
    polarity_requested = str(signal_polarity or "high").strip().lower()
    if polarity_requested not in {"high", "low", "auto"}:
        polarity_requested = "high"
    out = {
        "econ_ls_num_days": 0.0,
        "econ_ls_strategy_total_return": float("nan"),
        "econ_ls_strategy_ann_return": float("nan"),
        "econ_ls_strategy_ann_vol": float("nan"),
        "econ_ls_strategy_sharpe": float("nan"),
        "econ_ls_strategy_sortino": float("nan"),
        "econ_ls_strategy_calmar": float("nan"),
        "econ_ls_strategy_max_drawdown": float("nan"),
        "econ_ls_strategy_max_drawdown_duration_days": float("nan"),
        "econ_ls_strategy_var_95_daily": float("nan"),
        "econ_ls_strategy_cvar_95_daily": float("nan"),
        "econ_ls_strategy_hit_rate_daily": float("nan"),
        "econ_ls_equal_weight_total_return": float("nan"),
        "econ_ls_equal_weight_ann_return": float("nan"),
        "econ_ls_equal_weight_ann_vol": float("nan"),
        "econ_ls_equal_weight_sharpe": float("nan"),
        "econ_ls_equal_weight_sortino": float("nan"),
        "econ_ls_equal_weight_calmar": float("nan"),
        "econ_ls_equal_weight_max_drawdown": float("nan"),
        "econ_ls_equal_weight_max_drawdown_duration_days": float("nan"),
        "econ_ls_equal_weight_var_95_daily": float("nan"),
        "econ_ls_equal_weight_cvar_95_daily": float("nan"),
        "econ_ls_equal_weight_hit_rate_daily": float("nan"),
        "econ_ls_ann_return_uplift": float("nan"),
        "econ_ls_sharpe_uplift": float("nan"),
        "econ_ls_sortino_uplift": float("nan"),
        "econ_ls_calmar_uplift": float("nan"),
        "econ_ls_max_drawdown_delta": float("nan"),
        "econ_ls_max_drawdown_duration_delta": float("nan"),
        "econ_ls_var_95_daily_delta": float("nan"),
        "econ_ls_cvar_95_daily_delta": float("nan"),
        "econ_ls_hit_rate_delta": float("nan"),
        "econ_ls_turnover_mean_daily": float("nan"),
        "econ_ls_avg_cost_bps_applied": float("nan"),
        "econ_ls_avg_borrow_bps_applied": float("nan"),
        "econ_ls_signal_polarity_requested": polarity_requested,
        "econ_ls_signal_polarity_effective": "high",
        "econ_ls_top_frac": float(top_frac),
        "econ_ls_bottom_frac": float(bottom_frac),
        "econ_ls_top_k": float(top_k or 0),
        "econ_ls_bottom_k": float(bottom_k or 0),
        "econ_ls_min_names": float(max(2, int(min_names))),
        "econ_ls_max_gross_exposure": float(max_gross_exposure),
        "econ_ls_uncertainty_scale": float(uncertainty_scale),
        "econ_ls_avg_names": float("nan"),
        "econ_ls_avg_gross_exposure": float("nan"),
        "econ_ls_oos_folds_requested": float(max(1, int(oos_folds))),
        "econ_ls_oos_folds_used": 0.0,
        "econ_ls_oos_min_fold_days": float(max(10, int(oos_min_fold_days))),
        "econ_ls_oos_sharpe_uplift_mean": float("nan"),
        "econ_ls_oos_sharpe_uplift_min": float("nan"),
        "econ_ls_oos_ann_return_uplift_mean": float("nan"),
        "econ_ls_oos_ann_return_uplift_min": float("nan"),
    }
    if (
        dates is None
        or len(dates) == 0
        or not node_tickers
        or not node_scores
        or forward_returns is None
        or forward_returns.empty
    ):
        return out

    min_names = max(2, int(min_names))
    max_gross = max(0.0, float(max_gross_exposure))
    cost_bps = max(0.0, float(turnover_cost_bps))
    borrow_bps = max(0.0, float(short_borrow_bps))
    unc_scale = max(0.0, float(uncertainty_scale))
    fwd = forward_returns.copy()
    fwd.index = pd.to_datetime(fwd.index)
    if fwd.index.has_duplicates:
        fwd = fwd.groupby(level=0).mean()

    def _build_candidate(high_first: bool):
        strat_rets: list[float] = []
        bench_rets: list[float] = []
        turnovers: list[float] = []
        borrow_rates: list[float] = []
        names_used: list[int] = []
        gross_used: list[float] = []
        prev_weights: pd.Series | None = None

        for idx, date_raw in enumerate(dates):
            if idx >= len(node_tickers) or idx >= len(node_scores):
                break
            date = pd.to_datetime(date_raw)
            if date not in fwd.index:
                continue
            tickers = [str(t).upper().strip() for t in list(node_tickers[idx])]
            scores = np.asarray(node_scores[idx], dtype=float)
            if scores.size != len(tickers):
                continue
            unc = None
            if uncertainty_scores is not None and idx < len(uncertainty_scores):
                unc_raw = uncertainty_scores[idx]
                if unc_raw is not None:
                    unc_arr = np.asarray(unc_raw, dtype=float)
                    if unc_arr.shape == scores.shape:
                        unc = unc_arr
            returns_row = fwd.loc[date]
            if isinstance(returns_row, pd.DataFrame):
                returns_row = returns_row.iloc[0]
            returns_vals = returns_row.reindex(tickers).to_numpy(dtype=float)
            mask = np.isfinite(scores) & np.isfinite(returns_vals)
            if unc is not None:
                mask = mask & np.isfinite(unc)
            if int(mask.sum()) < min_names:
                continue
            tickers_arr = np.asarray(tickers, dtype=object)[mask]
            score_arr = np.asarray(scores[mask], dtype=float)
            return_arr = np.asarray(returns_vals[mask], dtype=float)
            unc_arr = np.asarray(unc[mask], dtype=float) if unc is not None else None
            if unc_arr is not None and unc_scale > 0:
                score_rank = score_arr / (1.0 + unc_scale * np.maximum(0.0, unc_arr))
            else:
                score_rank = score_arr

            n_names = int(score_arr.shape[0])
            k_long = max(1, int(round(float(top_frac) * n_names)))
            k_short = max(1, int(round(float(bottom_frac) * n_names)))
            if top_k is not None and int(top_k) > 0:
                k_long = min(k_long, int(top_k))
            if bottom_k is not None and int(bottom_k) > 0:
                k_short = min(k_short, int(bottom_k))
            k_long = min(k_long, max(1, n_names // 2))
            k_short = min(k_short, max(1, n_names - k_long))
            if k_long < 1 or k_short < 1:
                continue

            order = np.argsort(score_rank)
            long_idx = order[-k_long:] if high_first else order[:k_long]
            short_idx = order[:k_short] if high_first else order[-k_short:]
            selected_unc = None
            if unc_arr is not None and (long_idx.size + short_idx.size) > 0:
                selected_unc = float(
                    np.nanmean(
                        np.concatenate(
                            [
                                unc_arr[long_idx],
                                unc_arr[short_idx],
                            ]
                        )
                    )
                )
            gross_scale = 1.0
            if selected_unc is not None and unc_scale > 0:
                gross_scale = float(np.clip(np.exp(-unc_scale * max(0.0, selected_unc)), 0.0, 1.0))
            gross = max_gross * gross_scale
            long_weight_total = 0.5 * gross
            short_weight_total = 0.5 * gross

            weights = pd.Series(0.0, index=pd.Index(tickers_arr, dtype=object), dtype=float)
            weights.iloc[long_idx] = weights.iloc[long_idx] + (long_weight_total / max(1, long_idx.size))
            weights.iloc[short_idx] = weights.iloc[short_idx] - (short_weight_total / max(1, short_idx.size))
            weights = weights.groupby(level=0).sum()
            ret_aligned = returns_row.reindex(weights.index).to_numpy(dtype=float)
            bench_mask = np.isfinite(return_arr)
            bench_ret = float(np.nanmean(return_arr[bench_mask])) if np.any(bench_mask) else float("nan")
            turnover = _weighted_turnover(prev_weights, weights)
            borrow_rate = borrow_bps * 1e-4 * float(np.maximum(0.0, -weights).sum())
            strat_ret = float(np.dot(weights.to_numpy(dtype=float), ret_aligned))
            strat_ret -= cost_bps * 1e-4 * turnover
            strat_ret -= borrow_rate

            strat_rets.append(strat_ret)
            bench_rets.append(bench_ret)
            turnovers.append(turnover)
            borrow_rates.append(borrow_rate)
            names_used.append(int(n_names))
            gross_used.append(float(np.abs(weights).sum()))
            prev_weights = weights

        return (
            np.asarray(strat_rets, dtype=float),
            np.asarray(bench_rets, dtype=float),
            np.asarray(turnovers, dtype=float),
            np.asarray(borrow_rates, dtype=float),
            np.asarray(names_used, dtype=float),
            np.asarray(gross_used, dtype=float),
        )

    candidates = {
        "high": _build_candidate(high_first=True),
        "low": _build_candidate(high_first=False),
    }
    effective = polarity_requested
    if polarity_requested == "auto":
        high_stats = strategy_stats("econ_ls_high", candidates["high"][0], trading_days=trading_days)
        low_stats = strategy_stats("econ_ls_low", candidates["low"][0], trading_days=trading_days)
        high_sharpe = float(high_stats.get("sharpe", float("nan")))
        low_sharpe = float(low_stats.get("sharpe", float("nan")))
        if np.isfinite(low_sharpe) and (not np.isfinite(high_sharpe) or low_sharpe > high_sharpe):
            effective = "low"
        else:
            effective = "high"
    elif effective not in candidates:
        effective = "high"

    strat_ret_1, bench_ret_1, turnover, borrow_rates, names_used, gross_used = candidates[effective]
    if strat_ret_1.size == 0 or bench_ret_1.size == 0:
        out["econ_ls_signal_polarity_effective"] = effective
        return out

    st = strategy_stats("goodness_cross_sectional_long_short", strat_ret_1, trading_days=trading_days)
    bh = strategy_stats("constituent_equal_weight_long_only", bench_ret_1, trading_days=trading_days)
    out.update(
        {
            "econ_ls_num_days": float(st["num_days"]),
            "econ_ls_strategy_total_return": float(st["total_return"]),
            "econ_ls_strategy_ann_return": float(st["ann_return"]),
            "econ_ls_strategy_ann_vol": float(st["ann_vol"]),
            "econ_ls_strategy_sharpe": float(st["sharpe"]),
            "econ_ls_strategy_sortino": float(st["sortino"]),
            "econ_ls_strategy_calmar": float(st["calmar"]),
            "econ_ls_strategy_max_drawdown": float(st["max_drawdown"]),
            "econ_ls_strategy_max_drawdown_duration_days": float(st["max_drawdown_duration_days"]),
            "econ_ls_strategy_var_95_daily": float(st["var_95_daily"]),
            "econ_ls_strategy_cvar_95_daily": float(st["cvar_95_daily"]),
            "econ_ls_strategy_hit_rate_daily": float(st["hit_rate_daily"]),
            "econ_ls_equal_weight_total_return": float(bh["total_return"]),
            "econ_ls_equal_weight_ann_return": float(bh["ann_return"]),
            "econ_ls_equal_weight_ann_vol": float(bh["ann_vol"]),
            "econ_ls_equal_weight_sharpe": float(bh["sharpe"]),
            "econ_ls_equal_weight_sortino": float(bh["sortino"]),
            "econ_ls_equal_weight_calmar": float(bh["calmar"]),
            "econ_ls_equal_weight_max_drawdown": float(bh["max_drawdown"]),
            "econ_ls_equal_weight_max_drawdown_duration_days": float(bh["max_drawdown_duration_days"]),
            "econ_ls_equal_weight_var_95_daily": float(bh["var_95_daily"]),
            "econ_ls_equal_weight_cvar_95_daily": float(bh["cvar_95_daily"]),
            "econ_ls_equal_weight_hit_rate_daily": float(bh["hit_rate_daily"]),
            "econ_ls_ann_return_uplift": _nan_sub(float(st["ann_return"]), float(bh["ann_return"])),
            "econ_ls_sharpe_uplift": _nan_sub(float(st["sharpe"]), float(bh["sharpe"])),
            "econ_ls_sortino_uplift": _nan_sub(float(st["sortino"]), float(bh["sortino"])),
            "econ_ls_calmar_uplift": _nan_sub(float(st["calmar"]), float(bh["calmar"])),
            "econ_ls_max_drawdown_delta": _nan_sub(float(st["max_drawdown"]), float(bh["max_drawdown"])),
            "econ_ls_max_drawdown_duration_delta": _nan_sub(
                float(st["max_drawdown_duration_days"]),
                float(bh["max_drawdown_duration_days"]),
            ),
            "econ_ls_var_95_daily_delta": _nan_sub(float(st["var_95_daily"]), float(bh["var_95_daily"])),
            "econ_ls_cvar_95_daily_delta": _nan_sub(float(st["cvar_95_daily"]), float(bh["cvar_95_daily"])),
            "econ_ls_hit_rate_delta": _nan_sub(float(st["hit_rate_daily"]), float(bh["hit_rate_daily"])),
            "econ_ls_turnover_mean_daily": float(np.nanmean(turnover)) if turnover.size else 0.0,
            "econ_ls_avg_cost_bps_applied": float(np.nanmean(1e4 * cost_bps * 1e-4 * turnover))
            if turnover.size
            else 0.0,
            "econ_ls_avg_borrow_bps_applied": float(np.nanmean(1e4 * borrow_rates))
            if borrow_rates.size
            else 0.0,
            "econ_ls_signal_polarity_effective": effective,
            "econ_ls_avg_names": float(np.nanmean(names_used)) if names_used.size else float("nan"),
            "econ_ls_avg_gross_exposure": float(np.nanmean(gross_used)) if gross_used.size else float("nan"),
        }
    )

    valid_mask = np.isfinite(strat_ret_1) & np.isfinite(bench_ret_1)
    strat_valid = np.asarray(strat_ret_1[valid_mask], dtype=float)
    bench_valid = np.asarray(bench_ret_1[valid_mask], dtype=float)
    requested_folds = max(1, int(oos_folds))
    min_fold_days = max(10, int(oos_min_fold_days))
    if requested_folds >= 2 and strat_valid.size >= 2 * min_fold_days:
        max_folds = int(strat_valid.size // min_fold_days)
        use_folds = max(2, min(requested_folds, max_folds))
        base_fold_size = int(strat_valid.size // use_folds)
        remainder = int(strat_valid.size % use_folds)
        start = 0
        used_folds = 0
        sharpe_uplifts: list[float] = []
        ann_uplifts: list[float] = []
        for fold_idx in range(use_folds):
            fold_size = base_fold_size + (1 if fold_idx < remainder else 0)
            end = start + fold_size
            if fold_size < min_fold_days:
                start = end
                continue
            st_fold = strategy_stats(
                f"goodness_cross_sectional_long_short_fold_{fold_idx}",
                strat_valid[start:end],
                trading_days=trading_days,
            )
            bh_fold = strategy_stats(
                f"constituent_equal_weight_long_only_fold_{fold_idx}",
                bench_valid[start:end],
                trading_days=trading_days,
            )
            sharpe_u = _nan_sub(float(st_fold["sharpe"]), float(bh_fold["sharpe"]))
            ann_u = _nan_sub(float(st_fold["ann_return"]), float(bh_fold["ann_return"]))
            if np.isfinite(sharpe_u):
                sharpe_uplifts.append(float(sharpe_u))
            if np.isfinite(ann_u):
                ann_uplifts.append(float(ann_u))
            used_folds += 1
            start = end
        out["econ_ls_oos_folds_used"] = float(used_folds)
        if sharpe_uplifts:
            out["econ_ls_oos_sharpe_uplift_mean"] = float(np.mean(sharpe_uplifts))
            out["econ_ls_oos_sharpe_uplift_min"] = float(np.min(sharpe_uplifts))
        if ann_uplifts:
            out["econ_ls_oos_ann_return_uplift_mean"] = float(np.mean(ann_uplifts))
            out["econ_ls_oos_ann_return_uplift_min"] = float(np.min(ann_uplifts))
    return out
