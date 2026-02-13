from __future__ import annotations

from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd
import torch
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
            "max_drawdown": float("nan"),
            "cvar_95_daily": float("nan"),
            "hit_rate_daily": float("nan"),
        }
    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    ann_return = float(equity[-1] ** (trading_days / max(1, r.size)) - 1.0)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(trading_days)) if r.size > 1 else 0.0
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-12 else float("nan")
    cvar_cut = np.quantile(r, 0.05)
    cvar_95 = float(r[r <= cvar_cut].mean()) if np.any(r <= cvar_cut) else float(cvar_cut)
    return {
        "strategy": name,
        "num_days": int(r.size),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
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
            h = model(batch.x, batch.edge_index, edge_weight=edge_weight)
            g = goodness(h, batch.batch, temperature=float(goodness_temp))
            vals.extend(g.detach().cpu().tolist())
    return np.asarray(vals, dtype=float)


def _nan_sub(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(a - b)


def evaluate_goodness_strategy(
    dates,
    goodness_scores,
    fwd_ret_1: pd.Series,
    signal_window: int = 126,
    signal_quantile: float = 0.5,
    turnover_cost_bps: float = 0.0,
    trading_days: int = 252,
) -> dict[str, float]:
    out = {
        "econ_num_days": 0.0,
        "econ_bh_total_return": float("nan"),
        "econ_bh_ann_return": float("nan"),
        "econ_bh_ann_vol": float("nan"),
        "econ_bh_sharpe": float("nan"),
        "econ_bh_max_drawdown": float("nan"),
        "econ_bh_cvar_95_daily": float("nan"),
        "econ_bh_hit_rate_daily": float("nan"),
        "econ_strategy_total_return": float("nan"),
        "econ_strategy_ann_return": float("nan"),
        "econ_strategy_ann_vol": float("nan"),
        "econ_strategy_sharpe": float("nan"),
        "econ_strategy_max_drawdown": float("nan"),
        "econ_strategy_cvar_95_daily": float("nan"),
        "econ_strategy_hit_rate_daily": float("nan"),
        "econ_ann_return_uplift": float("nan"),
        "econ_sharpe_uplift": float("nan"),
        "econ_max_drawdown_delta": float("nan"),
        "econ_cvar_95_daily_delta": float("nan"),
        "econ_hit_rate_delta": float("nan"),
        "econ_turnover_mean_daily": float("nan"),
        "econ_signal_window": float(signal_window),
        "econ_signal_quantile": float(signal_quantile),
        "econ_turnover_cost_bps": float(turnover_cost_bps),
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
    roll_q = df["goodness"].rolling(
        sw,
        min_periods=max(10, sw // 3),
    ).quantile(sq)
    signal = (df["goodness"] >= roll_q).astype(float).fillna(1.0)
    bench_ret_1 = df["fwd_ret_1"].to_numpy(dtype=float)
    strat_ret_1 = signal.to_numpy(dtype=float) * bench_ret_1
    turnover = signal.diff().abs().fillna(0.0).to_numpy(dtype=float)
    cost = max(0.0, float(turnover_cost_bps)) * 1e-4
    if cost > 0:
        strat_ret_1 = strat_ret_1 - cost * turnover

    bh = strategy_stats("benchmark_buy_and_hold", bench_ret_1, trading_days=trading_days)
    st = strategy_stats("goodness_risk_on_off", strat_ret_1, trading_days=trading_days)
    out.update(
        {
            "econ_num_days": float(st["num_days"]),
            "econ_bh_total_return": float(bh["total_return"]),
            "econ_bh_ann_return": float(bh["ann_return"]),
            "econ_bh_ann_vol": float(bh["ann_vol"]),
            "econ_bh_sharpe": float(bh["sharpe"]),
            "econ_bh_max_drawdown": float(bh["max_drawdown"]),
            "econ_bh_cvar_95_daily": float(bh["cvar_95_daily"]),
            "econ_bh_hit_rate_daily": float(bh["hit_rate_daily"]),
            "econ_strategy_total_return": float(st["total_return"]),
            "econ_strategy_ann_return": float(st["ann_return"]),
            "econ_strategy_ann_vol": float(st["ann_vol"]),
            "econ_strategy_sharpe": float(st["sharpe"]),
            "econ_strategy_max_drawdown": float(st["max_drawdown"]),
            "econ_strategy_cvar_95_daily": float(st["cvar_95_daily"]),
            "econ_strategy_hit_rate_daily": float(st["hit_rate_daily"]),
            "econ_ann_return_uplift": _nan_sub(float(st["ann_return"]), float(bh["ann_return"])),
            "econ_sharpe_uplift": _nan_sub(float(st["sharpe"]), float(bh["sharpe"])),
            "econ_max_drawdown_delta": _nan_sub(float(st["max_drawdown"]), float(bh["max_drawdown"])),
            "econ_cvar_95_daily_delta": _nan_sub(float(st["cvar_95_daily"]), float(bh["cvar_95_daily"])),
            "econ_hit_rate_delta": _nan_sub(float(st["hit_rate_daily"]), float(bh["hit_rate_daily"])),
            "econ_turnover_mean_daily": float(np.nanmean(turnover)) if turnover.size else 0.0,
            "econ_signal_window": float(sw),
            "econ_signal_quantile": float(sq),
            "econ_turnover_cost_bps": float(turnover_cost_bps),
        }
    )
    return out
