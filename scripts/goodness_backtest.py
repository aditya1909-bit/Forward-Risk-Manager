#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import GCNEncoder
from frisk.ff import goodness
from frisk.eval_metrics import binary_auroc
from frisk.econ_eval import max_drawdown, resolve_price_ticker, strategy_stats


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest goodness vs future risk.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--prices", default="", help="Override prices.csv path")
    parser.add_argument(
        "--ticker",
        default="AUTO",
        help="Benchmark ticker (e.g., MDY). Use AUTO to select a viable ticker.",
    )
    parser.add_argument("--horizons", default="5,21", help="Comma-separated horizons (days)")
    parser.add_argument(
        "--max-abs-logret",
        type=float,
        default=0.5,
        help="Drop benchmark log-returns with abs(value) above this threshold (<=0 disables).",
    )
    parser.add_argument(
        "--out-csv",
        default="runs/experiments/manual/diagnostics/goodness_backtest.csv",
        help="Output CSV",
    )
    parser.add_argument(
        "--out-quantiles",
        default="runs/experiments/manual/diagnostics/goodness_quantiles.csv",
        help="Quantile CSV",
    )
    parser.add_argument(
        "--out-plot",
        default="runs/experiments/manual/diagnostics/goodness_scatter.png",
        help="Scatter plot",
    )
    parser.add_argument(
        "--out-events",
        default="runs/experiments/manual/diagnostics/goodness_events.csv",
        help="Regime/OOD summary CSV",
    )
    parser.add_argument(
        "--out-strategy",
        default="runs/experiments/manual/diagnostics/goodness_strategy_metrics.csv",
        help="Economic strategy metrics CSV",
    )
    parser.add_argument(
        "--out-timeline",
        default="runs/experiments/manual/diagnostics/goodness_timeline.png",
        help="Goodness timeline plot",
    )
    parser.add_argument(
        "--goodness-z-window",
        type=int,
        default=126,
        help="Rolling window for goodness z-score anomaly.",
    )
    parser.add_argument(
        "--signal-window",
        type=int,
        default=126,
        help="Rolling window for goodness threshold used by risk-on/off strategy.",
    )
    parser.add_argument(
        "--signal-quantile",
        type=float,
        default=0.5,
        help="Rolling goodness quantile for risk-on signal; higher goodness => risk-on.",
    )
    parser.add_argument(
        "--turnover-cost-bps",
        type=float,
        default=0.0,
        help="One-way turnover cost in basis points for the strategy.",
    )
    args = parser.parse_args()
    if args.ticker:
        args.ticker = args.ticker.strip().upper()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})
    build_cfg = cfg.get("build_graphs", {})

    graphs_path = Path(train_cfg.get("graphs", "data/processed/graphs.pt"))
    try:
        payload = torch.load(graphs_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(graphs_path, map_location="cpu")
    graphs = payload["graphs"]
    dates = payload.get("dates", [])
    if not graphs or not dates:
        raise ValueError("Graphs or dates missing.")

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=int(train_cfg.get("hidden_dim", 64)),
        num_layers=int(train_cfg.get("num_layers", 2)),
        dropout=float(train_cfg.get("dropout", 0.1)),
    )
    model_path = train_cfg.get("save_model", "")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError("Model checkpoint not found. Train and save model first.")
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    prices_path = args.prices or build_cfg.get("prices", "data/processed/prices.csv")
    ticker_eff, ticker_src, ticker_rows = resolve_price_ticker(
        prices_path=prices_path,
        requested_ticker=str(args.ticker or "AUTO"),
        min_rows=max(32, int(args.signal_window) // 2),
    )
    print(
        "backtest ticker: "
        f"requested={args.ticker} effective={ticker_eff} source={ticker_src} rows={ticker_rows}"
    )
    prices = pd.read_csv(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["ticker"].astype(str).str.upper() == ticker_eff].sort_values("date")
    if prices.empty:
        raise ValueError(f"Ticker {ticker_eff} not found in prices.")

    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px = prices.set_index("date")[price_col].astype(float)
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    px = px[px > 0]
    if px.index.has_duplicates:
        px = px.groupby(level=0).median()
    logret = np.log(px).diff().dropna()
    if args.max_abs_logret and args.max_abs_logret > 0:
        logret = logret[np.abs(logret) <= float(args.max_abs_logret)]

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    records = []

    for i, d in enumerate(dates):
        dt = pd.to_datetime(d)
        if dt not in logret.index:
            continue
        idx = logret.index.get_loc(dt)
        if isinstance(idx, slice):
            idx = idx.start
        elif isinstance(idx, (np.ndarray, list)):
            idx = int(idx[0])

        data = graphs[i]
        h = model(data.x, data.edge_index, edge_weight=getattr(data, "edge_weight", None))
        g = goodness(h, torch.zeros(data.num_nodes, dtype=torch.long), temperature=float(train_cfg.get("goodness_temp", 1.0))).mean().item()

        rec = {"date": dt, "goodness": g}
        if idx + 1 < len(logret):
            rec["fwd_ret_1"] = float(np.exp(logret.iloc[idx + 1]) - 1.0)
        else:
            rec["fwd_ret_1"] = np.nan
        for hlen in horizons:
            if idx + hlen >= len(logret):
                rec[f"fwd_vol_{hlen}"] = np.nan
                rec[f"fwd_dd_{hlen}"] = np.nan
                rec[f"fwd_ret_{hlen}"] = np.nan
                continue
            window = logret.iloc[idx + 1 : idx + 1 + hlen].values
            vol = float(np.std(window))
            cum = np.exp(np.cumsum(window))
            dd = max_drawdown(cum)
            ret = float(cum[-1] - 1.0)
            rec[f"fwd_vol_{hlen}"] = vol
            rec[f"fwd_dd_{hlen}"] = dd
            rec[f"fwd_ret_{hlen}"] = ret
        records.append(rec)

    df = pd.DataFrame(records).dropna()
    df = df.sort_values("date").reset_index(drop=True)
    z_window = max(20, int(args.goodness_z_window))
    roll_mean = df["goodness"].rolling(z_window, min_periods=max(10, z_window // 3)).mean()
    roll_std = df["goodness"].rolling(z_window, min_periods=max(10, z_window // 3)).std()
    df["goodness_z"] = (df["goodness"] - roll_mean) / (roll_std + 1e-8)
    # Low goodness => high anomaly score.
    df["goodness_anomaly"] = -df["goodness_z"].fillna(0.0)

    def _regime_label(ts: pd.Timestamp) -> str:
        d = pd.Timestamp(ts)
        if pd.Timestamp("2008-09-01") <= d <= pd.Timestamp("2009-06-30"):
            return "crisis_2008"
        if pd.Timestamp("2020-02-15") <= d <= pd.Timestamp("2020-05-31"):
            return "covid_2020"
        if pd.Timestamp("2022-01-01") <= d <= pd.Timestamp("2022-10-31"):
            return "regime_2022"
        return "other"

    df["regime"] = df["date"].apply(_regime_label)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Quantile table
    quant_rows = []
    df["goodness_decile"] = pd.qcut(df["goodness"], 10, labels=False, duplicates="drop")
    for hlen in horizons:
        g = df.groupby("goodness_decile")[f"fwd_dd_{hlen}"].mean()
        v = df.groupby("goodness_decile")[f"fwd_vol_{hlen}"].mean()
        r = df.groupby("goodness_decile")[f"fwd_ret_{hlen}"].mean()
        for dec in g.index:
            quant_rows.append(
                {
                    "horizon": hlen,
                    "decile": int(dec),
                    "avg_drawdown": float(g.loc[dec]),
                    "avg_vol": float(v.loc[dec]),
                    "avg_return": float(r.loc[dec]),
                }
            )
    quant_df = pd.DataFrame(quant_rows)
    out_q = Path(args.out_quantiles)
    out_q.parent.mkdir(parents=True, exist_ok=True)
    quant_df.to_csv(out_q, index=False)

    # Regime/OOD summary: does low goodness anomaly predict high forward volatility?
    regime_rows = []
    regime_specs = [
        ("all", None, None),
        ("crisis_2008", pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-30")),
        ("covid_2020", pd.Timestamp("2020-02-15"), pd.Timestamp("2020-05-31")),
        ("regime_2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31")),
    ]
    for name, start, end in regime_specs:
        if start is None:
            sub = df
        else:
            sub = df[(df["date"] >= start) & (df["date"] <= end)]
        if sub.empty:
            continue
        for hlen in horizons:
            vol_col = f"fwd_vol_{hlen}"
            if vol_col not in sub.columns:
                continue
            vol_q = float(df[vol_col].quantile(0.8))
            y = (sub[vol_col].to_numpy(dtype=float) >= vol_q).astype(int)
            scores = sub["goodness_anomaly"].to_numpy(dtype=float)
            auroc = binary_auroc(scores, y) if y.min() != y.max() else float("nan")
            regime_rows.append(
                {
                    "regime": name,
                    "horizon": int(hlen),
                    "num_points": int(sub.shape[0]),
                    "high_vol_threshold_q80": vol_q,
                    "mean_goodness": float(sub["goodness"].mean()),
                    "mean_goodness_anomaly": float(sub["goodness_anomaly"].mean()),
                    "mean_fwd_vol": float(sub[vol_col].mean()),
                    "ood_auroc_low_goodness_vs_high_vol": float(auroc),
                }
            )
    events_df = pd.DataFrame(regime_rows)
    out_events = Path(args.out_events)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(out_events, index=False)

    # Economic relevance: simple risk-on/off strategy driven by goodness.
    signal_window = max(20, int(args.signal_window))
    signal_q = float(np.clip(args.signal_quantile, 0.05, 0.95))
    roll_q = df["goodness"].rolling(
        signal_window,
        min_periods=max(10, signal_window // 3),
    ).quantile(signal_q)
    signal = (df["goodness"] >= roll_q).astype(float)
    signal = signal.fillna(1.0)
    bench_ret_1 = df["fwd_ret_1"].to_numpy(dtype=float)
    strat_ret_1 = signal.to_numpy(dtype=float) * bench_ret_1
    cost = max(0.0, float(args.turnover_cost_bps)) * 1e-4
    if cost > 0:
        turnover = signal.diff().abs().fillna(0.0).to_numpy(dtype=float)
        strat_ret_1 = strat_ret_1 - cost * turnover

    strat_rows = [
        strategy_stats("benchmark_buy_and_hold", bench_ret_1),
        strategy_stats("goodness_risk_on_off", strat_ret_1),
    ]
    strat_df = pd.DataFrame(strat_rows)
    strat_df["signal_window"] = signal_window
    strat_df["signal_quantile"] = signal_q
    strat_df["turnover_cost_bps"] = float(args.turnover_cost_bps)
    out_strategy = Path(args.out_strategy)
    out_strategy.parent.mkdir(parents=True, exist_ok=True)
    strat_df.to_csv(out_strategy, index=False)

    # Scatter plot: goodness vs forward vol (one plot per horizon)
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4))
    if len(horizons) == 1:
        axes = [axes]
    for ax, hlen in zip(axes, horizons):
        ax.scatter(df["goodness"], df[f"fwd_vol_{hlen}"], alpha=0.5)
        ax.set_title(f"Horizon {hlen}d")
        ax.set_xlabel("Goodness")
        ax.set_ylabel("Forward Volatility")
    fig.tight_layout()
    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    # Timeline plot: goodness + anomaly with highlighted crisis windows.
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax0, ax1 = axes
    ax0.plot(df["date"], df["goodness"], color="#1f77b4", linewidth=1.2)
    ax0.set_ylabel("Goodness")
    ax0.set_title("Goodness Timeline")
    ax1.plot(df["date"], df["goodness_anomaly"], color="#d62728", linewidth=1.1)
    ax1.set_ylabel("Low-Goodness Anomaly")
    ax1.set_xlabel("Date")

    highlights = [
        ("2008-09-01", "2009-06-30"),
        ("2020-02-15", "2020-05-31"),
        ("2022-01-01", "2022-10-31"),
    ]
    for start, end in highlights:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        ax0.axvspan(s, e, color="#999999", alpha=0.12)
        ax1.axvspan(s, e, color="#999999", alpha=0.12)
    fig.tight_layout()
    out_timeline = Path(args.out_timeline)
    out_timeline.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_timeline, dpi=150)
    plt.close(fig)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_q}")
    print(f"Wrote {out_plot}")
    print(f"Wrote {out_events}")
    print(f"Wrote {out_strategy}")
    print(f"Wrote {out_timeline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
