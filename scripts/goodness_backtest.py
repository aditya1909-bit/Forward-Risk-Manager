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


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def _max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(dd.min())


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest goodness vs future risk.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--prices", default="", help="Override prices.csv path")
    parser.add_argument("--ticker", default="MDY", help="Benchmark ticker (e.g., MDY)")
    parser.add_argument("--horizons", default="5,21", help="Comma-separated horizons (days)")
    parser.add_argument("--out-csv", default="reports/goodness_backtest.csv", help="Output CSV")
    parser.add_argument("--out-quantiles", default="reports/goodness_quantiles.csv", help="Quantile CSV")
    parser.add_argument("--out-plot", default="reports/goodness_scatter.png", help="Scatter plot")
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
    prices = pd.read_csv(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["ticker"] == args.ticker].sort_values("date")
    if prices.empty:
        raise ValueError(f"Ticker {args.ticker} not found in prices.")

    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px = prices.set_index("date")[price_col].astype(float)
    logret = np.log(px).diff().dropna()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    records = []

    for i, d in enumerate(dates):
        dt = pd.to_datetime(d)
        if dt not in logret.index:
            continue
        idx = logret.index.get_loc(dt)

        data = graphs[i]
        h = model(data.x, data.edge_index, edge_weight=getattr(data, "edge_weight", None))
        g = goodness(h, torch.zeros(data.num_nodes, dtype=torch.long), temperature=float(train_cfg.get("goodness_temp", 1.0))).mean().item()

        rec = {"date": dt, "goodness": g}
        for hlen in horizons:
            if idx + hlen >= len(logret):
                rec[f"fwd_vol_{hlen}"] = np.nan
                rec[f"fwd_dd_{hlen}"] = np.nan
                rec[f"fwd_ret_{hlen}"] = np.nan
                continue
            window = logret.iloc[idx + 1 : idx + 1 + hlen].values
            vol = float(np.std(window))
            cum = np.exp(np.cumsum(window))
            dd = _max_drawdown(cum)
            ret = float(cum[-1] - 1.0)
            rec[f"fwd_vol_{hlen}"] = vol
            rec[f"fwd_dd_{hlen}"] = dd
            rec[f"fwd_ret_{hlen}"] = ret
        records.append(rec)

    df = pd.DataFrame(records).dropna()
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

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_q}")
    print(f"Wrote {out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
