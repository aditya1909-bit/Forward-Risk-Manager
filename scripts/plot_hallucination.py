#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import tomllib

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import GCNEncoder
from frisk.hallucinate import HallucinationConfig, hallucinate_negative


def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot real vs hallucinated return windows.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--graph-index", type=int, default=None, help="Graph index to plot")
    parser.add_argument("--date", default=None, help="Graph date (YYYY-MM-DD) to plot")
    parser.add_argument("--top-k", type=int, default=5, help="Number of tickers to plot")
    parser.add_argument("--out", default="reports/hallucination_plot.png", help="Output PNG")
    parser.add_argument(
        "--save-csv",
        default="reports/hallucination_window.csv",
        help="Output CSV of real vs hallucinated returns",
    )
    parser.add_argument(
        "--save-csv-all",
        default="",
        help="Output CSV of real vs hallucinated returns for all nodes",
    )
    parser.add_argument(
        "--list-dates",
        action="store_true",
        help="Print available graph dates and exit",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})
    build_cfg = cfg.get("build_graphs", {})

    graphs_path = Path(train_cfg.get("graphs", "data/processed/graphs.pt"))
    try:
        payload = torch.load(graphs_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(graphs_path, map_location="cpu")
    graphs = payload["graphs"]
    tickers_list = payload.get("tickers", [])
    dates = payload.get("dates", [])
    if not graphs:
        raise ValueError("No graphs found.")

    if args.list_dates:
        for d in dates:
            print(d)
        return 0

    if args.date and dates:
        try:
            idx = dates.index(args.date)
        except ValueError:
            raise ValueError(f"Date {args.date} not found in graphs metadata.")
    elif args.graph_index is None:
        idx = random.randint(0, len(graphs) - 1)
    else:
        idx = max(0, min(args.graph_index, len(graphs) - 1))

    data = graphs[idx]
    tickers = tickers_list[idx] if idx < len(tickers_list) else [f"n{i}" for i in range(data.num_nodes)]

    window = int(build_cfg.get("window", 20))
    feature_mode = build_cfg.get("feature_mode", "window")
    returns_len = window if feature_mode in ("window", "window_plus_summary") else 1

    x_pos = data.x.clone()
    ret_mean = getattr(data, "ret_mean", None)
    ret_std = getattr(data, "ret_std", None)

    if ret_mean is not None and ret_std is not None:
        ret_mean = ret_mean.view(-1, 1)
        ret_std = ret_std.view(-1, 1)
        pos_returns = x_pos[:, :returns_len] * ret_std + ret_mean
    else:
        pos_returns = x_pos[:, :returns_len]

    input_dim = data.x.shape[1]
    model = GCNEncoder(
        in_dim=input_dim,
        hidden_dim=int(train_cfg.get("hidden_dim", 64)),
        num_layers=int(train_cfg.get("num_layers", 2)),
        dropout=float(train_cfg.get("dropout", 0.1)),
    )
    model_path = train_cfg.get("save_model", "")
    if model_path and Path(model_path).exists():
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()

    hall_cfg = HallucinationConfig(
        steps=int(train_cfg.get("hallucinate_steps", 4)),
        lr=float(train_cfg.get("hallucinate_lr", 0.05)),
        l2_weight=float(train_cfg.get("hallucinate_l2", 0.02)),
        mean_weight=float(train_cfg.get("hallucinate_mean", 0.01)),
        std_weight=float(train_cfg.get("hallucinate_std", 0.01)),
        corr_weight=float(train_cfg.get("hallucinate_corr", 0.2)),
        clamp_std=float(train_cfg.get("hallucinate_clamp_std", 3.0)),
        goodness_temp=float(train_cfg.get("goodness_temp", 1.0)),
    )

    x_neg = hallucinate_negative(
        model,
        data.x,
        data.edge_index,
        getattr(data, "edge_attr", None),
        torch.zeros(data.num_nodes, dtype=torch.long),
        hall_cfg,
        edge_weight=getattr(data, "edge_weight", None),
    )

    if ret_mean is not None and ret_std is not None:
        neg_returns = x_neg[:, :returns_len] * ret_std + ret_mean
    else:
        neg_returns = x_neg[:, :returns_len]

    pos_returns = pos_returns.numpy()
    neg_returns = neg_returns.detach().numpy()

    pos_cum = np.exp(np.cumsum(pos_returns, axis=1))
    neg_cum = np.exp(np.cumsum(neg_returns, axis=1))

    diff = neg_cum[:, -1] - pos_cum[:, -1]
    ranked = np.argsort(diff)  # most negative first

    selected = []
    if "MDY" in tickers:
        selected.append(tickers.index("MDY"))
    for idx_i in ranked:
        if idx_i not in selected:
            selected.append(idx_i)
        if len(selected) >= args.top_k:
            break

    plt.figure(figsize=(9, 5))
    for i in selected:
        label = tickers[i]
        plt.plot(pos_cum[i], alpha=0.6, label=f"{label} real")
        plt.plot(neg_cum[i], alpha=0.6, linestyle="--", label=f"{label} halluc")

    plt.title(f"Graph {idx}: Real vs Hallucinated (window={returns_len})")
    plt.xlabel("Window step")
    plt.ylabel("Cumulative return (exp(cum log ret))")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")

    if args.save_csv:
        import csv

        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["ticker", "series", "t"] + [f"r{i}" for i in range(pos_returns.shape[1])]
            w.writerow(header)
            for i in selected:
                w.writerow([tickers[i], "real", idx] + list(pos_returns[i]))
                w.writerow([tickers[i], "halluc", idx] + list(neg_returns[i]))
        print(f"Wrote {csv_path}")

    if args.save_csv_all:
        import csv

        csv_path = Path(args.save_csv_all)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["ticker", "series", "t"] + [f"r{i}" for i in range(pos_returns.shape[1])]
            w.writerow(header)
            for i in range(len(tickers)):
                w.writerow([tickers[i], "real", idx] + list(pos_returns[i]))
                w.writerow([tickers[i], "halluc", idx] + list(neg_returns[i]))
        print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
