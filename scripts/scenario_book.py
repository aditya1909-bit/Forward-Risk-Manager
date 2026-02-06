#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import GCNEncoder
from frisk.hallucinate import HallucinationConfig, hallucinate_negative


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def _select_indices(dates, num_scenarios, seed, indices, date_list):
    if indices:
        return indices
    if date_list:
        idxs = []
        for d in date_list:
            if d in dates:
                idxs.append(dates.index(d))
            else:
                raise ValueError(f"Date {d} not found in graphs metadata.")
        return idxs
    rng = random.Random(seed)
    idxs = list(range(len(dates)))
    rng.shuffle(idxs)
    return idxs[:num_scenarios]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a scenario book of hallucinated windows.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--num-scenarios", type=int, default=10, help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--indices", default="", help="Comma-separated graph indices")
    parser.add_argument("--dates", default="", help="Comma-separated graph dates")
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Limit tickers per scenario (0 = all)",
    )
    parser.add_argument(
        "--out",
        default="reports/scenario_book.csv",
        help="Output CSV path",
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

    indices = [int(x) for x in args.indices.split(",") if x.strip()] if args.indices else []
    date_list = [d.strip() for d in args.dates.split(",") if d.strip()] if args.dates else []
    idxs = _select_indices(dates, args.num_scenarios, args.seed, indices, date_list)

    window = int(build_cfg.get("window", 20))
    feature_mode = build_cfg.get("feature_mode", "window")
    returns_len = window if feature_mode in ("window", "window_plus_summary") else 1

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
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
        node_fraction=float(train_cfg.get("hallucinate_node_fraction", 1.0)),
        node_min=int(train_cfg.get("hallucinate_node_min", 1)),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["scenario_id", "graph_index", "date", "ticker", "series"] + [
            f"r{i}" for i in range(returns_len)
        ]
        w.writerow(header)

        for scenario_id, idx in enumerate(idxs):
            data = graphs[idx]
            tickers = tickers_list[idx] if idx < len(tickers_list) else [f"n{i}" for i in range(data.num_nodes)]
            date = dates[idx] if idx < len(dates) else ""

            ret_mean = getattr(data, "ret_mean", None)
            ret_std = getattr(data, "ret_std", None)

            x_pos = data.x.clone()
            if ret_mean is not None and ret_std is not None:
                ret_mean = ret_mean.view(-1, 1)
                ret_std = ret_std.view(-1, 1)
                pos_returns = x_pos[:, :returns_len] * ret_std + ret_mean
            else:
                pos_returns = x_pos[:, :returns_len]

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

            selected = list(range(len(tickers)))
            if args.max_tickers and args.max_tickers > 0:
                pos_cum = np.exp(np.cumsum(pos_returns, axis=1))
                neg_cum = np.exp(np.cumsum(neg_returns, axis=1))
                diff = neg_cum[:, -1] - pos_cum[:, -1]
                ranked = np.argsort(diff)
                selected = ranked[: args.max_tickers].tolist()
                if "MDY" in tickers and tickers.index("MDY") not in selected:
                    selected.insert(0, tickers.index("MDY"))

            for i in selected:
                w.writerow([scenario_id, idx, date, tickers[i], "real"] + list(pos_returns[i]))
                w.writerow([scenario_id, idx, date, tickers[i], "halluc"] + list(neg_returns[i]))

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
