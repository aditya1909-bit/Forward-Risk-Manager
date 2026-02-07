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


def _sample_ticker_preview(tickers_list, max_items=20):
    if not tickers_list:
        return ""
    uniq = sorted({t for sub in tickers_list for t in sub})
    if not uniq:
        return ""
    preview = uniq[:max_items]
    suffix = "..." if len(uniq) > max_items else ""
    return ", ".join(preview) + suffix


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a scenario book of hallucinated windows.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--num-scenarios", type=int, default=10, help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--indices", default="", help="Comma-separated graph indices")
    parser.add_argument("--dates", default="", help="Comma-separated graph dates")
    parser.add_argument("--target-ticker", default="", help="Ticker to constrain (e.g., NVDA)")
    parser.add_argument(
        "--target-drop",
        type=float,
        default=0.0,
        help="Target cumulative return over window (e.g., -0.10 for -10%)",
    )
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=10.0,
        help="Penalty weight for the constraint",
    )
    parser.add_argument(
        "--hall-steps",
        type=int,
        default=None,
        help="Override hallucination steps",
    )
    parser.add_argument(
        "--hall-lr",
        type=float,
        default=None,
        help="Override hallucination learning rate",
    )
    parser.add_argument(
        "--hall-l2",
        type=float,
        default=None,
        help="Override hallucination L2 weight",
    )
    parser.add_argument(
        "--hall-corr",
        type=float,
        default=None,
        help="Override hallucination correlation weight",
    )
    parser.add_argument(
        "--hall-mean-weight",
        type=float,
        default=None,
        help="Override hallucination mean penalty weight",
    )
    parser.add_argument(
        "--hall-std-weight",
        type=float,
        default=None,
        help="Override hallucination std penalty weight",
    )
    parser.add_argument(
        "--hall-clamp-std",
        type=float,
        default=None,
        help="Override hallucination clamp std (set higher to allow larger moves)",
    )
    parser.add_argument(
        "--hall-node-fraction",
        type=float,
        default=None,
        help="Override hallucination node fraction",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Adapt hallucination hyperparameters until constraint hit rate is met",
    )
    parser.add_argument(
        "--target-hit-rate",
        type=float,
        default=0.6,
        help="Target hit rate for adaptive constraint tuning",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=0.01,
        help="Tolerance for constraint hit (absolute)",
    )
    parser.add_argument(
        "--max-adapt-steps",
        type=int,
        default=6,
        help="Maximum adaptive iterations",
    )
    parser.add_argument(
        "--adapt-constraint-mult",
        type=float,
        default=1.5,
        help="Multiplier for constraint weight when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-step-inc",
        type=int,
        default=2,
        help="Step increment when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-lr-mult",
        type=float,
        default=1.2,
        help="LR multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-l2-mult",
        type=float,
        default=0.8,
        help="L2 multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-mean-mult",
        type=float,
        default=0.8,
        help="Mean penalty multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-std-mult",
        type=float,
        default=0.8,
        help="Std penalty multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-corr-mult",
        type=float,
        default=0.8,
        help="Corr penalty multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-node-inc",
        type=float,
        default=0.1,
        help="Node fraction increment when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-clamp-inc",
        type=float,
        default=0.5,
        help="Clamp std increment when under-hitting",
    )
    parser.add_argument(
        "--adapt-max-constraint",
        type=float,
        default=200.0,
        help="Max constraint weight in adaptive mode",
    )
    parser.add_argument(
        "--adapt-max-steps",
        type=int,
        default=20,
        help="Max hallucination steps in adaptive mode",
    )
    parser.add_argument(
        "--adapt-max-lr",
        type=float,
        default=0.2,
        help="Max hallucination LR in adaptive mode",
    )
    parser.add_argument(
        "--adapt-max-clamp-std",
        type=float,
        default=8.0,
        help="Max clamp std in adaptive mode",
    )
    parser.add_argument(
        "--adapt-min-l2",
        type=float,
        default=0.005,
        help="Min hallucination L2 in adaptive mode",
    )
    parser.add_argument(
        "--adapt-min-mean",
        type=float,
        default=0.001,
        help="Min hallucination mean penalty in adaptive mode",
    )
    parser.add_argument(
        "--adapt-min-std",
        type=float,
        default=0.001,
        help="Min hallucination std penalty in adaptive mode",
    )
    parser.add_argument(
        "--adapt-min-corr",
        type=float,
        default=0.01,
        help="Min hallucination corr penalty in adaptive mode",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=["exact", "at_least"],
        default="at_least",
        help="Exact match or at-least constraint",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Limit tickers per scenario (0 = all)",
    )
    parser.add_argument(
        "--diag-out",
        default="",
        help="Optional CSV to record constraint diagnostics",
    )
    parser.add_argument(
        "--out",
        default="reports/scenario_book.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    if args.target_ticker:
        args.target_ticker = args.target_ticker.strip().upper()
    if args.adaptive and not args.target_ticker:
        raise ValueError("--adaptive requires --target-ticker")

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

    target_candidates = None
    if args.target_ticker:
        if not tickers_list:
            raise ValueError("Graph payload missing tickers metadata; rebuild graphs with build_graphs.py.")
        target_candidates = [
            i for i, tickers in enumerate(tickers_list) if args.target_ticker in tickers
        ]
        if not target_candidates:
            preview = _sample_ticker_preview(tickers_list)
            hint = f" Sample tickers: {preview}" if preview else ""
            raise ValueError(
                f"Target ticker {args.target_ticker} not found in any graph.{hint} "
                "If you expected it, rebuild graphs with its price data and include it at build time."
            )

    if indices or date_list:
        idxs = _select_indices(dates, args.num_scenarios, args.seed, indices, date_list)
        if target_candidates is not None:
            missing = [i for i in idxs if i not in target_candidates]
            if missing:
                missing_dates = [dates[i] if i < len(dates) else "n/a" for i in missing]
                raise ValueError(
                    "Target ticker missing for some selected graphs. "
                    f"Missing indices: {missing} | dates: {missing_dates}"
                )
    else:
        if target_candidates is not None:
            rng = random.Random(args.seed)
            idxs = list(target_candidates)
            rng.shuffle(idxs)
            if args.num_scenarios <= 0:
                raise ValueError("--num-scenarios must be >= 1")
            if len(idxs) < args.num_scenarios:
                print(
                    f"Requested {args.num_scenarios} scenarios but only {len(idxs)} "
                    f"graphs include {args.target_ticker}. Using {len(idxs)}."
                )
            else:
                idxs = idxs[: args.num_scenarios]
        else:
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
        steps=int(args.hall_steps)
        if args.hall_steps is not None
        else int(train_cfg.get("hallucinate_steps", 4)),
        lr=float(args.hall_lr)
        if args.hall_lr is not None
        else float(train_cfg.get("hallucinate_lr", 0.05)),
        l2_weight=float(args.hall_l2)
        if args.hall_l2 is not None
        else float(train_cfg.get("hallucinate_l2", 0.02)),
        mean_weight=float(args.hall_mean_weight)
        if args.hall_mean_weight is not None
        else float(train_cfg.get("hallucinate_mean", 0.01)),
        std_weight=float(args.hall_std_weight)
        if args.hall_std_weight is not None
        else float(train_cfg.get("hallucinate_std", 0.01)),
        corr_weight=float(args.hall_corr)
        if args.hall_corr is not None
        else float(train_cfg.get("hallucinate_corr", 0.2)),
        clamp_std=float(args.hall_clamp_std)
        if args.hall_clamp_std is not None
        else float(train_cfg.get("hallucinate_clamp_std", 3.0)),
        goodness_temp=float(train_cfg.get("goodness_temp", 1.0)),
        node_fraction=float(args.hall_node_fraction)
        if args.hall_node_fraction is not None
        else float(train_cfg.get("hallucinate_node_fraction", 1.0)),
        node_min=int(train_cfg.get("hallucinate_node_min", 1)),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    def _run_once(hcfg: HallucinationConfig, constraint_weight: float):
        scenario_rows = []
        diag_rows = []

        for scenario_id, idx in enumerate(idxs):
            data = graphs[idx]
            tickers = (
                tickers_list[idx]
                if idx < len(tickers_list)
                else [f"n{i}" for i in range(data.num_nodes)]
            )
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

            constraint_fn = None
            force_indices = None
            target_idx = None
            if args.target_ticker:
                if args.target_ticker not in tickers:
                    raise ValueError(f"Target ticker {args.target_ticker} not in graph.")
                target_idx = tickers.index(args.target_ticker)
                force_indices = [target_idx]
                if ret_mean is not None and ret_std is not None:
                    ret_mean_t = ret_mean.view(-1, 1)
                    ret_std_t = ret_std.view(-1, 1)
                else:
                    ret_mean_t = None
                    ret_std_t = None

                def _constraint(x_var, idx=target_idx):
                    if ret_mean_t is not None and ret_std_t is not None:
                        rets = x_var[idx, :returns_len] * ret_std_t[idx] + ret_mean_t[idx]
                    else:
                        rets = x_var[idx, :returns_len]
                    cum = torch.exp(rets.sum()) - 1.0
                    if args.constraint_mode == "at_least":
                        if args.target_drop < 0:
                            diff = torch.relu(cum - args.target_drop)
                        else:
                            diff = torch.relu(args.target_drop - cum)
                    else:
                        diff = cum - args.target_drop
                    return constraint_weight * diff.pow(2)

                constraint_fn = _constraint

            x_neg = hallucinate_negative(
                model,
                data.x,
                data.edge_index,
                getattr(data, "edge_attr", None),
                torch.zeros(data.num_nodes, dtype=torch.long),
                hcfg,
                edge_weight=getattr(data, "edge_weight", None),
                constraint_fn=constraint_fn,
                force_indices=force_indices,
            )

            if ret_mean is not None and ret_std is not None:
                neg_returns = x_neg[:, :returns_len] * ret_std + ret_mean
            else:
                neg_returns = x_neg[:, :returns_len]

            pos_returns = pos_returns.numpy()
            neg_returns = neg_returns.detach().numpy()

            if target_idx is not None:
                real_cum = float(np.exp(np.sum(pos_returns[target_idx])) - 1.0)
                hall_cum = float(np.exp(np.sum(neg_returns[target_idx])) - 1.0)
                diff = hall_cum - args.target_drop
                print(
                    f"scenario {scenario_id} {date} {args.target_ticker}: "
                    f"target={args.target_drop:.2%} real={real_cum:.2%} "
                    f"hall={hall_cum:.2%} diff={diff:.2%}"
                )
                diag_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "graph_index": idx,
                        "date": date,
                        "ticker": args.target_ticker,
                        "target_drop": args.target_drop,
                        "real_cum_return": real_cum,
                        "hall_cum_return": hall_cum,
                        "hall_minus_target": diff,
                    }
                )

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
                scenario_rows.append(
                    [scenario_id, idx, date, tickers[i], "real"] + list(pos_returns[i])
                )
                scenario_rows.append(
                    [scenario_id, idx, date, tickers[i], "halluc"] + list(neg_returns[i])
                )

        return scenario_rows, diag_rows

    constraint_weight = float(args.constraint_weight)
    max_steps = max(1, int(args.max_adapt_steps)) if args.adaptive else 1
    attempt = 0
    final_rows = []
    final_diag = []

    while attempt < max_steps:
        prev_state = (
            constraint_weight,
            hall_cfg.steps,
            hall_cfg.lr,
            hall_cfg.l2_weight,
            hall_cfg.mean_weight,
            hall_cfg.std_weight,
            hall_cfg.corr_weight,
            hall_cfg.node_fraction,
            hall_cfg.clamp_std,
        )
        attempt += 1
        print(
            "adaptive attempt "
            f"{attempt}/{max_steps} | "
            f"constraint_weight={constraint_weight:.3f} | "
            f"hall_steps={hall_cfg.steps} | "
            f"hall_lr={hall_cfg.lr:.4f} | "
            f"hall_l2={hall_cfg.l2_weight:.4f} | "
            f"hall_node_fraction={hall_cfg.node_fraction:.2f} | "
            f"hall_corr={hall_cfg.corr_weight:.3f} | "
            f"hall_mean={hall_cfg.mean_weight:.4f} | "
            f"hall_std={hall_cfg.std_weight:.4f} | "
            f"hall_clamp={hall_cfg.clamp_std:.2f}"
        )
        scenario_rows, diag_rows = _run_once(hall_cfg, constraint_weight)
        final_rows = scenario_rows
        final_diag = diag_rows

        if not args.adaptive or not diag_rows:
            break

        diffs = [row["hall_minus_target"] for row in diag_rows]
        hits = sum(1 for d in diffs if abs(d) <= args.target_tolerance)
        hit_rate = hits / len(diffs)
        mean_diff = sum(diffs) / len(diffs)
        med_diff = sorted(diffs)[len(diffs) // 2]
        print(
            "constraint summary: "
            f"hit_rate={hits}/{len(diffs)} ({hit_rate:.1%}) | "
            f"mean_diff={mean_diff:.4f} | median_diff={med_diff:.4f}"
        )
        if hit_rate >= args.target_hit_rate:
            print("Target hit rate reached.")
            break

        # Adaptive adjustments
        if mean_diff > args.target_tolerance:
            constraint_weight = min(
                constraint_weight * args.adapt_constraint_mult,
                args.adapt_max_constraint,
            )
            hall_cfg.steps = min(hall_cfg.steps + args.adapt_hall_step_inc, args.adapt_max_steps)
            hall_cfg.lr = min(hall_cfg.lr * args.adapt_hall_lr_mult, args.adapt_max_lr)
            hall_cfg.l2_weight = max(hall_cfg.l2_weight * args.adapt_hall_l2_mult, args.adapt_min_l2)
            hall_cfg.mean_weight = max(hall_cfg.mean_weight * args.adapt_hall_mean_mult, args.adapt_min_mean)
            hall_cfg.std_weight = max(hall_cfg.std_weight * args.adapt_hall_std_mult, args.adapt_min_std)
            hall_cfg.corr_weight = max(hall_cfg.corr_weight * args.adapt_hall_corr_mult, args.adapt_min_corr)
            hall_cfg.node_fraction = min(
                1.0, hall_cfg.node_fraction + args.adapt_hall_node_inc
            )
            hall_cfg.clamp_std = min(hall_cfg.clamp_std + args.adapt_hall_clamp_inc, args.adapt_max_clamp_std)
        elif mean_diff < -args.target_tolerance:
            constraint_weight = max(
                constraint_weight / args.adapt_constraint_mult, 1.0
            )
            hall_cfg.steps = max(1, hall_cfg.steps - args.adapt_hall_step_inc)
            hall_cfg.lr = max(hall_cfg.lr / args.adapt_hall_lr_mult, 0.001)
            hall_cfg.l2_weight = min(hall_cfg.l2_weight / args.adapt_hall_l2_mult, 0.2)
            hall_cfg.mean_weight = min(hall_cfg.mean_weight / args.adapt_hall_mean_mult, 0.2)
            hall_cfg.std_weight = min(hall_cfg.std_weight / args.adapt_hall_std_mult, 0.2)
            hall_cfg.corr_weight = min(hall_cfg.corr_weight / args.adapt_hall_corr_mult, 1.0)
            hall_cfg.node_fraction = max(0.1, hall_cfg.node_fraction - args.adapt_hall_node_inc)
            hall_cfg.clamp_std = max(1.0, hall_cfg.clamp_std - args.adapt_hall_clamp_inc)
        else:
            # mean diff close to target but hit rate low -> increase diversity slightly
            hall_cfg.steps = min(hall_cfg.steps + 1, args.adapt_max_steps)
            hall_cfg.lr = min(hall_cfg.lr * 1.05, args.adapt_max_lr)

        new_state = (
            constraint_weight,
            hall_cfg.steps,
            hall_cfg.lr,
            hall_cfg.l2_weight,
            hall_cfg.mean_weight,
            hall_cfg.std_weight,
            hall_cfg.corr_weight,
            hall_cfg.node_fraction,
            hall_cfg.clamp_std,
        )
        if new_state == prev_state and attempt < max_steps:
            print("adaptive tuning saturated at current caps; stopping early.")
            break

    # Write final scenario book
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["scenario_id", "graph_index", "date", "ticker", "series"] + [
            f"r{i}" for i in range(returns_len)
        ]
        w.writerow(header)
        for row in final_rows:
            w.writerow(row)

    if args.diag_out and final_diag:
        diag_path = Path(args.diag_out)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv

        with diag_path.open("w", newline="") as f:
            w = _csv.DictWriter(
                f,
                fieldnames=[
                    "scenario_id",
                    "graph_index",
                    "date",
                    "ticker",
                    "target_drop",
                    "real_cum_return",
                    "hall_cum_return",
                    "hall_minus_target",
                ],
            )
            w.writeheader()
            for row in final_diag:
                w.writerow(row)
        print(f"Wrote {diag_path}")
        diffs = [row["hall_minus_target"] for row in final_diag]
        if diffs:
            hits = sum(1 for d in diffs if abs(d) <= args.target_tolerance)
            hit_rate = hits / len(diffs)
            mean_diff = sum(diffs) / len(diffs)
            med_diff = sorted(diffs)[len(diffs) // 2]
            print(
                "constraint summary: "
                f"hit_rate={hits}/{len(diffs)} ({hit_rate:.1%}) | "
                f"mean_diff={mean_diff:.4f} | median_diff={med_diff:.4f}"
            )

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
