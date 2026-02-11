#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import json
from collections import defaultdict

import numpy as np


def _load_pairs(path: Path):
    data = {}
    with path.open() as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("CSV missing header.")
        if "t" in r.fieldnames:
            id_col = "t"
        elif "scenario_id" in r.fieldnames:
            id_col = "scenario_id"
        elif "graph_index" in r.fieldnames:
            id_col = "graph_index"
        else:
            raise ValueError("CSV must include one of: t, scenario_id, graph_index")

        ret_cols = [c for c in r.fieldnames if c.startswith("r")]
        ret_cols = sorted(ret_cols, key=lambda c: int(c[1:]))
        if not ret_cols:
            raise ValueError("CSV missing return columns (r0, r1, ...)")

        for row in r:
            ticker = row["ticker"]
            series = row["series"]
            t = int(row[id_col])
            vals = np.array([float(row[c]) for c in ret_cols], dtype=float)
            data[(ticker, t, series)] = vals

    pairs = defaultdict(list)
    for (ticker, t, series), vals in data.items():
        if series != "real":
            continue
        key_h = (ticker, t, "halluc")
        if key_h not in data:
            continue
        pairs[ticker].append((vals, data[key_h]))

    if not pairs:
        raise ValueError("No real/halluc pairs found.")
    return dict(pairs)


def _kl_js(real, hall, bins=60, eps=1e-8):
    hist_r, edges = np.histogram(real, bins=bins, density=True)
    hist_h, _ = np.histogram(hall, bins=edges, density=True)
    p = hist_r + eps
    q = hist_h + eps
    p /= p.sum()
    q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    m = 0.5 * (p + q)
    js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    return kl, float(js)


def _summarize(real: np.ndarray, hall: np.ndarray, bins: int = 60) -> dict:
    diffs = hall - real
    corr = float(np.corrcoef(real, hall)[0, 1]) if real.std() > 0 and hall.std() > 0 else 0.0
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs**2)))
    kl, js = _kl_js(real, hall, bins=bins)
    tail_real = float(np.quantile(np.abs(real), 0.99))
    tail_hall = float(np.quantile(np.abs(hall), 0.99))
    tail_ratio = float(tail_hall / tail_real) if tail_real > 0 else 0.0
    return {
        "corr_real_hall": corr,
        "mae": mae,
        "rmse": rmse,
        "kl_divergence": kl,
        "js_divergence": js,
        "tail_abs_p99_real": tail_real,
        "tail_abs_p99_hall": tail_hall,
        "tail_ratio_p99": tail_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate hallucinations vs real distributions.")
    parser.add_argument(
        "--csv",
        default="runs/experiments/manual/metrics/scenario_book.csv",
        help="Input CSV",
    )
    parser.add_argument(
        "--out",
        default="runs/experiments/manual/diagnostics/hallucination_calibration.json",
        help="Output JSON",
    )
    parser.add_argument(
        "--out-by-ticker",
        default="runs/experiments/manual/diagnostics/hallucination_calibration_by_ticker.csv",
        help="Optional per-ticker calibration CSV (empty to disable).",
    )
    parser.add_argument("--target-ticker", default="", help="Optional target ticker for focused diagnostics.")
    parser.add_argument("--bins", type=int, default=60)
    args = parser.parse_args()
    target_ticker = args.target_ticker.strip().upper() if args.target_ticker else ""

    pairs = _load_pairs(Path(args.csv))
    ticker_rows = []
    real_all = []
    hall_all = []
    target_metrics = None

    for ticker, ticker_pairs in pairs.items():
        real_vals = np.concatenate([rv for rv, _ in ticker_pairs])
        hall_vals = np.concatenate([hv for _, hv in ticker_pairs])
        real_all.append(real_vals)
        hall_all.append(hall_vals)
        summary = _summarize(real_vals, hall_vals, bins=args.bins)
        row = {
            "ticker": ticker,
            "num_pairs": int(len(ticker_pairs)),
            "num_points": int(real_vals.size),
        }
        row.update(summary)
        ticker_rows.append(row)
        if target_ticker and ticker == target_ticker:
            target_metrics = row

    real = np.concatenate(real_all)
    hall = np.concatenate(hall_all)
    metrics = _summarize(real, hall, bins=args.bins)
    metrics["num_pairs"] = int(sum(len(v) for v in pairs.values()))
    metrics["num_points"] = int(real.size)
    metrics["num_tickers"] = int(len(pairs))
    if target_ticker:
        metrics["target_ticker"] = target_ticker
        metrics["target_metrics"] = target_metrics
    ticker_rows = sorted(ticker_rows, key=lambda r: float(r["mae"]), reverse=True)
    metrics["worst_tickers_by_mae"] = ticker_rows[:5]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")

    if args.out_by_ticker:
        out_t = Path(args.out_by_ticker)
        out_t.parent.mkdir(parents=True, exist_ok=True)
        with out_t.open("w", newline="") as f:
            fieldnames = [
                "ticker",
                "num_pairs",
                "num_points",
                "corr_real_hall",
                "mae",
                "rmse",
                "kl_divergence",
                "js_divergence",
                "tail_abs_p99_real",
                "tail_abs_p99_hall",
                "tail_ratio_p99",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in ticker_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"Wrote {out_t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
