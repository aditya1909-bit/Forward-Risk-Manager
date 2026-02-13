#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize which return steps/tickers move most between real and hallucinated scenarios."
    )
    parser.add_argument(
        "--csv",
        default="runs/experiments/default/metrics/scenario_book.csv",
        help="Scenario book CSV from scripts/scenario_book.py",
    )
    parser.add_argument(
        "--out",
        default="runs/experiments/default/diagnostics/scenario_delta_report.csv",
        help="Output ticker delta report",
    )
    args = parser.parse_args()

    path = Path(args.csv)
    rows: dict[tuple[str, int], dict[str, np.ndarray]] = defaultdict(dict)
    with path.open() as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("CSV missing header.")
        step_cols = sorted([c for c in r.fieldnames if c.startswith("r")], key=lambda c: int(c[1:]))
        if not step_cols:
            raise ValueError("CSV missing return columns (r0, r1, ...).")
        for row in r:
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            sid = int(row.get("scenario_id", 0))
            series = str(row.get("series", "")).strip().lower()
            if series not in {"real", "halluc"}:
                continue
            vals = np.asarray([float(row[c]) for c in step_cols], dtype=float)
            rows[(ticker, sid)][series] = vals

    agg: dict[str, list[np.ndarray]] = defaultdict(list)
    for (ticker, _sid), pair in rows.items():
        if "real" not in pair or "halluc" not in pair:
            continue
        agg[ticker].append(pair["halluc"] - pair["real"])

    report_rows = []
    for ticker, deltas in agg.items():
        d = np.vstack(deltas)
        mean_abs_by_step = np.mean(np.abs(d), axis=0)
        cum_real_hall = np.exp(np.sum(d, axis=1)) - 1.0
        row = {
            "ticker": ticker,
            "num_pairs": int(d.shape[0]),
            "mean_abs_delta": float(np.mean(np.abs(d))),
            "p90_abs_delta": float(np.quantile(np.abs(d), 0.9)),
            "mean_cum_return_delta": float(np.mean(cum_real_hall)),
            "worst_step": int(np.argmax(mean_abs_by_step)),
            "worst_step_mean_abs_delta": float(np.max(mean_abs_by_step)),
        }
        report_rows.append(row)

    report_rows.sort(key=lambda r: r["mean_abs_delta"], reverse=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ticker",
                "num_pairs",
                "mean_abs_delta",
                "p90_abs_delta",
                "mean_cum_return_delta",
                "worst_step",
                "worst_step_mean_abs_delta",
            ],
        )
        w.writeheader()
        for row in report_rows:
            w.writerow(row)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
