#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import json

import numpy as np


def _load_pairs(path: Path):
    data = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            ticker = row["ticker"]
            series = row["series"]
            t = int(row["t"])
            vals = np.array([float(row[f"r{i}"]) for i in range(20)])
            data[(ticker, t, series)] = vals

    real_vals = []
    hall_vals = []
    for (ticker, t, series), vals in data.items():
        if series != "real":
            continue
        key_h = (ticker, t, "halluc")
        if key_h not in data:
            continue
        real_vals.append(vals)
        hall_vals.append(data[key_h])

    if not real_vals:
        raise ValueError("No real/halluc pairs found.")
    return np.concatenate(real_vals), np.concatenate(hall_vals)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate hallucinations vs real distributions.")
    parser.add_argument("--csv", default="reports/hallucination_window_all.csv", help="Input CSV")
    parser.add_argument("--out", default="reports/hallucination_calibration.json", help="Output JSON")
    parser.add_argument("--bins", type=int, default=60)
    args = parser.parse_args()

    real, hall = _load_pairs(Path(args.csv))
    diffs = hall - real

    corr = float(np.corrcoef(real, hall)[0, 1]) if real.std() > 0 and hall.std() > 0 else 0.0
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs**2)))
    kl, js = _kl_js(real, hall, bins=args.bins)

    # tail ratio: 99th percentile absolute return
    tail_real = float(np.quantile(np.abs(real), 0.99))
    tail_hall = float(np.quantile(np.abs(hall), 0.99))
    tail_ratio = float(tail_hall / tail_real) if tail_real > 0 else 0.0

    metrics = {
        "corr_real_hall": corr,
        "mae": mae,
        "rmse": rmse,
        "kl_divergence": kl,
        "js_divergence": js,
        "tail_abs_p99_real": tail_real,
        "tail_abs_p99_hall": tail_hall,
        "tail_ratio_p99": tail_ratio,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
