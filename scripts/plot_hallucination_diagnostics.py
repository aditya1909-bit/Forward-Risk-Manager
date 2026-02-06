#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


def _load_pairs(path: Path, window: int = 20):
    data = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            ticker = row["ticker"]
            series = row["series"]
            t = int(row["t"])
            vals = np.array([float(row[f"r{i}"]) for i in range(window)])
            data[(ticker, t, series)] = vals

    real_vals = []
    hall_vals = []
    diffs = []
    for (ticker, t, series), vals in data.items():
        if series != "real":
            continue
        key_h = (ticker, t, "halluc")
        if key_h not in data:
            continue
        v_real = vals
        v_hall = data[key_h]
        real_vals.append(v_real)
        hall_vals.append(v_hall)
        diffs.append(v_hall - v_real)

    if not real_vals:
        return None, None, None

    return np.concatenate(real_vals), np.concatenate(hall_vals), np.concatenate(diffs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot hallucination diagnostics.")
    parser.add_argument("--csv", default="reports/hallucination_window_all.csv", help="Input CSV")
    parser.add_argument(
        "--out",
        default="reports/hallucination_diagnostics.png",
        help="Output plot path",
    )
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(path)
    real_vals, hall_vals, diffs = _load_pairs(path)
    if real_vals is None:
        raise ValueError("No real/hallucination pairs found.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Overlay distributions
    ax = axes[0]
    ax.hist(real_vals, bins=args.bins, alpha=0.6, density=True, label="real")
    ax.hist(hall_vals, bins=args.bins, alpha=0.6, density=True, label="halluc")
    ax.set_title("Return Distribution")
    ax.set_xlabel("return")
    ax.set_ylabel("density")
    ax.legend()

    # Diff histogram
    ax = axes[1]
    ax.hist(diffs, bins=args.bins, alpha=0.7, color="#F58518", density=True)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Halluc - Real")
    ax.set_xlabel("diff")
    ax.set_ylabel("density")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
