#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv

import matplotlib.pyplot as plt


def _load_rows(path: Path):
    rows = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                if k in ("mode", "neg_mode"):
                    continue
                try:
                    row[k] = float(v)
                except Exception:
                    pass
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot FF sweep tradeoffs.")
    parser.add_argument("--csv", default="reports/ff_sweep.csv", help="Path to sweep CSV")
    parser.add_argument("--out", default="reports/ff_sweep_tradeoff.png", help="Output plot path")
    parser.add_argument(
        "--pareto-out",
        default="reports/ff_sweep_pareto.png",
        help="Output path for Pareto frontier plot",
    )
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _load_rows(path)
    if not rows:
        raise ValueError("No rows found in sweep CSV.")

    modes = sorted(set(r["mode"] for r in rows))
    colors = {"ff_layerwise": "#4C78A8", "ff_e2e": "#F58518"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in modes:
        xs = [r["graphs_per_s"] for r in rows if r["mode"] == mode]
        ys = [r["eval_sep"] for r in rows if r["mode"] == mode]
        ax.scatter(xs, ys, label=mode, alpha=0.8, color=colors.get(mode, None))

    ax.set_xlabel("graphs/sec")
    ax.set_ylabel("eval_sep (g_pos - g_neg)")
    ax.set_title("FF Sweep Tradeoff")
    ax.legend()
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

    # Pareto frontier (maximize eval_sep and graphs_per_s)
    pts = [(r["graphs_per_s"], r["eval_sep"], r["mode"]) for r in rows]
    # Sort by speed
    pts_sorted = sorted(pts, key=lambda p: p[0])
    frontier = []
    best_sep = -1e9
    for x, y, m in pts_sorted:
        if y > best_sep:
            frontier.append((x, y, m))
            best_sep = y

    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in modes:
        xs = [p[0] for p in pts if p[2] == mode]
        ys = [p[1] for p in pts if p[2] == mode]
        ax.scatter(xs, ys, label=mode, alpha=0.25, color=colors.get(mode, None))

    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax.plot(fx, fy, color="#54A24B", linewidth=2, label="Pareto frontier")
    ax.scatter(fx, fy, color="#54A24B", s=30)

    ax.set_xlabel("graphs/sec")
    ax.set_ylabel("eval_sep (g_pos - g_neg)")
    ax.set_title("FF Sweep Pareto Frontier")
    ax.legend()
    fig.tight_layout()
    pout = Path(args.pareto_out)
    pout.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pout, dpi=150)
    plt.close(fig)
    print(f"Wrote {pout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
