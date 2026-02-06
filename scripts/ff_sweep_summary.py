#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


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


def _pareto(rows):
    pts = [(r["graphs_per_s"], r["eval_sep"], r) for r in rows]
    pts_sorted = sorted(pts, key=lambda p: p[0])
    frontier = []
    best_sep = -1e9
    for x, y, r in pts_sorted:
        if y > best_sep:
            frontier.append((x, y, r))
            best_sep = y
    return frontier


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FF sweep results.")
    parser.add_argument("--csv", default="reports/ff_sweep.csv", help="Sweep CSV")
    parser.add_argument("--out", default="reports/ff_sweep_summary.txt", help="Output summary")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _load_rows(path)
    if not rows:
        raise ValueError("No rows found in sweep CSV.")

    best = max(rows, key=lambda r: r.get("eval_sep", float("-inf")))
    top = sorted(rows, key=lambda r: r.get("eval_sep", float("-inf")), reverse=True)[: args.top_k]
    frontier = _pareto(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        f.write(f"FF Sweep Summary ({path})\n")
        f.write(f"Rows: {len(rows)}\n\n")
        f.write("Best by eval_sep:\n")
        f.write(f"{best}\n\n")

        f.write(f"Top {args.top_k} by eval_sep:\n")
        for r in top:
            f.write(f"{r}\n")
        f.write("\nPareto frontier (maximize eval_sep and graphs_per_s):\n")
        for x, y, r in frontier:
            line = (
                f"graphs_per_s={x:.4f}, eval_sep={y:.6f}, mode={r.get('mode')}, params="
                f"temp={r.get('goodness_temp')}, target={r.get('goodness_target')}, "
                f"neg_mix_end={r.get('neg_mix_end')}, hall_steps={r.get('hall_steps')}, "
                f"hall_lr={r.get('hall_lr')}, hall_node_fraction={r.get('hall_node_fraction')}"
            )
            f.write(line + "\n")

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
