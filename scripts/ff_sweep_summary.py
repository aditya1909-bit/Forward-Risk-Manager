#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _to_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:  # NaN guard
        return None
    return out


def _load_rows(path: Path):
    rows = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                if k in ("mode", "neg_mode", "rank_metric", "eval_objective"):
                    continue
                try:
                    row[k] = float(v)
                except Exception:
                    pass
            rows.append(row)
    return rows


def _row_rank_metric_value(row):
    rank_value = _to_float(row.get("rank_value"))
    if rank_value is not None:
        metric = str(row.get("rank_metric", "rank_value")).strip() or "rank_value"
        return metric, rank_value

    objective = str(row.get("eval_objective", "ff")).strip().lower()
    if objective == "self_contrastive":
        sc_gap = _to_float(row.get("eval_sc_gap"))
        if sc_gap is not None:
            return "eval_sc_gap", sc_gap

    sep = _to_float(row.get("eval_sep"))
    if sep is not None:
        return "eval_sep", sep

    acc = _to_float(row.get("eval_acc"))
    if acc is not None:
        return "eval_acc", acc
    return "eval_sep", float("-inf")


def _pareto(rows):
    pts = [(r.get("graphs_per_s", 0.0), r.get("_rank_value", float("-inf")), r) for r in rows]
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
    parser.add_argument(
        "--csv",
        default="runs/experiments/manual/metrics/ff_sweep.csv",
        help="Sweep CSV",
    )
    parser.add_argument(
        "--out",
        default="runs/experiments/manual/logs/ff_sweep_summary.txt",
        help="Output summary",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _load_rows(path)
    if not rows:
        raise ValueError("No rows found in sweep CSV.")

    for row in rows:
        metric, value = _row_rank_metric_value(row)
        row["_rank_metric"] = metric
        row["_rank_value"] = value

    best = max(rows, key=lambda r: r.get("_rank_value", float("-inf")))
    top = sorted(rows, key=lambda r: r.get("_rank_value", float("-inf")), reverse=True)[
        : args.top_k
    ]
    frontier = _pareto(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        f.write(f"FF Sweep Summary ({path})\n")
        f.write(f"Rows: {len(rows)}\n\n")
        f.write("Best by objective-aware rank:\n")
        f.write(f"{best}\n\n")

        f.write(f"Top {args.top_k} by objective-aware rank:\n")
        for r in top:
            f.write(f"{r}\n")
        f.write("\nPareto frontier (maximize rank_value and graphs_per_s):\n")
        for x, y, r in frontier:
            line = (
                f"graphs_per_s={x:.4f}, rank={r.get('_rank_metric')}:{y:.6f}, mode={r.get('mode')}, params="
                f"temp={r.get('goodness_temp')}, target={r.get('goodness_target')}, "
                f"neg_mix_end={r.get('neg_mix_end')}, hall_steps={r.get('hall_steps')}, "
                f"hall_lr={r.get('hall_lr')}, hall_node_fraction={r.get('hall_node_fraction')}"
            )
            f.write(line + "\n")

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
