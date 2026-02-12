#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _to_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return default


def _load_rows(path: Path):
    with path.open() as f:
        r = csv.DictReader(f)
        return list(r)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run anti-triviality sanity checks on benchmark outputs.")
    parser.add_argument(
        "--benchmark-csv",
        default="runs/experiments/manual/metrics/benchmark.csv",
        help="Benchmark CSV path",
    )
    parser.add_argument(
        "--easy-neg-acc-max",
        type=float,
        default=0.995,
        help="Max allowed eval_acc for easy negatives (shuffle/noise).",
    )
    parser.add_argument(
        "--timeflip-sep-min",
        type=float,
        default=0.05,
        help="Min required eval_time_flip_sep for at least one FF row.",
    )
    args = parser.parse_args()

    rows = _load_rows(Path(args.benchmark_csv))
    if not rows:
        raise ValueError("benchmark CSV is empty.")

    failed = False
    easy_modes = {"shuffle", "noise", "shuffle+noise"}
    easy_acc = []
    timeflip_sep = []
    for row in rows:
        eval_mode = str(row.get("eval_neg_mode_effective", "")).strip().lower()
        acc = _to_float(row.get("eval_acc"))
        if eval_mode in easy_modes:
            easy_acc.append(acc)
        tf_sep = _to_float(row.get("eval_time_flip_sep"))
        if tf_sep == tf_sep:
            timeflip_sep.append(tf_sep)

    if easy_acc:
        easy_max = max(a for a in easy_acc if a == a)
        if easy_max > args.easy_neg_acc_max:
            print(
                f"FAIL easy-negative acc too high: max={easy_max:.4f} > {args.easy_neg_acc_max:.4f}"
            )
            failed = True
        else:
            print(
                f"PASS easy-negative acc: max={easy_max:.4f} <= {args.easy_neg_acc_max:.4f}"
            )
    else:
        print("WARN no easy-negative eval rows found in benchmark CSV.")

    if timeflip_sep:
        tf_best = max(timeflip_sep)
        if tf_best < args.timeflip_sep_min:
            print(
                f"FAIL time-flip separation too low: best={tf_best:.4f} < {args.timeflip_sep_min:.4f}"
            )
            failed = True
        else:
            print(
                f"PASS time-flip separation: best={tf_best:.4f} >= {args.timeflip_sep_min:.4f}"
            )
    else:
        print("WARN no eval_time_flip_sep found; enable benchmark.eval_neg_modes to include time_flip.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
