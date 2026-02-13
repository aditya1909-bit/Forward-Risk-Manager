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


def _passes_min(value: float, threshold: float, atol: float) -> bool:
    return value >= (threshold - max(0.0, float(atol)))


def _is_critic_objective(objective: str) -> bool:
    obj = str(objective).strip().lower()
    return obj in {"ff", "forward_forward", "forward-forward"} or obj.startswith("ff_")


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
        help="Min required eval_time_flip_sep for at least one critic/FF row.",
    )
    parser.add_argument(
        "--sc-timeflip-sep-min",
        type=float,
        default=0.0,
        help="Legacy: min required eval_time_flip_sep for self_contrastive (only used with --enforce-sc-timeflip-checks).",
    )
    parser.add_argument(
        "--sc-timeflip-auroc-min",
        type=float,
        default=0.5,
        help="Legacy: min required eval_time_flip_auroc for self_contrastive (only used with --enforce-sc-timeflip-checks).",
    )
    parser.add_argument(
        "--sc-gap-min",
        type=float,
        default=float("nan"),
        help="Optional min required eval_sc_gap for at least one self_contrastive row.",
    )
    parser.add_argument(
        "--skip-sc-timeflip-checks",
        action="store_true",
        help="Deprecated compatibility flag; self_contrastive time-flip checks are disabled by default.",
    )
    parser.add_argument(
        "--enforce-sc-timeflip-checks",
        action="store_true",
        help="Legacy opt-in: re-enable self_contrastive time-flip separation/AUROC checks.",
    )
    parser.add_argument(
        "--sep-atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for separation min-threshold comparisons.",
    )
    parser.add_argument(
        "--auroc-atol",
        type=float,
        default=5e-3,
        help="Absolute tolerance for AUROC min-threshold comparisons.",
    )
    args = parser.parse_args()

    rows = _load_rows(Path(args.benchmark_csv))
    if not rows:
        raise ValueError("benchmark CSV is empty.")

    failed = False
    easy_modes = {"shuffle", "noise", "shuffle+noise"}
    easy_acc = []
    critic_timeflip_sep = []
    sc_gap = []
    sc_timeflip_sep = []
    sc_timeflip_auroc = []
    for row in rows:
        eval_mode = str(row.get("eval_neg_mode_effective", "")).strip().lower()
        objective = str(row.get("eval_objective", "")).strip().lower()
        acc = _to_float(row.get("eval_acc"))
        if eval_mode in easy_modes:
            easy_acc.append(acc)
        tf_sep = _to_float(row.get("eval_time_flip_sep"))
        if tf_sep == tf_sep and _is_critic_objective(objective):
            critic_timeflip_sep.append(tf_sep)
        if objective == "self_contrastive":
            if tf_sep == tf_sep:
                sc_timeflip_sep.append(tf_sep)
            gap = _to_float(row.get("eval_sc_gap"))
            if gap == gap:
                sc_gap.append(gap)
            tf_auroc = _to_float(row.get("eval_time_flip_auroc"))
            if tf_auroc == tf_auroc:
                sc_timeflip_auroc.append(tf_auroc)

    if easy_acc:
        easy_finite = [a for a in easy_acc if a == a]
        if not easy_finite:
            print("WARN easy-negative eval rows found, but eval_acc values are all non-finite.")
        else:
            easy_max = max(easy_finite)
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
        print(
            "WARN no easy-negative eval rows found in benchmark CSV "
            "(non-blocking; add a shuffle/noise eval mode if you want this check)."
        )

    if critic_timeflip_sep:
        tf_best = max(critic_timeflip_sep)
        if not _passes_min(tf_best, args.timeflip_sep_min, args.sep_atol):
            print(
                "FAIL critic time-flip separation too low: "
                f"best={tf_best:.6f} < {args.timeflip_sep_min:.6f} "
                f"(atol={args.sep_atol:.6g})"
            )
            failed = True
        else:
            delta = tf_best - args.timeflip_sep_min
            if delta < 0:
                print(
                    "PASS critic time-flip separation within tolerance: "
                    f"best={tf_best:.6f}, min={args.timeflip_sep_min:.6f}, "
                    f"delta={delta:.6g}, atol={args.sep_atol:.6g}"
                )
            else:
                print(
                    "PASS critic time-flip separation: "
                    f"best={tf_best:.4f} >= {args.timeflip_sep_min:.4f}"
                )
    else:
        print(
            "WARN no critic eval_time_flip_sep found; ensure at least one FF/critic row "
            "is evaluated with time-flip negatives."
        )

    sc_gap_min = _to_float(args.sc_gap_min)
    if sc_gap_min == sc_gap_min:
        if sc_gap:
            sc_best_gap = max(sc_gap)
            if not _passes_min(sc_best_gap, sc_gap_min, args.sep_atol):
                print(
                    "FAIL self_contrastive gap too low: "
                    f"best={sc_best_gap:.6f} < {sc_gap_min:.6f} "
                    f"(atol={args.sep_atol:.6g})"
                )
                failed = True
            else:
                delta = sc_best_gap - sc_gap_min
                if delta < 0:
                    print(
                        "PASS self_contrastive gap within tolerance: "
                        f"best={sc_best_gap:.6f}, min={sc_gap_min:.6f}, "
                        f"delta={delta:.6g}, atol={args.sep_atol:.6g}"
                    )
                else:
                    print(
                        "PASS self_contrastive gap: "
                        f"best={sc_best_gap:.4f} >= {sc_gap_min:.4f}"
                    )
        else:
            print("WARN no self_contrastive eval_sc_gap found.")

    if not args.enforce_sc_timeflip_checks:
        if args.skip_sc_timeflip_checks:
            print(
                "INFO --skip-sc-timeflip-checks is deprecated; self_contrastive time-flip checks "
                "are already disabled by default."
            )
        else:
            print(
                "INFO self_contrastive time-flip checks disabled by default "
                "(critic is the arrow-of-time gate)."
            )
        return 1 if failed else 0

    if args.skip_sc_timeflip_checks:
        print("INFO skipping self_contrastive time-flip separation/AUROC checks.")
        return 1 if failed else 0

    if sc_timeflip_sep:
        sc_best_sep = max(sc_timeflip_sep)
        if not _passes_min(sc_best_sep, args.sc_timeflip_sep_min, args.sep_atol):
            print(
                "FAIL self_contrastive time-flip separation too low: "
                f"best={sc_best_sep:.6f} < {args.sc_timeflip_sep_min:.6f} "
                f"(atol={args.sep_atol:.6g})"
            )
            failed = True
        else:
            delta = sc_best_sep - args.sc_timeflip_sep_min
            if delta < 0:
                print(
                    "PASS self_contrastive time-flip separation within tolerance: "
                    f"best={sc_best_sep:.6f}, min={args.sc_timeflip_sep_min:.6f}, "
                    f"delta={delta:.6g}, atol={args.sep_atol:.6g}"
                )
            else:
                print(
                    "PASS self_contrastive time-flip separation: "
                    f"best={sc_best_sep:.4f} >= {args.sc_timeflip_sep_min:.4f}"
                )
    else:
        print("WARN no self_contrastive eval_time_flip_sep found.")

    if sc_timeflip_auroc:
        sc_best_auroc = max(sc_timeflip_auroc)
        if not _passes_min(sc_best_auroc, args.sc_timeflip_auroc_min, args.auroc_atol):
            print(
                "FAIL self_contrastive time-flip AUROC too low: "
                f"best={sc_best_auroc:.6f} < {args.sc_timeflip_auroc_min:.6f} "
                f"(atol={args.auroc_atol:.6g})"
            )
            failed = True
        else:
            delta = sc_best_auroc - args.sc_timeflip_auroc_min
            if delta < 0:
                print(
                    "PASS self_contrastive time-flip AUROC within tolerance: "
                    f"best={sc_best_auroc:.6f}, min={args.sc_timeflip_auroc_min:.6f}, "
                    f"delta={delta:.6g}, atol={args.auroc_atol:.6g}"
                )
            else:
                print(
                    "PASS self_contrastive time-flip AUROC: "
                    f"best={sc_best_auroc:.4f} >= {args.sc_timeflip_auroc_min:.4f}"
                )
    else:
        print("WARN no self_contrastive eval_time_flip_auroc found.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
