#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


def _to_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _load_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _primary_metric(row):
    objective = str(row.get("eval_objective", "")).strip().lower()
    if objective == "self_contrastive":
        sc_gap = _to_float(row.get("eval_sc_gap"))
        if sc_gap is not None:
            return "eval_sc_gap", sc_gap

    sep = _to_float(row.get("eval_sep"))
    if sep is not None:
        return "eval_sep", sep

    auroc = _to_float(row.get("eval_auroc"))
    if auroc is not None:
        return "eval_auroc", auroc

    auprc = _to_float(row.get("eval_auprc"))
    if auprc is not None:
        return "eval_auprc", auprc

    sc_gap = _to_float(row.get("eval_sc_gap"))
    if sc_gap is not None:
        return "eval_sc_gap", sc_gap

    acc = _to_float(row.get("eval_acc"))
    if acc is not None:
        return "eval_acc", acc

    return "none", float("-inf")


def _minmax_norm(value, lo, hi):
    if value is None:
        return 0.0
    if hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _range(values: Iterable[float | None]):
    xs = [x for x in values if x is not None]
    if not xs:
        return 0.0, 0.0
    return min(xs), max(xs)


def _attach_derived(rows, source):
    out = []
    for row in rows:
        metric_name, metric_value = _primary_metric(row)
        copy = dict(row)
        copy["_source"] = source
        copy["_primary_metric"] = metric_name
        copy["_primary_value"] = metric_value
        copy["_eval_acc"] = _to_float(copy.get("eval_acc"))
        copy["_eval_auroc"] = _to_float(copy.get("eval_auroc"))
        copy["_eval_auprc"] = _to_float(copy.get("eval_auprc"))
        copy["_avg_epoch_s"] = _to_float(copy.get("avg_epoch_s"))
        copy["_graphs_per_s"] = _to_float(copy.get("graphs_per_s"))
        out.append(copy)
    return out


def _score_e2e(rows, primary_weight):
    lo_p, hi_p = _range(r.get("_primary_value") for r in rows)
    lo_a, hi_a = _range(r.get("_eval_auroc") for r in rows)
    use_acc_fallback = hi_a <= lo_a
    if use_acc_fallback:
        lo_a, hi_a = _range(r.get("_eval_acc") for r in rows)
    speed_weight = 1.0 - primary_weight
    scored = []
    for row in rows:
        p = _minmax_norm(row.get("_primary_value"), lo_p, hi_p)
        aux_val = row.get("_eval_acc") if use_acc_fallback else row.get("_eval_auroc")
        a = _minmax_norm(aux_val, lo_a, hi_a)
        row = dict(row)
        row["_dual_score"] = primary_weight * p + speed_weight * a
        scored.append(row)
    return scored


def _apply_e2e_backprop_penalty(rows, backprop_acc: float | None, min_ratio: float):
    if backprop_acc is None or backprop_acc <= 0 or min_ratio <= 0:
        out = []
        for row in rows:
            copy = dict(row)
            copy["_acc_ratio_vs_backprop"] = None
            copy["_penalty_factor"] = 1.0
            copy["_dual_score_raw"] = copy.get("_dual_score")
            out.append(copy)
        return out

    out = []
    for row in rows:
        copy = dict(row)
        acc = copy.get("_eval_acc")
        ratio = None
        penalty = 1.0
        if acc is not None:
            ratio = max(0.0, float(acc) / float(backprop_acc))
            if ratio < min_ratio:
                penalty = max(0.0, ratio / min_ratio)
        copy["_acc_ratio_vs_backprop"] = ratio
        copy["_penalty_factor"] = penalty
        copy["_dual_score_raw"] = copy.get("_dual_score")
        copy["_dual_score"] = float(copy.get("_dual_score", 0.0)) * penalty
        out.append(copy)
    return out


def _score_layerwise(rows, speed_weight):
    lo_s, hi_s = _range(r.get("_graphs_per_s") for r in rows)
    lo_p, hi_p = _range(r.get("_primary_value") for r in rows)
    quality_weight = 1.0 - speed_weight
    scored = []
    for row in rows:
        s = _minmax_norm(row.get("_graphs_per_s"), lo_s, hi_s)
        p = _minmax_norm(row.get("_primary_value"), lo_p, hi_p)
        row = dict(row)
        row["_dual_score"] = speed_weight * s + quality_weight * p
        scored.append(row)
    return scored


def _pick_best(rows):
    if not rows:
        return None
    return max(
        rows,
        key=lambda r: (
            r.get("_dual_score", float("-inf")),
            r.get("_primary_value", float("-inf")),
            r.get("_eval_acc", float("-inf")),
            r.get("_graphs_per_s", float("-inf")),
        ),
    )


def _fmt(x, nd=6):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _track_rows(sweep_rows, bench_rows, mode):
    sweep_mode = [r for r in sweep_rows if str(r.get("mode", "")).strip() == mode]
    if sweep_mode:
        return sweep_mode
    return [r for r in bench_rows if str(r.get("mode", "")).strip() == mode]


def _write_text_report(path: Path, rows, backprop):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("Dual Track FF Report\n")
        f.write("====================\n\n")
        if backprop is None:
            f.write("Reference backprop row: missing\n\n")
        else:
            f.write("Reference backprop row:\n")
            f.write(
                "  "
                f"primary({backprop['_primary_metric']})={_fmt(backprop.get('_primary_value'))}, "
                f"eval_acc={_fmt(backprop.get('_eval_acc'))}, "
                f"avg_epoch_s={_fmt(backprop.get('_avg_epoch_s'))}, "
                f"graphs_per_s={_fmt(backprop.get('_graphs_per_s'))}\n\n"
            )

        for row in rows:
            f.write(f"[{row['track']}] mode={row['mode']} source={row['source']}\n")
            f.write(
                "  "
            f"dual_score={_fmt(row.get('dual_score'))}, "
            f"dual_score_raw={_fmt(row.get('dual_score_raw'))}, "
            f"penalty_factor={_fmt(row.get('penalty_factor'))}, "
            f"primary({row['primary_metric']})={_fmt(row.get('primary_value'))}, "
            f"eval_acc={_fmt(row.get('eval_acc'))}, "
            f"eval_auroc={_fmt(row.get('eval_auroc'))}\n"
            )
            f.write(
                "  "
                f"avg_epoch_s={_fmt(row.get('avg_epoch_s'))}, "
                f"graphs_per_s={_fmt(row.get('graphs_per_s'))}, "
                f"speed_ratio_vs_backprop={_fmt(row.get('speed_ratio_vs_backprop'))}\n"
            )
            f.write(
                "  "
                f"delta_primary_vs_backprop={_fmt(row.get('delta_primary_vs_backprop'))}, "
                f"delta_acc_vs_backprop={_fmt(row.get('delta_acc_vs_backprop'))}\n"
            )
            if row.get("track") == "e2e_accuracy":
                f.write(
                    "  "
                    f"acc_ratio_vs_backprop={_fmt(row.get('acc_ratio_vs_backprop'))}\n"
                )
            params = []
            for k in (
                "goodness_target",
                "goodness_temp",
                "neg_mix_end",
                "hall_steps",
                "hall_lr",
                "hall_node_fraction",
                "distance_forward_weight",
                "self_contrastive_ff_weight",
            ):
                if k in row and str(row[k]).strip():
                    params.append(f"{k}={row[k]}")
            if params:
                f.write("  params: " + ", ".join(params) + "\n")
            f.write("\n")


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track",
        "mode",
        "source",
        "dual_score",
        "dual_score_raw",
        "penalty_factor",
        "primary_metric",
        "primary_value",
        "eval_acc",
        "eval_auroc",
        "eval_auprc",
        "acc_ratio_vs_backprop",
        "avg_epoch_s",
        "graphs_per_s",
        "speed_ratio_vs_backprop",
        "delta_primary_vs_backprop",
        "delta_acc_vs_backprop",
        "goodness_target",
        "goodness_temp",
        "neg_mix_end",
        "hall_steps",
        "hall_lr",
        "hall_node_fraction",
        "distance_forward_weight",
        "self_contrastive_ff_weight",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pick FF recommendations with split goals: e2e accuracy and layerwise speed."
    )
    parser.add_argument(
        "--benchmark",
        default="runs/experiments/manual/metrics/benchmark.csv",
        help="Benchmark CSV path",
    )
    parser.add_argument(
        "--sweep",
        default="runs/experiments/manual/metrics/ff_sweep.csv",
        help="Sweep CSV path",
    )
    parser.add_argument(
        "--sweep-e2e",
        default="",
        help="Optional e2e-focused sweep CSV path. If present, ff_e2e candidates are taken from here.",
    )
    parser.add_argument(
        "--out",
        default="runs/experiments/manual/logs/dual_score_report.txt",
        help="Text report path",
    )
    parser.add_argument(
        "--out-csv",
        default="runs/experiments/manual/metrics/dual_score_report.csv",
        help="Machine-readable report path",
    )
    parser.add_argument(
        "--e2e-primary-weight",
        type=float,
        default=0.8,
        help="Weight for objective-aware quality in e2e score (remaining weight goes to eval_acc).",
    )
    parser.add_argument(
        "--layer-speed-weight",
        type=float,
        default=0.8,
        help="Weight for graphs_per_s in layerwise score (remaining weight goes to objective-aware quality).",
    )
    parser.add_argument(
        "--e2e-min-acc-ratio-vs-backprop",
        type=float,
        default=0.5,
        help="If e2e eval_acc/backprop eval_acc is below this ratio, down-weight e2e dual score.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.e2e_primary_weight <= 1.0):
        raise ValueError("--e2e-primary-weight must be in [0, 1].")
    if not (0.0 <= args.layer_speed_weight <= 1.0):
        raise ValueError("--layer-speed-weight must be in [0, 1].")
    if args.e2e_min_acc_ratio_vs_backprop < 0.0:
        raise ValueError("--e2e-min-acc-ratio-vs-backprop must be >= 0.")

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)

    sweep_path = Path(args.sweep)
    sweep_e2e_path = Path(args.sweep_e2e) if str(args.sweep_e2e).strip() else None
    benchmark_rows = _attach_derived(_load_rows(benchmark_path), source="benchmark")
    sweep_rows = _attach_derived(_load_rows(sweep_path), source="sweep")
    sweep_e2e_rows = []
    if sweep_e2e_path is not None:
        if sweep_e2e_path.exists():
            sweep_e2e_rows = _attach_derived(_load_rows(sweep_e2e_path), source="sweep_e2e")
        else:
            print(
                f"warning: --sweep-e2e path not found ({sweep_e2e_path}); "
                "falling back to --sweep for e2e candidates."
            )

    if not benchmark_rows:
        raise ValueError("No rows found in benchmark CSV.")

    backprop = next((r for r in benchmark_rows if str(r.get("mode", "")).strip() == "backprop"), None)

    e2e_source_rows = sweep_e2e_rows if sweep_e2e_rows else sweep_rows
    e2e_candidates = _track_rows(e2e_source_rows, benchmark_rows, mode="ff_e2e")
    layer_candidates = _track_rows(sweep_rows, benchmark_rows, mode="ff_layerwise")
    if not e2e_candidates:
        raise ValueError("No ff_e2e rows found in benchmark or sweep CSV.")
    if not layer_candidates:
        raise ValueError("No ff_layerwise rows found in benchmark or sweep CSV.")

    backprop_acc = backprop.get("_eval_acc") if backprop is not None else None
    e2e_scored = _score_e2e(e2e_candidates, args.e2e_primary_weight)
    e2e_scored = _apply_e2e_backprop_penalty(
        e2e_scored,
        backprop_acc=backprop_acc,
        min_ratio=float(args.e2e_min_acc_ratio_vs_backprop),
    )
    best_e2e = _pick_best(e2e_scored)
    best_layer = _pick_best(_score_layerwise(layer_candidates, args.layer_speed_weight))

    out_rows = []
    for track, chosen in (
        ("e2e_accuracy", best_e2e),
        ("layerwise_speed", best_layer),
    ):
        row = {
            "track": track,
            "mode": chosen.get("mode"),
            "source": chosen.get("_source"),
            "dual_score": chosen.get("_dual_score"),
            "dual_score_raw": chosen.get("_dual_score_raw", chosen.get("_dual_score")),
            "penalty_factor": chosen.get("_penalty_factor", 1.0),
            "primary_metric": chosen.get("_primary_metric"),
            "primary_value": chosen.get("_primary_value"),
            "eval_acc": chosen.get("_eval_acc"),
            "eval_auroc": chosen.get("_eval_auroc"),
            "eval_auprc": chosen.get("_eval_auprc"),
            "acc_ratio_vs_backprop": chosen.get("_acc_ratio_vs_backprop"),
            "avg_epoch_s": chosen.get("_avg_epoch_s"),
            "graphs_per_s": chosen.get("_graphs_per_s"),
            "speed_ratio_vs_backprop": None,
            "delta_primary_vs_backprop": None,
            "delta_acc_vs_backprop": None,
        }
        if backprop is not None:
            back_speed = backprop.get("_graphs_per_s")
            back_primary = backprop.get("_primary_value")
            back_acc = backprop.get("_eval_acc")
            if back_speed and back_speed > 0 and row["graphs_per_s"] is not None:
                row["speed_ratio_vs_backprop"] = row["graphs_per_s"] / back_speed
            if back_primary is not None and row["primary_value"] is not None:
                row["delta_primary_vs_backprop"] = row["primary_value"] - back_primary
            if back_acc is not None and row["eval_acc"] is not None:
                row["delta_acc_vs_backprop"] = row["eval_acc"] - back_acc

        for k in (
            "goodness_target",
            "goodness_temp",
            "neg_mix_end",
            "hall_steps",
            "hall_lr",
            "hall_node_fraction",
            "distance_forward_weight",
            "self_contrastive_ff_weight",
        ):
            if k in chosen:
                row[k] = chosen.get(k, "")
        out_rows.append(row)

    out_txt = Path(args.out)
    out_csv = Path(args.out_csv)
    _write_text_report(out_txt, out_rows, backprop)
    _write_csv(out_csv, out_rows)

    print(f"Wrote {out_txt}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
