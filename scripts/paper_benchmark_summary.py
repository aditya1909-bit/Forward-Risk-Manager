#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    s = str(value).strip()
    if not s:
        return None
    try:
        out = float(s)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value * 100:.{digits}f}%"


def _read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _mode_order_key(mode: str) -> tuple[int, str]:
    order = {"ff_layerwise": 0, "ff_e2e": 1, "backprop": 2}
    return (order.get(mode, 99), mode)


def _pick_aggregate_rows(rows: list[dict]) -> list[dict]:
    if "row_type" in rows[0]:
        agg = [r for r in rows if str(r.get("row_type", "")).strip().lower() == "aggregate"]
        if agg:
            return agg
    return rows


def _summarize_folds(fold_rows: list[dict]) -> dict[str, dict[str, float]]:
    by_mode: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)  # type: ignore[arg-type]
    )
    metrics = [
        "primary_eval_metric",
        "graphs_per_s",
        "avg_epoch_s",
        "econ_sharpe_uplift",
        "econ_ann_return_uplift",
    ]
    for row in fold_rows:
        mode = str(row.get("mode", "")).strip()
        if not mode:
            continue
        for m in metrics:
            v = _to_float(row.get(m))
            if v is not None:
                by_mode[mode][m].append(v)

    out: dict[str, dict[str, float]] = {}
    for mode, metric_map in by_mode.items():
        out[mode] = {}
        for metric, vals in metric_map.items():
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                std = math.sqrt(var)
            else:
                std = 0.0
            out[mode][f"{metric}_mean"] = mean
            out[mode][f"{metric}_std"] = std
            out[mode][f"{metric}_n"] = float(len(vals))
    return out


def _build_summary_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        mode = str(row.get("mode", "")).strip()
        q_name = str(row.get("primary_eval_metric_name", "")).strip() or "eval_sep"
        q_val = _to_float(row.get("primary_eval_metric"))
        if q_val is None:
            q_val = _to_float(row.get("eval_sep"))
        summary = {
            "mode": mode,
            "objective_track": str(row.get("objective_track", "")).strip(),
            "quality_metric": q_name,
            "quality_value": q_val,
            "quality_std": _to_float(row.get("primary_eval_metric_std"))
            or _to_float(row.get("eval_sep_std")),
            "eval_acc": _to_float(row.get("eval_acc")),
            "eval_auroc": _to_float(row.get("eval_auroc")),
            "eval_auprc": _to_float(row.get("eval_auprc")),
            "avg_epoch_s": _to_float(row.get("avg_epoch_s")),
            "graphs_per_s": _to_float(row.get("graphs_per_s")),
            "econ_sharpe_uplift": _to_float(row.get("econ_sharpe_uplift")),
            "econ_ann_return_uplift": _to_float(row.get("econ_ann_return_uplift")),
            "econ_strategy_ann_return": _to_float(row.get("econ_strategy_ann_return")),
            "econ_strategy_ann_vol": _to_float(row.get("econ_strategy_ann_vol")),
            "econ_strategy_max_drawdown": _to_float(row.get("econ_strategy_max_drawdown")),
            "econ_strategy_sharpe": _to_float(row.get("econ_strategy_sharpe")),
            "econ_bh_ann_return": _to_float(row.get("econ_bh_ann_return")),
            "econ_bh_sharpe": _to_float(row.get("econ_bh_sharpe")),
            "walk_forward_num_folds": _to_float(row.get("walk_forward_num_folds")),
        }
        out.append(summary)
    out.sort(key=lambda r: _mode_order_key(str(r["mode"])))
    return out


def _markdown_table(summary_rows: list[dict]) -> str:
    headers = [
        "mode",
        "quality_metric",
        "quality_value",
        "eval_auroc",
        "eval_auprc",
        "avg_epoch_s",
        "graphs_per_s",
        "econ_sharpe_uplift",
        "econ_ann_return_uplift",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["mode"]),
                    str(r["quality_metric"]),
                    _fmt(r["quality_value"]),
                    _fmt(r["eval_auroc"]),
                    _fmt(r["eval_auprc"]),
                    _fmt(r["avg_epoch_s"], digits=3),
                    _fmt(r["graphs_per_s"], digits=1),
                    _fmt(r["econ_sharpe_uplift"], digits=3),
                    _fmt_pct(r["econ_ann_return_uplift"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _pick_best(rows: list[dict], key: str) -> dict | None:
    best_row = None
    best_val = None
    for row in rows:
        val = _to_float(row.get(key))
        if val is None:
            continue
        if best_val is None or val > best_val:
            best_val = val
            best_row = row
    return best_row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready benchmark summary table from benchmark.csv."
    )
    parser.add_argument("--benchmark", required=True, help="Path to benchmark.csv")
    parser.add_argument(
        "--folds",
        default="",
        help="Optional benchmark fold CSV (defaults to benchmark_walk_forward_folds.csv if present)",
    )
    parser.add_argument("--out-md", default="", help="Optional markdown output path")
    parser.add_argument("--out-csv", default="", help="Optional compact CSV output path")
    parser.add_argument("--out-json", default="", help="Optional JSON output path")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    rows = _read_rows(benchmark_path)
    agg_rows = _pick_aggregate_rows(rows)
    summary_rows = _build_summary_rows(agg_rows)

    folds_path = Path(args.folds) if args.folds else benchmark_path.with_name(
        "benchmark_walk_forward_folds.csv"
    )
    fold_summary = {}
    if folds_path.exists():
        fold_rows = _read_rows(folds_path)
        fold_summary = _summarize_folds(fold_rows)

    md = _markdown_table(summary_rows)

    fastest = _pick_best(summary_rows, "graphs_per_s")
    best_quality = _pick_best(summary_rows, "quality_value")
    best_econ = _pick_best(summary_rows, "econ_sharpe_uplift")

    print("Paper Benchmark Summary")
    print(f"source: {benchmark_path}")
    print(md)
    print("")
    if fastest:
        print(
            "fastest_mode: "
            f"{fastest['mode']} ({_fmt(fastest['graphs_per_s'], 1)} graphs/s, "
            f"{_fmt(fastest['avg_epoch_s'], 3)} s/epoch)"
        )
    if best_quality:
        print(
            "best_quality_mode: "
            f"{best_quality['mode']} ({best_quality['quality_metric']}="
            f"{_fmt(best_quality['quality_value'])})"
        )
    if best_econ:
        print(
            "best_econ_mode: "
            f"{best_econ['mode']} (econ_sharpe_uplift={_fmt(best_econ['econ_sharpe_uplift'], 3)}, "
            f"econ_ann_return_uplift={_fmt_pct(best_econ['econ_ann_return_uplift'])})"
        )
    else:
        print("best_econ_mode: unavailable (econ metrics are missing or NaN)")

    if fold_summary:
        print("")
        print("fold_summary:")
        print(json.dumps(fold_summary, indent=2, sort_keys=True))

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Paper Benchmark Summary", "", f"Source: `{benchmark_path}`", "", md]
        key_points = []
        if fastest:
            key_points.append(
                f"- Fastest: `{fastest['mode']}` ({_fmt(fastest['graphs_per_s'], 1)} graphs/s)."
            )
        if best_quality:
            key_points.append(
                f"- Best quality: `{best_quality['mode']}` ({best_quality['quality_metric']}={_fmt(best_quality['quality_value'])})."
            )
        if best_econ:
            key_points.append(
                f"- Best economics: `{best_econ['mode']}` (Sharpe uplift={_fmt(best_econ['econ_sharpe_uplift'], 3)}, ann return uplift={_fmt_pct(best_econ['econ_ann_return_uplift'])})."
            )
        else:
            key_points.append(
                "- Best economics: unavailable (econ metrics are missing or NaN)."
            )
        if key_points:
            lines.extend(["", "## Key Points", *key_points])
        out_md.write_text("\n".join(lines) + "\n")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(benchmark_path),
            "rows": summary_rows,
            "fastest_mode": fastest,
            "best_quality_mode": best_quality,
            "best_econ_mode": best_econ,
            "fold_summary": fold_summary,
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
