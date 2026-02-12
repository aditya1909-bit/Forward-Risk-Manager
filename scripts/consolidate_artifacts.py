#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Iterable


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _move_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return "missing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    shutil.move(str(src), str(dst))
    return "moved"


def _copy_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return "missing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return "same"
    if src.is_file():
        shutil.copy2(src, dst)
        return "copied"
    return "skipped"


def _copy_first_exists(srcs: Iterable[Path], dst: Path) -> tuple[str, Path | None]:
    for src in srcs:
        status = _copy_if_exists(src, dst)
        if status in {"copied", "same"}:
            return status, src
    return "missing", None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate legacy report artifacts into run-scoped layout."
    )
    parser.add_argument(
        "--run-id",
        default="legacy-20260211-colab",
        help="Run id under runs/experiments/<run_id>",
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    run_id = str(args.run_id).strip()
    if not run_id:
        raise ValueError("--run-id must be non-empty")

    run_root = root / "runs" / "experiments" / run_id
    ts = _utc_now()

    move_map = [
        ("reports/benchmark.csv", "metrics/benchmark.csv"),
        ("reports/benchmark.png", "plots/benchmark_bar.png"),
        ("reports/benchmark_bar.png", "plots/benchmark_bar.png"),
        ("reports/benchmark_speed_sep.png", "plots/benchmark_speed_sep.png"),
        ("reports/dual_score_report.csv", "metrics/dual_score_report.csv"),
        ("reports/dual_score_report.txt", "logs/dual_score_report.txt"),
        ("reports/ff_sweep.csv", "metrics/ff_sweep.csv"),
        ("reports/ff_sweep_e2e.csv", "metrics/ff_sweep_e2e.csv"),
        ("reports/ff_sweep_pareto.png", "plots/ff_sweep_pareto.png"),
        ("reports/ff_sweep_tradeoff.png", "plots/ff_sweep_tradeoff.png"),
        ("reports/ff_sweep_summary.txt", "logs/ff_sweep_summary.txt"),
        ("reports/ff_sweep_e2e_summary.txt", "logs/ff_sweep_e2e_summary.txt"),
        ("reports/sweep_parallel_tune.csv", "metrics/sweep_parallel_tune.csv"),
        ("reports/ff_train.csv", "metrics/ff_train_default.csv"),
        ("reports/ff_train.png", "plots/ff_train_default.png"),
        ("reports/ff_train_long_constituents.csv", "metrics/ff_train_long_constituents.csv"),
        ("reports/ff_train_long_constituents.png", "plots/ff_train_long_constituents.png"),
        ("reports/ff_train_long_alltickers.csv", "metrics/ff_train_long_alltickers.csv"),
        ("reports/ff_train_long_alltickers.png", "plots/ff_train_long_alltickers.png"),
        ("reports/scenario_book.csv", "metrics/scenario_book.csv"),
        ("reports/scenario_book_100.csv", "metrics/scenario_book_top100.csv"),
        ("reports/scenario_constraint_diagnostics.csv", "diagnostics/scenario_constraint_diagnostics.csv"),
        ("reports/stress_test_report.csv", "metrics/stress_test_report.csv"),
        ("reports/stress_test_report.png", "plots/stress_test_report.png"),
        ("reports/stress_test_report_100.csv", "metrics/stress_test_report_top100.csv"),
        ("reports/stress_test_report_100.png", "plots/stress_test_report_top100.png"),
        ("reports/hallucination_calibration.json", "diagnostics/hallucination_calibration.json"),
        (
            "reports/hallucination_calibration_by_ticker.csv",
            "diagnostics/hallucination_calibration_by_ticker.csv",
        ),
        ("reports/hallucination_diagnostics.png", "diagnostics/hallucination_diagnostics.png"),
        ("reports/hallucination_plot.png", "diagnostics/hallucination_plot.png"),
        ("reports/hallucination_window.csv", "diagnostics/hallucination_window.csv"),
        ("reports/hallucination_window_all.csv", "diagnostics/hallucination_window_all.csv"),
        ("reports/goodness_backtest.csv", "diagnostics/goodness_backtest.csv"),
        ("reports/goodness_quantiles.csv", "diagnostics/goodness_quantiles.csv"),
        ("reports/goodness_scatter.png", "diagnostics/goodness_scatter.png"),
        ("reports/colab_intensive_bundle.zip", "bundles/colab_intensive_bundle.zip"),
        ("runs/ff_model.pt", "models/ff_model_default.pt"),
        ("runs/ff_model_long_constituents.pt", "models/ff_model_long_constituents.pt"),
        ("runs/ff_model_long_alltickers.pt", "models/ff_model_long_alltickers.pt"),
    ]

    publish_map = [
        (["metrics/benchmark.csv"], "benchmark/latest.csv", "metric"),
        (["plots/benchmark_speed_sep.png"], "benchmark/latest_speed_sep.png", "plot"),
        (["plots/benchmark_bar.png", "plots/benchmark.png"], "benchmark/latest_bar.png", "plot"),
        (["metrics/dual_score_report.csv"], "benchmark/dual_score_latest.csv", "metric"),
        (["logs/dual_score_report.txt"], "benchmark/dual_score_latest.txt", "summary"),
        (["metrics/ff_sweep.csv"], "sweep/latest.csv", "metric"),
        (["metrics/ff_sweep_e2e.csv"], "sweep/latest_e2e.csv", "metric"),
        (["plots/ff_sweep_pareto.png"], "sweep/latest_pareto.png", "plot"),
        (["plots/ff_sweep_tradeoff.png"], "sweep/latest_tradeoff.png", "plot"),
        (["logs/ff_sweep_summary.txt"], "sweep/latest_summary.txt", "summary"),
        (["logs/ff_sweep_e2e_summary.txt"], "sweep/latest_e2e_summary.txt", "summary"),
        (["metrics/sweep_parallel_tune.csv"], "sweep/latest_tune.csv", "metric"),
        (["metrics/ff_train_default.csv", "metrics/ff_train.csv"], "train/default_latest.csv", "metric"),
        (["plots/ff_train_default.png", "plots/ff_train.png"], "train/default_latest.png", "plot"),
        (
            ["metrics/ff_train_long_constituents.csv", "metrics/ff_train.csv"],
            "train/long_constituents_latest.csv",
            "metric",
        ),
        (
            ["plots/ff_train_long_constituents.png", "plots/ff_train.png"],
            "train/long_constituents_latest.png",
            "plot",
        ),
        (["metrics/scenario_book.csv"], "scenario/latest.csv", "metric"),
        (
            ["diagnostics/scenario_constraint_diagnostics.csv"],
            "scenario/constraints_latest.csv",
            "diagnostic",
        ),
        (["metrics/stress_test_report.csv"], "scenario/stress_latest.csv", "metric"),
        (["plots/stress_test_report.png"], "scenario/stress_latest.png", "plot"),
        (["diagnostics/hallucination_calibration.json"], "hallucination/calibration_latest.json", "diagnostic"),
        (
            ["diagnostics/hallucination_calibration_by_ticker.csv"],
            "hallucination/calibration_by_ticker_latest.csv",
            "diagnostic",
        ),
        (["diagnostics/hallucination_diagnostics.png"], "hallucination/diagnostics_latest.png", "diagnostic"),
        (["diagnostics/hallucination_plot.png"], "hallucination/plot_latest.png", "diagnostic"),
    ]

    moved_rows = []
    for src_rel, dst_rel in move_map:
        src = root / src_rel
        dst = run_root / dst_rel
        status = _move_if_exists(src, dst)
        moved_rows.append(
            {
                "src": src_rel,
                "dst": str(Path("runs/experiments") / run_id / dst_rel),
                "status": status,
            }
        )

    published_rows = []
    for src_rels, pub_rel, kind in publish_map:
        src_candidates = [run_root / src_rel for src_rel in src_rels]
        dst = root / "reports" / "published" / pub_rel
        status, _ = _copy_first_exists(src_candidates, dst)
        if status in {"copied", "same"}:
            published_rows.append(
                {
                    "category": pub_rel.split("/", 1)[0],
                    "name": Path(pub_rel).name,
                    "path": str(Path("reports/published") / pub_rel),
                    "source_run": run_id,
                    "kind": kind,
                    "updated_at_utc": ts,
                }
            )

    manifest = {
        "run_id": run_id,
        "created_at_utc": ts,
        "moved": moved_rows,
        "published": published_rows,
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    registry_path = root / "runs" / "registry" / "runs.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_rows = []
    if registry_path.exists():
        with registry_path.open() as f:
            reader = csv.DictReader(f)
            registry_rows = [r for r in reader if r.get("run_id") != run_id]
    registry_rows.append(
        {
            "run_id": run_id,
            "created_at_utc": ts,
            "run_root": str(Path("runs/experiments") / run_id),
        }
    )
    with registry_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "created_at_utc", "run_root"])
        writer.writeheader()
        for row in registry_rows:
            writer.writerow(row)

    index_path = root / "reports" / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="") as f:
        fieldnames = ["category", "name", "path", "source_run", "kind", "updated_at_utc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(published_rows, key=lambda x: (x["category"], x["name"])):
            writer.writerow(row)

    print(f"Consolidated artifacts into runs/experiments/{run_id}")
    print(f"Wrote {manifest_path.relative_to(root)}")
    print(f"Wrote {registry_path.relative_to(root)}")
    print(f"Wrote {index_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
