#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Iterable


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_first_exists(srcs: Iterable[Path], dst: Path) -> tuple[bool, Path | None]:
    for src in srcs:
        if _copy_if_exists(src, dst):
            return True, src
    return False, None


def _remove_if_exists(path: Path) -> bool:
    if path.exists():
        path.unlink()
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish curated artifacts from runs/experiments/<run_id> to reports/published."
    )
    parser.add_argument("--run-id", required=True, help="Run id under runs/experiments/<run_id>")
    args = parser.parse_args()

    root = Path(".").resolve()
    run_id = str(args.run_id).strip()
    run_root = root / "runs" / "experiments" / run_id
    if not run_root.exists():
        raise FileNotFoundError(f"Run directory not found: {run_root}")

    ts = _utc_now()
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
    run_name = run_id.lower()
    if "long_constituents" in run_name:
        publish_map.extend(
            [
                (["metrics/ff_train_long_constituents.csv", "metrics/ff_train.csv"], "train/long_constituents_latest.csv", "metric"),
                (["plots/ff_train_long_constituents.png", "plots/ff_train.png"], "train/long_constituents_latest.png", "plot"),
            ]
        )
    if run_name == "default" or run_name.startswith("default_"):
        publish_map.extend(
            [
                (["metrics/ff_train_default.csv", "metrics/ff_train.csv"], "train/default_latest.csv", "metric"),
                (["plots/ff_train_default.png", "plots/ff_train.png"], "train/default_latest.png", "plot"),
            ]
        )
    # Keep train track outputs disjoint by profile; remove opposite-track latest files.
    if "long_constituents" in run_name:
        publish_map.extend(
            [
                ([], "train/default_latest.csv", "metric"),
                ([], "train/default_latest.png", "plot"),
            ]
        )
    if run_name == "default" or run_name.startswith("default_"):
        publish_map.extend(
            [
                ([], "train/long_constituents_latest.csv", "metric"),
                ([], "train/long_constituents_latest.png", "plot"),
            ]
        )

    published_rows = []
    touched_keys = set()
    for src_rels, pub_rel, kind in publish_map:
        key = (pub_rel.split("/", 1)[0], Path(pub_rel).name)
        touched_keys.add(key)
        src_candidates = [run_root / src_rel for src_rel in src_rels]
        dst = root / "reports" / "published" / pub_rel
        copied, src = _copy_first_exists(src_candidates, dst)
        if not copied:
            if _remove_if_exists(dst):
                print(f"Removed stale artifact: {dst.relative_to(root)}")
            continue
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

    index_path = root / "reports" / "index.csv"
    existing = []
    if index_path.exists():
        with index_path.open() as f:
            reader = csv.DictReader(f)
            existing = list(reader)

    existing = [r for r in existing if (r.get("category"), r.get("name")) not in touched_keys]
    existing.extend(published_rows)
    existing.sort(key=lambda r: (r.get("category", ""), r.get("name", "")))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="") as f:
        fieldnames = ["category", "name", "path", "source_run", "kind", "updated_at_utc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing:
            writer.writerow(row)

    print(f"Published {len(published_rows)} artifacts from run {run_id}")
    print("Wrote reports/index.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
