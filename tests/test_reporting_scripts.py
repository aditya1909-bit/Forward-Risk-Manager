from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_min_scenario_csv(path: Path, *, target_ticker: str = "AAA") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario_id",
                "graph_index",
                "date",
                "target_ticker",
                "ticker",
                "series",
                "r0",
                "r1",
            ]
        )
        w.writerow([0, 0, "2024-01-01", target_ticker, target_ticker, "real", 0.01, -0.02])
        w.writerow([0, 0, "2024-01-01", target_ticker, target_ticker, "halluc", 0.02, -0.01])


def test_dual_score_primary_metric_prefers_backprop_auroc():
    mod = _load_script("dual_score_report.py")
    metric, value = mod._primary_metric(
        {
            "eval_objective": "bce",
            "eval_sep": "0.0004",
            "eval_auroc": "0.992",
            "eval_auprc": "0.991",
        }
    )
    assert metric == "eval_auroc"
    assert value == pytest.approx(0.992)


def test_dual_score_primary_metric_uses_robust_metric_when_present():
    mod = _load_script("dual_score_report.py")
    metric, value = mod._primary_metric(
        {
            "eval_objective": "self_contrastive",
            "eval_sc_gap": "0.86",
            "primary_eval_metric_robust": "0.86",
            "primary_eval_metric_robust_name": "eval_sc_gap",
        }
    )
    assert metric == "eval_sc_gap"
    assert value == pytest.approx(0.86)


def test_promote_auto_rank_prefers_robust_metric():
    mod = _load_script("promote_sweep_best.py")
    rows = [
        {"rank_value": "0.9", "primary_eval_metric_robust": "0.1"},
        {"rank_value": "0.8", "primary_eval_metric_robust": "0.2"},
    ]
    assert mod._pick_rank_column(rows, "auto") == "primary_eval_metric_robust"


def test_stress_report_rejects_mismatched_target_ticker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_script("stress_test_report.py")
    scenario_csv = tmp_path / "scenario.csv"
    _write_min_scenario_csv(scenario_csv, target_ticker="AAA")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stress_test_report.py",
            "--csv",
            str(scenario_csv),
            "--target-ticker",
            "BBB",
            "--out-csv",
            str(tmp_path / "stress.csv"),
            "--out-plot",
            str(tmp_path / "stress.png"),
        ],
    )
    with pytest.raises(ValueError, match="does not match scenario CSV target_ticker metadata"):
        mod.main()


def test_hallucination_calibration_rejects_mismatched_target_ticker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_script("hallucination_calibration.py")
    scenario_csv = tmp_path / "scenario.csv"
    _write_min_scenario_csv(scenario_csv, target_ticker="AAA")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hallucination_calibration.py",
            "--csv",
            str(scenario_csv),
            "--target-ticker",
            "BBB",
            "--out",
            str(tmp_path / "cal.json"),
            "--out-by-ticker",
            str(tmp_path / "cal_by_ticker.csv"),
        ],
    )
    with pytest.raises(ValueError, match="does not match scenario CSV target_ticker metadata"):
        mod.main()


def test_publish_run_removes_stale_latest_tune(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_script("publish_run.py")
    monkeypatch.chdir(tmp_path)

    run_root = tmp_path / "runs" / "experiments" / "testrun"
    (run_root / "metrics").mkdir(parents=True, exist_ok=True)
    (run_root / "plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "published" / "sweep").mkdir(parents=True, exist_ok=True)

    (run_root / "metrics" / "benchmark.csv").write_text("mode,eval_sep\nff_layerwise,0.1\n")
    stale_tune = tmp_path / "reports" / "published" / "sweep" / "latest_tune.csv"
    stale_tune.write_text("stale\n")

    index_path = tmp_path / "reports" / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["category", "name", "path", "source_run", "kind", "updated_at_utc"],
        )
        w.writeheader()
        w.writerow(
            {
                "category": "sweep",
                "name": "latest_tune.csv",
                "path": "reports/published/sweep/latest_tune.csv",
                "source_run": "legacy",
                "kind": "metric",
                "updated_at_utc": "2026-02-11T00:00:00Z",
            }
        )

    monkeypatch.setattr(sys, "argv", ["publish_run.py", "--run-id", "testrun"])
    mod.main()

    assert not stale_tune.exists()
    with index_path.open() as f:
        rows = list(csv.DictReader(f))
    assert not any(r["category"] == "sweep" and r["name"] == "latest_tune.csv" for r in rows)


def test_sanity_checks_handles_non_finite_easy_negative_acc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_script("sanity_checks.py")
    bench_csv = tmp_path / "benchmark.csv"
    with bench_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eval_neg_mode_effective",
                "eval_objective",
                "eval_acc",
                "eval_time_flip_sep",
            ]
        )
        w.writerow(["shuffle", "ff", "", "0.10"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_checks.py",
            "--benchmark-csv",
            str(bench_csv),
            "--timeflip-sep-min",
            "0.05",
        ],
    )
    assert mod.main() == 0


def test_sanity_checks_does_not_gate_self_contrastive_timeflip_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_script("sanity_checks.py")
    bench_csv = tmp_path / "benchmark.csv"
    with bench_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eval_neg_mode_effective",
                "eval_objective",
                "eval_acc",
                "eval_time_flip_sep",
                "eval_time_flip_auroc",
            ]
        )
        w.writerow(["time_flip+noise", "ff", "0.80", "0.22", "0.61"])
        w.writerow(["self_contrastive", "self_contrastive", "0.33", "-2.1e-06", "0.49793754865218404"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_checks.py",
            "--benchmark-csv",
            str(bench_csv),
            "--timeflip-sep-min",
            "0.05",
            "--sc-timeflip-sep-min",
            "0.0",
            "--sc-timeflip-auroc-min",
            "0.5",
        ],
    )
    assert mod.main() == 0


def test_sanity_checks_can_optionally_enforce_sc_timeflip_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_script("sanity_checks.py")
    bench_csv = tmp_path / "benchmark.csv"
    with bench_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eval_neg_mode_effective",
                "eval_objective",
                "eval_acc",
                "eval_time_flip_sep",
                "eval_time_flip_auroc",
            ]
        )
        w.writerow(["time_flip+noise", "ff", "0.80", "0.22", "0.61"])
        w.writerow(["self_contrastive", "self_contrastive", "0.33", "-2.1e-06", "0.49793754865218404"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_checks.py",
            "--benchmark-csv",
            str(bench_csv),
            "--timeflip-sep-min",
            "0.05",
            "--sc-timeflip-sep-min",
            "0.0",
            "--sc-timeflip-auroc-min",
            "0.5",
            "--enforce-sc-timeflip-checks",
            "--sep-atol",
            "0.0",
            "--auroc-atol",
            "0.0",
        ],
    )
    assert mod.main() == 1


def test_sanity_checks_critic_timeflip_gate_ignores_self_contrastive_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_script("sanity_checks.py")
    bench_csv = tmp_path / "benchmark.csv"
    with bench_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eval_neg_mode_effective",
                "eval_objective",
                "eval_acc",
                "eval_time_flip_sep",
                "eval_time_flip_auroc",
            ]
        )
        # Critic row should fail the gate.
        w.writerow(["time_flip+noise", "ff", "0.80", "0.01", "0.61"])
        # SC row should not rescue critic time-flip failure.
        w.writerow(["self_contrastive", "self_contrastive", "0.99", "0.99", "0.99"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_checks.py",
            "--benchmark-csv",
            str(bench_csv),
            "--timeflip-sep-min",
            "0.05",
        ],
    )
    assert mod.main() == 1


def test_sanity_checks_can_gate_self_contrastive_by_gap_and_skip_timeflip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_script("sanity_checks.py")
    bench_csv = tmp_path / "benchmark.csv"
    with bench_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eval_neg_mode_effective",
                "eval_objective",
                "eval_acc",
                "eval_sc_gap",
                "eval_time_flip_sep",
                "eval_time_flip_auroc",
            ]
        )
        w.writerow(["time_flip+noise", "ff", "0.80", "", "0.22", "0.61"])
        # Strong SC gap but intentionally bad SC time-flip metrics.
        w.writerow(["self_contrastive", "self_contrastive", "1.0", "0.91", "-0.02", "0.47"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_checks.py",
            "--benchmark-csv",
            str(bench_csv),
            "--timeflip-sep-min",
            "0.05",
            "--sc-gap-min",
            "0.2",
            "--skip-sc-timeflip-checks",
        ],
    )
    assert mod.main() == 0


def test_sanity_checks_sc_gap_min_fails_when_gap_is_too_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_script("sanity_checks.py")
    bench_csv = tmp_path / "benchmark.csv"
    with bench_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eval_neg_mode_effective",
                "eval_objective",
                "eval_acc",
                "eval_sc_gap",
                "eval_time_flip_sep",
            ]
        )
        w.writerow(["time_flip+noise", "ff", "0.80", "", "0.22"])
        w.writerow(["self_contrastive", "self_contrastive", "1.0", "0.08", "-0.01"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_checks.py",
            "--benchmark-csv",
            str(bench_csv),
            "--timeflip-sep-min",
            "0.05",
            "--sc-gap-min",
            "0.2",
            "--skip-sc-timeflip-checks",
        ],
    )
    assert mod.main() == 1


def test_sanity_checks_easy_negative_acc_ignores_backprop_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_script("sanity_checks.py")
    bench_csv = tmp_path / "benchmark.csv"
    with bench_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eval_neg_mode_effective",
                "eval_objective",
                "eval_acc",
                "eval_time_flip_sep",
            ]
        )
        # FF row should be used for easy-negative gate.
        w.writerow(["shuffle+noise", "ff", "0.99", "0.10"])
        # Backprop row can saturate eval_acc and should not fail easy-negative gate.
        w.writerow(["shuffle+noise", "bce", "1.0", "0.00"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_checks.py",
            "--benchmark-csv",
            str(bench_csv),
            "--easy-neg-acc-max",
            "0.995",
            "--timeflip-sep-min",
            "0.05",
        ],
    )
    assert mod.main() == 0
