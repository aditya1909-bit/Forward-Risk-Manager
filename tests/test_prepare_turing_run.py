from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_prepare_turing_run_rewrites_core_paths(tmp_path: Path):
    base_cfg = tmp_path / "base.toml"
    runtime_cfg = tmp_path / "runtime.toml"
    base_cfg.write_text(
        "\n".join(
            [
                "[build_graphs]",
                'prices = "data/processed/prices.csv"',
                'out = "data/processed/graphs_master_ff_rich.pt"',
                "",
                "[train]",
                'graphs = "data/processed/graphs_master_ff_rich.pt.sharded"',
                'log_csv = "runs/experiments/default/metrics/ff_train.csv"',
                'save_encoder = "runs/experiments/default/models/encoder.pt"',
                'risk_cache_dir = "runs/cache"',
                "",
                "[benchmark]",
                'out_csv = "runs/experiments/default/metrics/benchmark.csv"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_turing_run.py",
            "--base-config",
            str(base_cfg),
            "--runtime-config",
            str(runtime_cfg),
            "--netid",
            "adutta",
            "--scratch-root",
            str(tmp_path / "scratch"),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    text = runtime_cfg.read_text(encoding="utf-8")
    assert str(tmp_path / "scratch" / "data" / "processed" / "prices.csv") in text
    assert str(tmp_path / "scratch" / "runs" / "experiments" / "default" / "metrics" / "ff_train.csv") in text
    assert str(tmp_path / "scratch" / ".cache") in text


def test_prepare_turing_run_applies_non_cluster_overlay_sections(tmp_path: Path):
    base_cfg = tmp_path / "base.toml"
    overlay_cfg = tmp_path / "overlay.toml"
    runtime_cfg = tmp_path / "runtime.toml"
    scratch_root = tmp_path / "scratch"
    sharded_graph = scratch_root / "data" / "processed" / "graphs_master_ff_rich.pt.sharded"
    base_cfg.write_text(
        "\n".join(
            [
                "[train]",
                'graphs = "data/processed/graphs.pt"',
                "distributed = true",
                'risk_ticker = "AUTO"',
                "portfolio_head_enabled = false",
                "",
                "[benchmark]",
                "enable_risk_head = false",
                "enable_portfolio_head = false",
                'econ_ticker = "AUTO"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    overlay_cfg.write_text(
        "\n".join(
            [
                "[cluster]",
                'netid = "asdutta"',
                "",
                "[train]",
                f'graphs = "{sharded_graph}"',
                "distributed = false",
                'risk_ticker = "SPY"',
                "portfolio_head_enabled = true",
                "",
                "[benchmark]",
                "enable_risk_head = true",
                "enable_portfolio_head = true",
                'econ_ticker = "SPY"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_turing_run.py",
            "--base-config",
            str(base_cfg),
            "--cluster-config",
            str(overlay_cfg),
            "--runtime-config",
            str(runtime_cfg),
            "--netid",
            "asdutta",
            "--scratch-root",
            str(scratch_root),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    text = runtime_cfg.read_text(encoding="utf-8")
    assert "distributed = false" in text
    assert 'risk_ticker = "SPY"' in text
    assert "portfolio_head_enabled = true" in text
    assert "enable_risk_head = true" in text
    assert "enable_portfolio_head = true" in text
    assert 'econ_ticker = "SPY"' in text
