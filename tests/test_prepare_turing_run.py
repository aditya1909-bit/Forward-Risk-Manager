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
