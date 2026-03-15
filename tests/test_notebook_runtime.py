from __future__ import annotations

import json
from pathlib import Path

from frisk.notebook_runtime import (
    build_source_fingerprint,
    merge_csv_files,
    record_run_step,
    step_is_complete,
    write_toml_overrides,
)


def test_write_toml_overrides_replaces_and_appends(tmp_path: Path):
    base = tmp_path / "base.toml"
    base.write_text(
        "\n".join(
            [
                "[train]",
                'epochs = 10',
                'device = "cpu"',
                "",
                "[benchmark]",
                'out_csv = "old.csv"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    out = tmp_path / "runtime.toml"
    write_toml_overrides(
        base,
        out,
        {
            "train": {"epochs": 25, "batch_size": 16},
            "new_section": {"enabled": True},
        },
    )

    text = out.read_text(encoding="utf-8")
    assert "epochs = 25" in text
    assert "batch_size = 16" in text
    assert "[new_section]" in text
    assert "enabled = true" in text


def test_merge_csv_files_unions_columns(tmp_path: Path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("mode,score\nff,0.1\n", encoding="utf-8")
    right.write_text("mode,graphs_per_s\nbp,12\n", encoding="utf-8")

    out = merge_csv_files([left, right], tmp_path / "merged.csv")
    merged = out.read_text(encoding="utf-8")
    assert "mode,score,graphs_per_s" in merged
    assert "ff,0.1," in merged
    assert "bp,,12" in merged


def test_record_run_step_and_completion_check(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "metrics.csv"
    output.write_text("x\n", encoding="utf-8")

    record_run_step(
        manifest_path,
        step="benchmark",
        status="completed",
        command="python scripts/benchmark_training.py",
        required_outputs=[output],
        metadata={"mode": "ff_e2e"},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["steps"]["benchmark"]["metadata"]["mode"] == "ff_e2e"
    assert step_is_complete(manifest_path, "benchmark")
    assert step_is_complete(manifest_path, "benchmark", required_outputs=[output])


def test_build_source_fingerprint_tracks_config_and_inputs(tmp_path: Path):
    prices = tmp_path / "prices.csv"
    prices.write_text("date,ticker,close\n2024-01-01,AAA,1.0\n", encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "\n".join(
            [
                "[build_graphs]",
                f'prices = "{prices}"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    fingerprint = build_source_fingerprint(cfg, tracked_keys=["prices"], extra_fields={"lags": {"corr": 1}})
    assert fingerprint["build_config"] == str(cfg)
    assert fingerprint["lags"]["corr"] == 1
    assert any(row["path"] == str(prices) for row in fingerprint["files"])
