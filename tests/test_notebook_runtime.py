from __future__ import annotations

from pathlib import Path

from frisk.notebook_runtime import (
    NotebookStage,
    format_duration,
    record_run_step,
    remaining_eta_seconds,
    stage_status_rows,
)


def test_format_duration_handles_minutes_and_hours():
    assert format_duration(59) == "59s"
    assert format_duration(61) == "1m 01s"
    assert format_duration(3661) == "1h 01m 01s"


def test_stage_status_rows_and_remaining_eta_respect_completed_outputs(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    complete_output = tmp_path / "done.txt"
    complete_output.write_text("ok", encoding="utf-8")
    record_run_step(
        manifest_path,
        step="benchmark",
        status="completed",
        required_outputs=[complete_output],
        metadata={"elapsed_s": 12.0, "eta_s": 120.0},
    )

    stages = [
        NotebookStage("benchmark", "Benchmark", (str(complete_output),), eta_s=120.0),
        NotebookStage("sweep", "Sweep", (str(tmp_path / "sweep.csv"),), eta_s=300.0),
    ]

    rows = stage_status_rows(manifest_path, stages)
    assert rows[0]["status"] == "completed"
    assert rows[0]["outputs_ready"] is True
    assert rows[1]["status"] == "pending"
    assert remaining_eta_seconds(manifest_path, stages) == 300.0
