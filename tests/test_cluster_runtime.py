from __future__ import annotations

from pathlib import Path

import pytest

from frisk.cluster_runtime import ensure_safe_cluster_path, resolve_cluster_layout
from frisk.distributed_runtime import checkpoint_due


def test_resolve_cluster_layout_expands_netid_tokens():
    layout = resolve_cluster_layout(
        {
            "enabled": True,
            "netid": "adutta",
            "scratch_root": "/local/scratch/<netid>/forward-risk-manager",
        }
    )
    assert str(layout.scratch_root) == "/local/scratch/adutta/forward-risk-manager"
    assert str(layout.repo_root).endswith("/repo")
    assert str(layout.cache_root).endswith("/.cache")


def test_ensure_safe_cluster_path_rejects_home_writes():
    with pytest.raises(ValueError):
        ensure_safe_cluster_path(
            "/home/adutta/results/model.pt",
            cluster_cfg={"enabled": True, "netid": "adutta"},
            label="train.save_encoder",
        )


def test_checkpoint_due_respects_interval_and_none():
    assert checkpoint_due(last_saved_at=None, every_minutes=55) is True
    assert checkpoint_due(last_saved_at=10_000.0, every_minutes=0) is False
