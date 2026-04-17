from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from frisk.benchmarking.semantics import (
    apply_mode_profile,
    attach_primary_metrics,
    canonical_mode_name,
)
from frisk.reporting.metrics import primary_metric_from_row
from frisk.training.objectives import (
    compute_multi_horizon_risk_loss,
    regression_eval_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_mode_name_maps_legacy_backprop_alias():
    assert canonical_mode_name("backprop") == "backprop_contrastive"


def test_apply_mode_profile_sets_fast_profile_contract():
    mode, cfg = apply_mode_profile("ff_fast", {"noise_std": 0.09})
    assert mode == "ff_fast"
    assert cfg["task_family"] == "algorithmic_parity"
    assert cfg["signal_family"] == "contrastive"
    assert cfg["amp"] is True
    assert cfg["torch_compile"] is True
    assert cfg["ff_concat_posneg"] is True


def test_attach_primary_metrics_uses_finance_first_for_supervised_return():
    row = {
        "task_family": "financial_value",
        "signal_family": "return_forecast",
        "eval_objective": "supervised_return",
        "econ_oos_sharpe_uplift_min": 0.18,
        "eval_return_corr": 0.41,
    }
    attach_primary_metrics(row)
    assert row["primary_eval_metric_name"] == "econ_oos_sharpe_uplift_min"
    assert row["primary_metric_family"] == "economics"


def test_compute_multi_horizon_risk_loss_shared_module_accepts_weighted_inputs():
    risk_head = torch.nn.Linear(4, 2)
    embeddings = torch.randn(3, 4)
    graph_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    risk_targets = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    sample_weight = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
    loss = compute_multi_horizon_risk_loss(
        risk_head=risk_head,
        embeddings=embeddings,
        graph_idx=graph_idx,
        risk_targets_by_horizon=risk_targets,
        device=torch.device("cpu"),
        risk_loss_type="huber",
        sample_weight=sample_weight,
    )
    assert loss is not None
    assert torch.isfinite(loss)


def test_regression_eval_metrics_shared_module_reports_rank_and_linear_corr():
    out = regression_eval_metrics(
        pred=[0.1, 0.2, 0.3, 0.4],
        target=[0.11, 0.19, 0.31, 0.39],
    )
    assert out["eval_return_corr"] > 0.9
    assert out["eval_return_rank_corr"] > 0.9


def test_reporting_primary_metric_uses_shared_financial_contract():
    metric, value = primary_metric_from_row(
        {
            "task_family": "financial_value",
            "primary_metric_family": "economics",
            "econ_oos_sharpe_uplift_min": "0.22",
            "eval_return_corr": "0.35",
        }
    )
    assert metric == "econ_oos_sharpe_uplift_min"
    assert value == 0.22


def test_notebook_hygiene_strip_outputs_removes_outputs_and_execution_counts(tmp_path: Path):
    mod = _load_script("notebook_hygiene.py")
    notebook_path = tmp_path / "demo.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 7,
                        "metadata": {},
                        "outputs": [{"output_type": "stream", "name": "stdout", "text": ["hi\n"]}],
                        "source": ["print('hi')\n"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )

    assert mod.strip_outputs(notebook_path) is True
    stats = mod.notebook_stats(notebook_path)
    assert stats["total_outputs"] == 0
    payload = json.loads(notebook_path.read_text())
    assert payload["cells"][0]["execution_count"] is None
