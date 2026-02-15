from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_positive_int_list_filters_invalid_values():
    mod = _load_script("benchmark_training.py")
    vals = mod._parse_positive_int_list("21, 0, -3, 42, 21, bad", fallback=7)
    assert vals == [21, 42]


def test_compute_multi_horizon_risk_loss_returns_finite_value():
    mod = _load_script("benchmark_training.py")
    risk_head = torch.nn.Linear(4, 2)
    embeddings = torch.randn(3, 4)
    graph_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    risk_targets = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    loss = mod._compute_multi_horizon_risk_loss(
        risk_head=risk_head,
        embeddings=embeddings,
        graph_idx=graph_idx,
        risk_targets_by_horizon=risk_targets,
        device=torch.device("cpu"),
        risk_loss_type="huber",
    )
    assert loss is not None
    assert torch.isfinite(loss)


def test_compute_multi_horizon_risk_loss_returns_none_for_all_missing_targets():
    mod = _load_script("benchmark_training.py")
    risk_head = torch.nn.Linear(4, 1)
    embeddings = torch.randn(2, 4)
    graph_idx = torch.tensor([0, 1], dtype=torch.long)
    risk_targets = [[None, None]]
    loss = mod._compute_multi_horizon_risk_loss(
        risk_head=risk_head,
        embeddings=embeddings,
        graph_idx=graph_idx,
        risk_targets_by_horizon=risk_targets,
        device=torch.device("cpu"),
        risk_loss_type="mse",
    )
    assert loss is None
