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


def test_aggregate_fold_results_preserves_objective_metadata():
    mod = _load_script("benchmark_training.py")
    fold_rows = [
        {
            "mode": "ff_e2e",
            "row_type": "fold",
            "eval_objective": "ff",
            "objective_track": "critic",
            "primary_eval_metric_name": "eval_sep",
            "primary_eval_metric_robust_name": "eval_sep",
            "neg_mode_effective": "shuffle+noise",
            "eval_neg_mode_effective": "shuffle+noise",
            "risk_head_enabled_effective": True,
            "eval_sep": 0.2,
        },
        {
            "mode": "ff_e2e",
            "row_type": "fold",
            "eval_objective": "ff",
            "objective_track": "critic",
            "primary_eval_metric_name": "eval_sep",
            "primary_eval_metric_robust_name": "eval_sep",
            "neg_mode_effective": "shuffle+noise",
            "eval_neg_mode_effective": "shuffle+noise",
            "risk_head_enabled_effective": True,
            "eval_sep": 0.4,
        },
    ]

    agg = mod._aggregate_fold_results(fold_rows)
    assert agg["eval_objective"] == "ff"
    assert agg["objective_track"] == "critic"
    assert agg["primary_eval_metric_name"] == "eval_sep"
    assert agg["primary_eval_metric_robust_name"] == "eval_sep"
    assert agg["neg_mode_effective"] == "shuffle+noise"
    assert agg["eval_neg_mode_effective"] == "shuffle+noise"
    assert agg["risk_head_enabled_effective"] is True


def test_ff_sweep_aggregate_fold_rows_preserves_objective_metadata():
    mod = _load_script("ff_sweep.py")
    fold_rows = [
        {
            "mode": "ff_e2e",
            "row_type": "fold",
            "eval_objective": "ff",
            "objective_track": "critic",
            "primary_eval_metric_name": "eval_sep",
            "primary_eval_metric_robust_name": "eval_sep",
            "neg_mode_effective": "shuffle+noise",
            "eval_neg_mode_effective": "shuffle+noise",
            "risk_head_enabled_effective": True,
            "eval_sep": 0.2,
        },
        {
            "mode": "ff_e2e",
            "row_type": "fold",
            "eval_objective": "ff",
            "objective_track": "critic",
            "primary_eval_metric_name": "eval_sep",
            "primary_eval_metric_robust_name": "eval_sep",
            "neg_mode_effective": "shuffle+noise",
            "eval_neg_mode_effective": "shuffle+noise",
            "risk_head_enabled_effective": True,
            "eval_sep": 0.4,
        },
    ]

    agg = mod._aggregate_fold_rows(fold_rows)
    assert agg["eval_objective"] == "ff"
    assert agg["objective_track"] == "critic"
    assert agg["primary_eval_metric_name"] == "eval_sep"
    assert agg["primary_eval_metric_robust_name"] == "eval_sep"
    assert agg["neg_mode_effective"] == "shuffle+noise"
    assert agg["eval_neg_mode_effective"] == "shuffle+noise"
    assert agg["risk_head_enabled_effective"] is True


def test_baseline_context_includes_seed_split_device_and_sizes():
    mod = _load_script("benchmark_training.py")
    cfg = {"seed": 11, "split_mode": "chronological", "batch_size": 32, "eval_frac": 0.2}
    ctx = mod._baseline_context(cfg, torch.device("cpu"), num_graphs=123)
    assert ctx["baseline_seed"] == 11
    assert ctx["baseline_split_mode"] == "chronological"
    assert ctx["baseline_device"] == "cpu"
    assert ctx["baseline_graphs_total"] == 123
    assert ctx["baseline_batch_size"] == 32


def test_make_negatives_timing_tracks_neg_generation():
    mod = _load_script("benchmark_training.py")
    x = torch.randn(6, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]], dtype=torch.long)
    hall_cfg = mod.HallucinationConfig()
    timing = {}
    x_neg = mod._make_negatives(
        model=None,
        x=x,
        batch=batch,
        edge_index=edge_index,
        edge_attr=None,
        edge_weight=None,
        use_mode="shuffle",
        noise_std=0.05,
        hall_cfg=hall_cfg,
        window_len=2,
        summary_dim=2,
        timing=timing,
    )
    assert x_neg.shape == x.shape
    assert timing.get("neg_gen", 0.0) > 0.0
