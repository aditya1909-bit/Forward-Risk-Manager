from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from torch_geometric.data import Data


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


def test_select_calibrated_goodness_target_falls_back_for_all_nan():
    mod = _load_script("benchmark_training.py")
    out = mod._select_calibrated_goodness_target(
        torch.tensor([float("nan"), float("inf")]),
        torch.tensor([float("nan"), float("-inf")]),
        default_target=2.0,
        quantiles=7,
    )

    assert out["target"] == 2.0
    assert out["valid"] == 0
    assert out["num_pos"] == 0
    assert out["num_neg"] == 0
    assert out["reason"] == "insufficient_finite_goodness"


def test_select_calibrated_goodness_target_reports_valid_diagnostics():
    mod = _load_script("benchmark_training.py")
    out = mod._select_calibrated_goodness_target(
        torch.tensor([2.0, 3.0, float("nan")]),
        torch.tensor([-1.0, 0.0, float("inf")]),
        default_target=1.0,
        quantiles=9,
    )

    assert torch.isfinite(torch.tensor(float(out["target"])))
    assert out["valid"] == 1
    assert out["num_pos"] == 2
    assert out["num_neg"] == 2
    assert out["reason"] == "ok"
    assert float(out["acc"]) >= 0.5


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


def test_ff_sweep_econ_metrics_passes_regime_threshold_controls():
    mod = _load_script("ff_sweep.py")
    eval_dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    fwd = pd.Series(
        [0.01, -0.02, 0.015, -0.005],
        index=pd.to_datetime(eval_dates),
    )

    cfg = {
        "econ_enabled": True,
        "econ_fwd_ret_1": fwd,
        "econ_signal_window": 63,
        "econ_signal_quantile": 0.55,
        "econ_signal_polarity": "low",
        "econ_oos_folds": 5,
        "econ_oos_min_fold_days": 21,
        "econ_regime_thresholding_enabled": True,
        "econ_regime_threshold_window": 42,
        "econ_regime_threshold_quantile": 0.6,
        "econ_regime_vol_window": 11,
        "econ_regime_low_quantile": 0.2,
        "econ_regime_high_quantile": 0.8,
        "econ_ticker": "AUTO",
        "econ_ticker_effective": "SPY",
        "econ_ticker_source": "auto",
        "econ_ticker_rows": 1024,
    }

    captured: dict[str, object] = {}

    def _fake_eval(dates, scores, fwd_ret_1, **kwargs):
        captured.update(kwargs)
        assert len(dates) == len(scores) == len(eval_dates)
        assert fwd_ret_1 is fwd
        return {
            "econ_regime_thresholding_enabled": float(kwargs["regime_thresholding_enabled"]),
            "econ_regime_threshold_window": float(kwargs["regime_threshold_window"]),
            "econ_regime_threshold_quantile": float(kwargs["regime_threshold_quantile"]),
            "econ_regime_vol_window": float(kwargs["regime_vol_window"]),
            "econ_regime_low_count": 1.0,
            "econ_regime_mid_count": 2.0,
            "econ_regime_high_count": 1.0,
        }

    orig_eval = mod.evaluate_goodness_strategy
    try:
        mod.evaluate_goodness_strategy = _fake_eval
        out = mod._compute_econ_metrics_for_eval(
            None,
            None,
            [],
            eval_dates,
            cfg,
            goodness_values=[0.2, 0.4, 0.6, 0.8],
        )
    finally:
        mod.evaluate_goodness_strategy = orig_eval

    assert captured["regime_thresholding_enabled"] is True
    assert int(captured["regime_threshold_window"]) == 42
    assert float(captured["regime_threshold_quantile"]) == 0.6
    assert str(captured["signal_polarity"]) == "low"
    assert int(captured["oos_folds"]) == 5
    assert int(captured["oos_min_fold_days"]) == 21
    assert int(captured["regime_vol_window"]) == 11
    assert float(captured["regime_low_quantile"]) == 0.2
    assert float(captured["regime_high_quantile"]) == 0.8

    # Metadata + thresholded econ outputs should both be present.
    assert out["econ_ticker_requested"] == "AUTO"
    assert out["econ_ticker_effective"] == "SPY"
    assert out["econ_ticker_source"] == "auto"
    assert float(out["econ_ticker_rows"]) == 1024.0
    assert float(out["econ_regime_thresholding_enabled"]) == 1.0
    assert float(out["econ_regime_threshold_window"]) == 42.0
    assert float(out["econ_regime_threshold_quantile"]) == 0.6
    assert float(out["econ_regime_vol_window"]) == 11.0


def test_ff_sweep_finance_rank_prefers_oos_metrics():
    mod = _load_script("ff_sweep.py")
    row = {
        "eval_objective": "ff",
        "econ_sharpe_uplift": 0.42,
        "econ_oos_sharpe_uplift_min": 0.11,
        "econ_oos_sharpe_uplift_mean": 0.2,
    }
    metric, value = mod._objective_rank_metric(row, rank_mode="finance_first")
    assert metric == "econ_oos_sharpe_uplift_min"
    assert float(value) == 0.11


def test_ff_sweep_finance_floor_metric_falls_back_from_oos_to_in_sample():
    mod = _load_script("ff_sweep.py")
    metric_1, value_1 = mod._finance_floor_metric(
        {
            "econ_oos_sharpe_uplift_min": 0.15,
            "econ_sharpe_uplift_min": 0.25,
        }
    )
    assert metric_1 == "econ_oos_sharpe_uplift_min"
    assert float(value_1) == 0.15

    metric_2, value_2 = mod._finance_floor_metric(
        {
            "econ_sharpe_uplift_min": 0.25,
            "econ_sharpe_uplift": 0.3,
        }
    )
    assert metric_2 == "econ_sharpe_uplift_min"
    assert float(value_2) == 0.25


def test_apply_mode_profile_sets_algorithmic_parity_core_defaults():
    mod = _load_script("benchmark_training.py")
    mode, cfg = mod._apply_mode_profile("ff_e2e_core", {"noise_std": 0.07})
    assert mode == "ff_e2e_core"
    assert cfg["task_family"] == "algorithmic_parity"
    assert cfg["signal_family"] == "contrastive"
    assert cfg["risk_head_enabled"] is False
    assert cfg["portfolio_head_enabled"] is False
    assert cfg["ff_loss_type"] == "softplus_margin"
    assert cfg["goodness_norm"] == "none"
    assert cfg["goodness_reducer"] == "logsumexp"


def test_apply_mode_profile_sets_bootstrap_rank_finance_defaults():
    mod = _load_script("benchmark_training.py")
    mode, cfg = mod._apply_mode_profile("ff_bootstrap_rank", {"hidden_dim": 32})
    assert mode == "ff_bootstrap_rank"
    assert cfg["task_family"] == "financial_value"
    assert cfg["signal_family"] == "goodness_rank"
    assert cfg["ff_loss_type"] == "symba"
    assert cfg["goodness_norm"] == "layernorm"
    assert cfg["goodness_reducer"] == "mean"
    assert cfg["bootstrap_graph_enabled"] is True
    assert float(cfg["bootstrap_graph_weight"]) > 0.0
    assert str(cfg["econ_strategy_kind"]) == "both"


def test_attach_primary_metrics_prefers_economics_for_financial_track():
    mod = _load_script("benchmark_training.py")
    row = {
        "task_family": "financial_value",
        "signal_family": "return_forecast",
        "eval_objective": "supervised_return",
        "econ_oos_sharpe_uplift_min": 0.12,
        "eval_return_corr": 0.35,
    }
    mod._attach_primary_metrics(row)
    assert row["primary_eval_metric_name"] == "econ_oos_sharpe_uplift_min"
    assert float(row["primary_eval_metric"]) == 0.12
    assert row["primary_metric_family"] == "economics"


def test_attach_primary_metrics_prefers_long_short_metric_for_goodness_rank():
    mod = _load_script("benchmark_training.py")
    row = {
        "task_family": "financial_value",
        "signal_family": "goodness_rank",
        "eval_objective": "ff",
        "econ_ls_oos_sharpe_uplift_min": 0.08,
        "econ_oos_sharpe_uplift_min": 0.02,
        "eval_sep": 0.5,
    }
    mod._attach_primary_metrics(row)
    assert row["primary_eval_metric_name"] == "econ_ls_oos_sharpe_uplift_min"
    assert float(row["primary_eval_metric"]) == 0.08


def test_regression_eval_metrics_reports_finite_regression_outputs():
    mod = _load_script("benchmark_training.py")
    out = mod._regression_eval_metrics(
        pred=[0.1, 0.2, 0.3, 0.4],
        target=[0.12, 0.18, 0.31, 0.38],
    )
    assert out["eval_return_mse"] >= 0.0
    assert out["eval_return_mae"] >= 0.0
    assert out["eval_return_corr"] > 0.9
    assert out["eval_return_rank_corr"] > 0.9


def test_ff_sweep_build_econ_payload_falls_back_to_inference(monkeypatch):
    mod = _load_script("ff_sweep.py")
    eval_dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    eval_graphs = [object(), object(), object()]
    cfg = {
        "econ_enabled": False,
        "econ_payload_enabled": True,
        "goodness_temp": 0.42,
        "batch_size": 8,
        "econ_loader_batch_size": 16,
    }
    called: dict[str, object] = {}

    def _fake_infer(model, graphs, goodness_temp, batch_size=128, critic=None):
        import numpy as np

        called["model"] = model
        called["graphs_len"] = len(graphs)
        called["goodness_temp"] = goodness_temp
        called["batch_size"] = batch_size
        called["critic"] = critic
        return np.asarray([0.11, 0.22, 0.33], dtype=float), None

    monkeypatch.setattr(mod, "infer_graph_goodness_with_uncertainty", _fake_infer)
    dummy_model = object()

    goodness_vals, payloads = mod._build_econ_payload(
        model=dummy_model,
        critic=None,
        eval_graphs=eval_graphs,
        eval_dates=eval_dates,
        cfg=cfg,
        eval_pos_cache=None,
    )

    assert called["model"] is dummy_model
    assert int(called["graphs_len"]) == 3
    assert float(called["goodness_temp"]) == 0.42
    assert int(called["batch_size"]) == 16
    assert goodness_vals == [0.11, 0.22, 0.33]
    assert payloads == [{"goodness": [0.11, 0.22, 0.33], "dates": eval_dates}]


def test_ff_sweep_build_econ_payload_prefers_cached_gpos(monkeypatch):
    mod = _load_script("ff_sweep.py")
    eval_dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    eval_pos_cache = {"g_pos_all": [0.4, 0.5, 0.6]}

    def _fail_infer(*_args, **_kwargs):
        raise AssertionError("fallback inference should not run when g_pos_all cache is present")

    monkeypatch.setattr(mod, "infer_graph_goodness_with_uncertainty", _fail_infer)
    goodness_vals, payloads = mod._build_econ_payload(
        model=object(),
        critic=None,
        eval_graphs=[object(), object(), object()],
        eval_dates=eval_dates,
        cfg={"econ_enabled": False, "econ_payload_enabled": True},
        eval_pos_cache=eval_pos_cache,
    )
    assert goodness_vals == [0.4, 0.5, 0.6]
    assert payloads == [{"goodness": [0.4, 0.5, 0.6], "dates": eval_dates}]


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


def test_compile_mode_candidates_prefer_no_cudagraphs_for_reduce_overhead():
    mod = _load_script("benchmark_training.py")
    cands_cuda = mod._compile_mode_candidates("reduce-overhead", torch.device("cuda"))
    assert cands_cuda[0] == "max-autotune-no-cudagraphs"
    assert cands_cuda[1] == "reduce-overhead"
    assert cands_cuda[-1] == "default"

    cands_cpu = mod._compile_mode_candidates("reduce-overhead", torch.device("cpu"))
    assert cands_cpu[0] == "reduce-overhead"
    assert cands_cpu[-1] == "default"


def test_maybe_compile_encoder_disables_after_fallback_failures():
    mod = _load_script("benchmark_training.py")
    model = torch.nn.Linear(4, 2)
    cfg = {"torch_compile": True, "torch_compile_mode": "reduce-overhead"}
    orig_compile = getattr(mod.torch, "compile", None)
    if orig_compile is None:
        return
    try:
        mod.torch.compile = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("compile fail"))
        out_model = mod._maybe_compile_encoder(model, cfg, torch.device("cpu"), context="unit-test")
    finally:
        mod.torch.compile = orig_compile
    assert out_model is model
    assert cfg["torch_compile"] is False


def test_backprop_concat_posneg_runs_and_returns_finite_metrics():
    mod = _load_script("benchmark_training.py")
    graphs = []
    for i in range(8):
        x = torch.randn(5, 6)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 1, 2], [1, 2, 3, 4, 0, 0, 4]],
            dtype=torch.long,
        )
        g = Data(x=x, edge_index=edge_index)
        g.graph_idx = i
        graphs.append(g)

    cfg = {
        "eval_frac": 0.25,
        "seed": 7,
        "split_mode": "chronological",
        "batch_size": 2,
        "loader_workers": 0,
        "hidden_dim": 8,
        "num_layers": 2,
        "dropout": 0.1,
        "encoder_conv_type": "gcn",
        "encoder_gat_heads": 2,
        "residual_edge_weight_enabled": False,
        "residual_edge_hidden_dim": 16,
        "residual_edge_max_delta": 0.25,
        "residual_edge_detach_features": True,
        "torch_compile": False,
        "torch_compile_mode": "max-autotune-no-cudagraphs",
        "lr": 1e-3,
        "backprop_fused_optimizer": False,
        "backprop_amp_dtype": "float16",
        "backprop_amp": False,
        "hall_steps": 1,
        "hall_lr": 0.01,
        "hall_l2": 0.01,
        "hall_mean": 0.0,
        "hall_std": 0.0,
        "hall_corr": 0.0,
        "hall_clamp": 3.0,
        "goodness_temp": 1.0,
        "hall_node_fraction": 1.0,
        "hall_node_min": 1,
        "hall_corr_every_n_steps": 1,
        "hall_corr_edge_fraction": 1.0,
        "hall_corr_edge_min": 1,
        "hall_adaptive_lr": False,
        "hall_adaptive_lr_patience": 2,
        "hall_adaptive_lr_decay": 0.5,
        "hall_adaptive_lr_min": 1e-4,
        "hall_early_stop_on_target_hit": False,
        "hall_target_hit_patience": 1,
        "hall_moment_mean": 0.0,
        "hall_moment_var": 0.0,
        "hall_moment_skew": 0.0,
        "hall_moment_scope": "returns",
        "hall_attack_hub_fraction": 0.2,
        "hall_attack_noise_mult": 3.0,
        "hall_attack_timeflip_prob": 0.5,
        "hall_attack_edge_drop_prob": 0.2,
        "hall_attack_sign_flip_prob": 0.2,
        "hall_attack_hub_weight_scale": 0.5,
        "neg_mode": "shuffle",
        "eval_neg_mode": "shuffle",
        "ff_hall_every_n_batches": 1,
        "ff_hall_warmup_epochs": 0,
        "epochs": 1,
        "noise_std": 0.01,
        "neg_warmup_epochs": 0,
        "neg_mix_start": 0.0,
        "neg_mix_end": 0.0,
        "neg_mix_ramp_epochs": 1,
        "window_len": 4,
        "summary_dim": 0,
        "grad_clip": 1.0,
        "risk_head_enabled": False,
        "risk_targets_by_horizon": None,
        "risk_horizons_effective": [],
        "portfolio_head_enabled": False,
        "portfolio_targets": None,
        "timing_warmup_epochs": 0,
        "ece_bins": 5,
        "econ_enabled": False,
        "backprop_concat_posneg": True,
    }
    out = mod._benchmark_backprop(
        graphs,
        torch.device("cpu"),
        cfg,
        train_graphs=graphs[:6],
        eval_graphs=graphs[6:],
        eval_dates=[],
    )
    assert out["eval_objective"] == "bce"
    assert torch.isfinite(torch.tensor(float(out["eval_bce"])))
    assert torch.isfinite(torch.tensor(float(out["avg_epoch_s"])))


def test_retry_safe_and_continue_records_failed_mode_row_and_keeps_success():
    mod = _load_script("benchmark_training.py")
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        graphs_path = tmp_path / "graphs.pt"
        payload_graphs = []
        for i in range(6):
            g = Data(
                x=torch.randn(4, 5),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            )
            g.graph_idx = i
            payload_graphs.append(g)
        torch.save({"graphs": payload_graphs, "dates": []}, graphs_path)

        out_csv = tmp_path / "benchmark.csv"
        cfg_path = tmp_path / "cfg.toml"
        cfg_path.write_text(
            "\n".join(
                [
                    "[train]",
                    f"graphs = \"{graphs_path}\"",
                    "device = \"cpu\"",
                    "seed = 7",
                    "risk_head_enabled = true",
                    "portfolio_head_enabled = true",
                    "",
                    "[build_graphs]",
                    "feature_mode = \"window\"",
                    "window = 5",
                    "",
                    "[benchmark]",
                    f"out_csv = \"{out_csv}\"",
                    "split_mode = \"chronological\"",
                    "eval_frac = 0.3",
                    "epochs = 1",
                    "retry_safe_on_error = true",
                    "continue_on_mode_error = true",
                    "enable_risk_head = false",
                    "enable_portfolio_head = false",
                    "econ_enabled = false",
                    "torch_compile = false",
                ]
            ),
            encoding="utf-8",
        )

        def _ff_fail(*_args, **_kwargs):
            raise RuntimeError("forced ff failure")

        def _bp_ok(_graphs, _device, cfg_run, **_kwargs):
            assert cfg_run["risk_head_enabled"] is False
            assert cfg_run["portfolio_head_enabled"] is False
            return {
                "avg_epoch_s": 1.0,
                "graphs_per_s": 2.0,
                "eval_objective": "bce",
                "eval_sep": 0.1,
                "eval_auroc": 0.6,
                "eval_auprc": 0.5,
            }

        orig_ff = mod._benchmark_ff
        orig_bp = mod._benchmark_backprop
        orig_argv = sys.argv[:]
        try:
            mod._benchmark_ff = _ff_fail
            mod._benchmark_backprop = _bp_ok
            sys.argv = [
                "benchmark_training.py",
                "--config",
                str(cfg_path),
                "--modes",
                "ff_layerwise,backprop",
            ]
            rc = mod.main()
        finally:
            mod._benchmark_ff = orig_ff
            mod._benchmark_backprop = orig_bp
            sys.argv = orig_argv

        assert rc == 0
        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert {"status", "error_type", "error_message", "retry_applied", "safe_mode_applied"}.issubset(
            set(df.columns)
        )
        fail = df[df["mode"] == "ff_layerwise"].iloc[0]
        ok = df[df["mode"] == "backprop_contrastive"].iloc[0]
        assert str(fail["status"]) == "failed"
        assert str(ok["status"]) == "ok"


def test_aggregate_fold_results_skips_failed_folds_in_metrics():
    mod = _load_script("benchmark_training.py")
    agg = mod._aggregate_fold_results(
        [
            {
                "mode": "backprop",
                "row_type": "fold",
                "status": "failed",
                "error_type": "RuntimeError",
                "error_message": "boom",
                "eval_sep": -10.0,
                "retry_applied": True,
                "safe_mode_applied": True,
            },
            {
                "mode": "backprop",
                "row_type": "fold",
                "status": "ok",
                "eval_objective": "bce",
                "objective_track": "classifier",
                "primary_eval_metric_name": "eval_auroc",
                "primary_eval_metric_robust_name": "eval_auroc",
                "eval_sep": 0.25,
                "retry_applied": False,
                "safe_mode_applied": False,
            },
        ]
    )
    assert agg["status"] == "ok"
    assert abs(float(agg["eval_sep"]) - 0.25) < 1e-8
    assert int(agg["walk_forward_num_folds"]) == 1
    assert int(agg["walk_forward_num_failed_folds"]) == 1


def test_aggregate_seed_results_reports_bootstrap_ci():
    mod = _load_script("benchmark_training.py")
    agg = mod._aggregate_seed_results(
        [
            {"mode": "ff_e2e", "status": "ok", "eval_sep": 0.2, "graphs_per_s": 10.0},
            {"mode": "ff_e2e", "status": "ok", "eval_sep": 0.4, "graphs_per_s": 12.0},
            {"mode": "ff_e2e", "status": "failed", "error_type": "RuntimeError"},
        ],
        bootstrap_samples=200,
        bootstrap_alpha=0.1,
        bootstrap_seed=13,
    )
    assert agg["status"] == "ok"
    assert int(agg["seed_num_runs"]) == 2
    assert int(agg["seed_num_failed_runs"]) == 1
    assert abs(float(agg["eval_sep"]) - 0.3) < 1e-8
    assert float(agg["eval_sep_ci_lo"]) <= float(agg["eval_sep"]) <= float(agg["eval_sep_ci_hi"])
