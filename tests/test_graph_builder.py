import numpy as np
import pandas as pd
import torch

from frisk.graph_builder import (
    GraphBuildConfig,
    _build_node_features,
    _select_edges,
    _safe_corr_matrix,
    build_rolling_corr_graphs,
)


def test_safe_corr_matrix_handles_constant_column():
    df = pd.DataFrame(
        {
            "A": [0.0, 0.1, 0.2, 0.1],
            "B": [0.0, 0.0, 0.0, 0.0],
            "C": [0.2, 0.2, -0.1, 0.0],
        }
    )
    corr = _safe_corr_matrix(df)
    assert corr.shape == (3, 3)
    assert np.allclose(corr, corr.T, atol=1e-6)
    # Constant column should yield zeros in its row/col
    assert np.allclose(corr[1], 0.0, atol=1e-6)
    assert np.allclose(corr[:, 1], 0.0, atol=1e-6)


def test_build_node_features_window_plus_summary_fund():
    window_df = pd.DataFrame(
        {
            "MDY": [0.01, -0.02, 0.03, 0.01],
            "AAA": [0.02, 0.01, -0.01, 0.0],
            "BBB": [-0.01, 0.0, 0.02, -0.02],
        }
    )
    volume_df = pd.DataFrame(
        {
            "MDY": [100, 105, 110, 98],
            "AAA": [200, 210, 190, 205],
            "BBB": [150, 155, 160, 158],
        }
    )
    fund = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
        ]
    )

    x, ret_mean, ret_std = _build_node_features(
        window_df,
        volume_df,
        feature_mode="window_plus_summary_fund",
        normalize=True,
        cross_sectional_norm=False,
        mdy_ticker="MDY",
        rsi_period=14,
        fund_features=fund,
    )
    # window_len=4, summary=5, fund=5 => 14 features
    assert x.shape == (3, 14)
    assert ret_mean is not None
    assert ret_std is not None
    assert ret_mean.shape == (3, 1)
    assert ret_std.shape == (3, 1)


def test_build_node_features_auto_market_proxy_beta_is_finite():
    window_df = pd.DataFrame(
        {
            "AAA": [0.01, -0.02, 0.02, 0.00],
            "BBB": [0.03, -0.01, 0.01, -0.01],
            "CCC": [-0.02, 0.01, 0.00, 0.02],
        }
    )
    x, _, _ = _build_node_features(
        window_df,
        volume_df=None,
        feature_mode="window_plus_summary",
        normalize=False,
        cross_sectional_norm=False,
        mdy_ticker="AUTO",
        rsi_period=14,
        fund_features=None,
    )
    # window=4 + summary=5 ; beta is summary slot index 3 => absolute index 7
    beta = x[:, 7]
    assert np.isfinite(beta).all()
    assert np.std(beta) > 0


def test_build_rolling_corr_graphs_basic():
    dates = pd.date_range("2020-01-01", periods=5, freq="D").strftime("%Y-%m-%d")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.0, -0.01, 0.03],
            "BBB": [0.0, -0.01, 0.02, 0.01, -0.02],
            "CCC": [0.02, 0.0, -0.01, 0.03, 0.01],
        },
        index=dates,
    )
    volume = pd.DataFrame(
        {
            "AAA": [100, 110, 105, 98, 120],
            "BBB": [200, 195, 205, 210, 190],
            "CCC": [150, 160, 155, 165, 158],
        },
        index=dates,
    )
    membership = {d: ["AAA", "BBB", "CCC"] for d in dates}
    cfg = GraphBuildConfig(
        window=3,
        step=1,
        top_k=1,
        corr_threshold=None,
        min_nodes=2,
        feature_mode="window_plus_summary",
        normalize=True,
        symmetric=True,
        rsi_period=14,
        mdy_ticker="AAA",
        edge_norm=False,
        edge_weight_mode="abs",
    )
    graphs, graph_dates, tickers, stats = build_rolling_corr_graphs(
        returns,
        volume,
        membership,
        cfg,
        fundamentals=None,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    assert stats["built"] == 3
    assert len(graphs) == 3
    assert len(graph_dates) == 3
    assert graphs[0].x.shape[0] == 3
    assert graphs[0].edge_index.shape[1] > 0


def test_build_rolling_corr_graphs_signed_edge_norm_is_finite():
    dates = pd.date_range("2020-01-01", periods=5, freq="D").strftime("%Y-%m-%d")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, -0.01, 0.01, -0.01, 0.01],
            "BBB": [-0.01, 0.01, -0.01, 0.01, -0.01],
            "CCC": [0.005, 0.002, -0.003, 0.004, -0.002],
        },
        index=dates,
    )
    membership = {d: ["AAA", "BBB", "CCC"] for d in dates}
    cfg = GraphBuildConfig(
        window=5,
        step=1,
        top_k=1,
        corr_threshold=None,
        min_nodes=2,
        feature_mode="window",
        normalize=True,
        symmetric=True,
        rsi_period=14,
        mdy_ticker="AAA",
        edge_norm=True,
        edge_weight_mode="raw",
    )
    graphs, _, _, stats = build_rolling_corr_graphs(
        returns,
        None,
        membership,
        cfg,
        fundamentals=None,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    assert stats["built"] == 1
    edge_weight = graphs[0].edge_weight
    assert torch.isfinite(edge_weight).all()
    assert (edge_weight < 0).any()


def test_build_rolling_corr_graphs_with_lags_avoids_lookahead():
    dates = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "BBB": [0.03, 0.02, 0.01, 0.00, -0.01, -0.02],
            "CCC": [0.00, -0.01, -0.02, -0.03, -0.04, -0.05],
        },
        index=dates,
    )
    membership = {d: ["AAA", "BBB", "CCC"] for d in dates}
    cfg = GraphBuildConfig(
        window=3,
        step=1,
        corr_lag_days=1,
        feature_lag_days=1,
        membership_lag_days=1,
        top_k=1,
        corr_threshold=None,
        min_nodes=2,
        feature_mode="last",
        normalize=False,
        symmetric=True,
        rsi_period=14,
        mdy_ticker="AAA",
        edge_norm=False,
        edge_weight_mode="raw",
    )
    graphs, graph_dates, tickers, stats = build_rolling_corr_graphs(
        returns,
        None,
        membership,
        cfg,
        fundamentals=None,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    # 4 candidate windows (end_idx=2..5), first is invalid due to lag+window history.
    assert stats["total_windows"] == 4
    assert stats["skipped_lag_history"] == 1
    assert stats["built"] == 3
    assert graph_dates[0] == dates[3]

    first_tickers = tickers[0]
    x_last = graphs[0].x.squeeze(1).numpy()
    expected_date = dates[2]  # feature lag of 1 day for graph date dates[3]
    expected = np.array([returns.loc[expected_date, t] for t in first_tickers], dtype=float)
    assert np.allclose(x_last, expected, atol=1e-8)


def test_build_node_features_cross_sectional_norm_zero_centered_by_date():
    window_df = pd.DataFrame(
        {
            "AAA": [1.0, 3.0, 5.0],
            "BBB": [2.0, 4.0, 6.0],
        }
    )
    x, _, _ = _build_node_features(
        window_df,
        volume_df=None,
        feature_mode="window",
        normalize=False,
        cross_sectional_norm=True,
        mdy_ticker="AUTO",
        rsi_period=14,
        fund_features=None,
    )
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-8)


def test_select_edges_significance_filters_weak_links():
    corr = np.array(
        [
            [0.0, 0.9, 0.1],
            [0.9, 0.0, 0.02],
            [0.1, 0.02, 0.0],
        ]
    )
    src, dst, _ = _select_edges(
        corr,
        top_k=None,
        corr_threshold=None,
        symmetric=True,
        mode="significance",
        significance_alpha=0.05,
        n_obs=120,
    )
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert (0, 1) in pairs
    assert (1, 0) in pairs
    assert (0, 2) not in pairs


def test_build_rolling_corr_graphs_supports_macro_and_edge_node_weighting():
    dates = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.01, -0.01, 0.00, 0.01],
            "BBB": [0.00, 0.01, 0.02, 0.01, -0.01, -0.02],
            "CCC": [0.02, 0.01, 0.00, -0.01, 0.00, 0.01],
        },
        index=dates,
    )
    volume = pd.DataFrame(
        {
            "AAA": [100, 100, 100, 100, 100, 100],
            "BBB": [300, 300, 300, 300, 300, 300],
            "CCC": [50, 50, 50, 50, 50, 50],
        },
        index=dates,
    )
    membership = {d: ["AAA", "BBB", "CCC"] for d in dates}
    macro = pd.DataFrame(
        {
            "m1": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "m2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        },
        index=dates,
    )
    cfg_none = GraphBuildConfig(
        window=3,
        step=1,
        top_k=1,
        min_nodes=2,
        feature_mode="window_plus_summary",
        normalize=True,
        cross_sectional_norm=True,
        edge_node_weighting="none",
    )
    cfg_weighted = GraphBuildConfig(
        window=3,
        step=1,
        top_k=1,
        min_nodes=2,
        feature_mode="window_plus_summary",
        normalize=True,
        cross_sectional_norm=True,
        edge_node_weighting="volume",
        edge_node_weight_power=0.5,
    )
    g_none, _, _, _ = build_rolling_corr_graphs(
        returns,
        volume,
        membership,
        cfg_none,
        fundamentals=None,
        macro=macro,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    g_w, _, _, _ = build_rolling_corr_graphs(
        returns,
        volume,
        membership,
        cfg_weighted,
        fundamentals=None,
        macro=macro,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    # window(3) + summary(5) + macro(2)
    assert g_w[0].x.shape[1] == 10
    assert torch.abs(g_w[0].x[:, -2:]).sum() > 0
    assert not torch.allclose(g_none[0].edge_attr, g_w[0].edge_attr)


def test_build_rolling_corr_graphs_macro_lag_days_uses_past_macro_row():
    dates = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.01, -0.01, 0.00, 0.01],
            "BBB": [0.00, 0.01, 0.02, 0.01, -0.01, -0.02],
            "CCC": [0.02, 0.01, 0.00, -0.01, 0.00, 0.01],
        },
        index=dates,
    )
    membership = {d: ["AAA", "BBB", "CCC"] for d in dates}
    macro = pd.DataFrame({"m1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}, index=dates)
    cfg = GraphBuildConfig(
        window=3,
        step=1,
        top_k=1,
        min_nodes=2,
        feature_mode="last",
        normalize=False,
        macro_lag_days=1,
    )
    graphs, graph_dates, _, stats = build_rolling_corr_graphs(
        returns,
        None,
        membership,
        cfg,
        fundamentals=None,
        macro=macro,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    assert stats["built"] >= 1
    assert graph_dates[0] == dates[2]
    expected_macro = float(macro.loc[dates[1], "m1"])
    assert np.allclose(graphs[0].x[:, -1].numpy(), expected_macro, atol=1e-8)


def test_build_rolling_corr_graphs_partial_corr_mode_builds_graphs():
    dates = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.03, 0.02, 0.01, 0.00],
            "BBB": [0.00, 0.01, 0.01, 0.00, -0.01, -0.02],
            "CCC": [0.02, 0.02, 0.01, 0.01, 0.00, -0.01],
        },
        index=dates,
    )
    membership = {d: ["AAA", "BBB", "CCC"] for d in dates}
    cfg = GraphBuildConfig(
        window=4,
        step=1,
        top_k=1,
        min_nodes=2,
        feature_mode="window",
        normalize=True,
        corr_method="partial",
        partial_corr_ridge=1e-3,
        edge_select_mode="top_k",
    )
    graphs, _, _, stats = build_rolling_corr_graphs(
        returns,
        None,
        membership,
        cfg,
        fundamentals=None,
        macro=None,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    assert stats["built"] >= 1
    assert torch.isfinite(graphs[0].edge_attr).all()


def test_build_rolling_corr_graphs_can_build_with_lead_lag_edges_only():
    dates = pd.date_range("2020-01-01", periods=8, freq="D").strftime("%Y-%m-%d")
    aaa = np.array([0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04, -0.04], dtype=float)
    bbb = np.concatenate([[0.0], aaa[:-1]])
    returns = pd.DataFrame(
        {
            "AAA": aaa,
            "BBB": bbb,
            "CCC": np.array([0.005, -0.004, 0.003, -0.002, 0.002, -0.001, 0.001, -0.001], dtype=float),
        },
        index=dates,
    )
    membership = {d: ["AAA", "BBB", "CCC"] for d in dates}
    cfg = GraphBuildConfig(
        window=6,
        step=1,
        top_k=None,
        corr_threshold=2.0,
        min_nodes=2,
        feature_mode="window",
        normalize=False,
        symmetric=False,
        lead_lag_enabled=True,
        lead_lag_max_lag=1,
        lead_lag_top_k=1,
        lead_lag_weight=1.0,
        lead_lag_mode="top_k",
    )
    graphs, _, tickers, stats = build_rolling_corr_graphs(
        returns,
        None,
        membership,
        cfg,
        fundamentals=None,
        macro=None,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    assert stats["built"] >= 1
    found_a_to_b = False
    for g, ts in zip(graphs, tickers):
        idx = {t: i for i, t in enumerate(ts)}
        if "AAA" not in idx or "BBB" not in idx:
            continue
        pairs = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
        if (idx["AAA"], idx["BBB"]) in pairs:
            found_a_to_b = True
            break
    assert found_a_to_b


def test_build_rolling_corr_graphs_supports_static_edge_overlay_without_corr_edges():
    dates = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.00, 0.01, 0.00, 0.01, 0.00],
            "BBB": [0.00, 0.01, 0.00, 0.01, 0.00, 0.01],
            "CCC": [0.02, -0.01, 0.02, -0.01, 0.02, -0.01],
        },
        index=dates,
    )
    membership = {d: ["AAA", "BBB", "CCC"] for d in dates}
    static_edges = pd.DataFrame(
        {
            "src": ["AAA"],
            "dst": ["CCC"],
            "weight": [2.0],
            "directed": [False],
        }
    )
    cfg = GraphBuildConfig(
        window=4,
        step=1,
        top_k=None,
        corr_threshold=2.0,
        min_nodes=2,
        feature_mode="window",
        normalize=True,
        symmetric=False,
        static_edge_weight=0.5,
    )
    graphs, _, tickers, stats = build_rolling_corr_graphs(
        returns,
        None,
        membership,
        cfg,
        fundamentals=None,
        macro=None,
        static_edges=static_edges,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    assert stats["built"] >= 1
    idx = {t: i for i, t in enumerate(tickers[0])}
    pairs = set(zip(graphs[0].edge_index[0].tolist(), graphs[0].edge_index[1].tolist()))
    assert (idx["AAA"], idx["CCC"]) in pairs
    assert (idx["CCC"], idx["AAA"]) in pairs


def test_sector_static_edges_work_even_when_feature_mode_is_not_fund():
    dates = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.00, 0.01, 0.00, 0.01, 0.00],
            "BBB": [0.01, 0.00, 0.01, 0.00, 0.01, 0.00],
            "CCC": [0.02, -0.01, 0.02, -0.01, 0.02, -0.01],
        },
        index=dates,
    )
    fundamentals = pd.DataFrame(
        {
            "date": [dates[-1], dates[-1], dates[-1]],
            "ticker": ["AAA", "BBB", "CCC"],
            "sector_code": [10, 10, 20],
        }
    )
    membership = {d: ["AAA", "BBB", "CCC"] for d in dates}
    cfg = GraphBuildConfig(
        window=4,
        step=1,
        top_k=None,
        corr_threshold=2.0,
        min_nodes=2,
        feature_mode="window",
        normalize=True,
        symmetric=False,
        sector_static_enabled=True,
        sector_static_weight=0.2,
        sector_static_top_k=2,
    )
    graphs, _, tickers, stats = build_rolling_corr_graphs(
        returns,
        None,
        membership,
        cfg,
        fundamentals=fundamentals,
        macro=None,
        static_edges=None,
        num_workers=1,
        parallel_backend="serial",
        progress=False,
    )
    assert stats["built"] >= 1
    idx = {t: i for i, t in enumerate(tickers[-1])}
    pairs = set(zip(graphs[-1].edge_index[0].tolist(), graphs[-1].edge_index[1].tolist()))
    assert (idx["AAA"], idx["BBB"]) in pairs
