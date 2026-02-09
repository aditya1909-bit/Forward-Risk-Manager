import numpy as np
import pandas as pd
import torch

from frisk.graph_builder import (
    GraphBuildConfig,
    _build_node_features,
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
        mdy_ticker="MDY",
        rsi_period=14,
        fund_features=fund,
    )
    # window_len=4, summary=5, fund=5 => 14 features
    assert x.shape == (3, 14)
    assert ret_mean is None
    assert ret_std is None


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
