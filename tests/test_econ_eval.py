import numpy as np
import pandas as pd

from frisk.econ_eval import (
    evaluate_goodness_strategy,
    load_forward_returns_from_prices,
    resolve_price_ticker,
)


def test_evaluate_goodness_strategy_reports_uplift():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    goodness = np.linspace(-1.0, 1.0, num=40)
    fwd = pd.Series(
        np.where(goodness > 0.0, 0.01, -0.01),
        index=dates,
    )

    out = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=20,
        signal_quantile=0.5,
        turnover_cost_bps=0.0,
    )

    assert out["econ_num_days"] > 0
    assert np.isfinite(out["econ_strategy_ann_return"])
    assert np.isfinite(out["econ_bh_ann_return"])
    assert np.isfinite(out["econ_strategy_sortino"])
    assert np.isfinite(out["econ_bh_calmar"])
    assert np.isfinite(out["econ_strategy_var_95_daily"])
    assert np.isfinite(out["econ_bh_var_95_daily"])
    assert np.isfinite(out["econ_strategy_max_drawdown_duration_days"])
    assert out["econ_ann_return_uplift"] > -1e-6


def test_evaluate_goodness_strategy_slippage_reduces_strategy_returns():
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    goodness = np.sin(np.linspace(0, 8, num=80))
    fwd = pd.Series(0.005 + 0.01 * np.sin(np.linspace(0, 10, num=80)), index=dates)

    no_slip = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=20,
        signal_quantile=0.5,
        turnover_cost_bps=0.0,
        slippage_bps=0.0,
        slippage_vol_scale=0.0,
        slippage_vol_lookback=10,
    )
    with_slip = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=20,
        signal_quantile=0.5,
        turnover_cost_bps=10.0,
        slippage_bps=5.0,
        slippage_vol_scale=20.0,
        slippage_vol_lookback=10,
    )

    assert with_slip["econ_avg_cost_bps_applied"] >= 0.0
    assert with_slip["econ_strategy_total_return"] <= no_slip["econ_strategy_total_return"] + 1e-9


def test_evaluate_goodness_strategy_reports_exposure_adjusted_baseline():
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    goodness = np.linspace(-1.0, 1.0, num=120)
    fwd = pd.Series(0.002 + 0.004 * np.sin(np.linspace(0, 8, num=120)), index=dates)

    out = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=30,
        signal_quantile=0.6,
        turnover_cost_bps=0.0,
        slippage_bps=0.0,
    )

    assert np.isfinite(out["econ_exposure_benchmark_exposure"])
    assert 0.0 <= out["econ_exposure_benchmark_exposure"] <= 1.0
    assert np.isfinite(out["econ_exposure_benchmark_ann_return"])
    assert np.isfinite(out["econ_exposure_adjusted_ann_return_uplift"])
    assert np.isfinite(out["econ_exposure_adjusted_sharpe_uplift"])


def test_load_forward_returns_from_prices_uses_close_and_dedups(tmp_path):
    csv_path = tmp_path / "prices.csv"
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ],
            "ticker": ["MDY", "MDY", "MDY", "MDY"],
            "close": [100.0, 100.0, 101.0, 102.0],
        }
    )
    df.to_csv(csv_path, index=False)

    fwd = load_forward_returns_from_prices(csv_path, ticker="MDY", max_abs_logret=0.5)

    assert isinstance(fwd, pd.Series)
    assert not fwd.empty
    assert fwd.index.is_unique


def test_resolve_price_ticker_auto_and_requested(tmp_path):
    csv_path = tmp_path / "prices.csv"
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
            ],
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "close": [10.0, 10.1, 10.2, 9.9],
        }
    )
    df.to_csv(csv_path, index=False)

    t_auto, src_auto, rows_auto = resolve_price_ticker(csv_path, requested_ticker="AUTO", min_rows=2)
    assert t_auto == "AAA"
    assert src_auto in {"auto_max_rows", "auto_max_rows_no_min_match"}
    assert rows_auto == 3

    t_req, src_req, rows_req = resolve_price_ticker(csv_path, requested_ticker="BBB", min_rows=2)
    assert t_req == "BBB"
    assert src_req == "requested"
    assert rows_req == 1


def test_resolve_price_ticker_requested_priority_with_auto_fallback(tmp_path):
    csv_path = tmp_path / "prices.csv"
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
            ],
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "close": [10.0, 10.1, 10.2, 9.9, 10.0],
        }
    )
    df.to_csv(csv_path, index=False)

    ticker, src, rows = resolve_price_ticker(
        csv_path,
        requested_ticker="SPY,BBB,AUTO",
        min_rows=2,
    )
    assert ticker == "BBB"
    assert src == "requested_priority"
    assert rows == 2


def test_resolve_price_ticker_requested_list_handles_whitespace(tmp_path):
    csv_path = tmp_path / "prices.csv"
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
            ],
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "close": [10.0, 10.1, 10.2, 9.9, 10.0],
        }
    )
    df.to_csv(csv_path, index=False)

    ticker, src, rows = resolve_price_ticker(
        csv_path,
        requested_ticker="SPY, BBB, AUTO",
        min_rows=2,
    )
    assert ticker == "BBB"
    assert src == "requested_priority"
    assert rows == 2


def test_evaluate_goodness_strategy_regime_thresholding_outputs_stats():
    dates = pd.date_range("2021-01-01", periods=160, freq="D")
    base = np.linspace(-1.0, 1.0, num=160)
    noise = 0.1 * np.sin(np.linspace(0, 20, num=160))
    goodness = base + noise
    # Piecewise volatility regimes.
    low = 0.002 * np.sin(np.linspace(0, 6, num=60))
    mid = 0.008 * np.sin(np.linspace(0, 6, num=50))
    high = 0.02 * np.sin(np.linspace(0, 6, num=50))
    fwd = pd.Series(np.concatenate([low, mid, high]), index=dates)

    out_on = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=40,
        signal_quantile=0.5,
        regime_thresholding_enabled=True,
        regime_threshold_window=40,
        regime_threshold_quantile=0.5,
        regime_vol_window=15,
        regime_low_quantile=0.33,
        regime_high_quantile=0.67,
    )
    out_off = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=40,
        signal_quantile=0.5,
        regime_thresholding_enabled=False,
    )

    assert out_on["econ_regime_thresholding_enabled"] == 1.0
    assert out_off["econ_regime_thresholding_enabled"] == 0.0
    assert out_on["econ_regime_low_count"] > 0
    assert out_on["econ_regime_mid_count"] > 0
    assert out_on["econ_regime_high_count"] > 0
    assert np.isfinite(out_on["econ_strategy_ann_return"])


def test_evaluate_goodness_strategy_signal_polarity_low_and_auto():
    dates = pd.date_range("2022-01-01", periods=120, freq="D")
    goodness = np.linspace(-1.0, 1.0, num=120)
    # Goodness is intentionally anti-correlated with future returns.
    fwd = pd.Series(np.where(goodness < 0.0, 0.01, -0.01), index=dates)

    out_high = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=30,
        signal_quantile=0.5,
        signal_polarity="high",
        turnover_cost_bps=0.0,
    )
    out_low = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=30,
        signal_quantile=0.5,
        signal_polarity="low",
        turnover_cost_bps=0.0,
    )
    out_auto = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=30,
        signal_quantile=0.5,
        signal_polarity="auto",
        turnover_cost_bps=0.0,
    )

    assert out_low["econ_strategy_sharpe"] > out_high["econ_strategy_sharpe"]
    assert out_auto["econ_signal_polarity_requested"] == "auto"
    assert out_auto["econ_signal_polarity_effective"] == "low"
    assert out_auto["econ_strategy_sharpe"] >= out_low["econ_strategy_sharpe"] - 1e-9


def test_evaluate_goodness_strategy_reports_oos_fold_uplifts():
    dates = pd.date_range("2021-01-01", periods=240, freq="D")
    goodness = np.sin(np.linspace(0, 20, num=240))
    fwd = pd.Series(np.where(goodness >= 0.0, 0.012, -0.006), index=dates)

    out = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=40,
        signal_quantile=0.5,
        signal_polarity="high",
        oos_folds=4,
        oos_min_fold_days=40,
        turnover_cost_bps=0.0,
    )

    assert int(out["econ_oos_folds_requested"]) == 4
    assert int(out["econ_oos_min_fold_days"]) == 40
    assert out["econ_oos_folds_used"] >= 2
    assert np.isfinite(out["econ_oos_sharpe_uplift_mean"])
    assert np.isfinite(out["econ_oos_sharpe_uplift_min"])
    assert out["econ_oos_sharpe_uplift_min"] <= out["econ_oos_sharpe_uplift_mean"] + 1e-9


def test_evaluate_goodness_strategy_short_borrow_fee_reduces_short_returns():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    goodness = np.linspace(-0.5, 0.5, num=120)
    fwd = pd.Series(np.full(120, 0.01, dtype=float), index=dates)

    no_borrow = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=30,
        signal_quantile=0.5,
        regime_gate_enabled=True,
        regime_min_confidence=1.0,
        regime_neutral_exposure=-1.0,
        short_borrow_bps=0.0,
    )
    with_borrow = evaluate_goodness_strategy(
        dates=dates,
        goodness_scores=goodness,
        fwd_ret_1=fwd,
        signal_window=30,
        signal_quantile=0.5,
        regime_gate_enabled=True,
        regime_min_confidence=1.0,
        regime_neutral_exposure=-1.0,
        short_borrow_bps=50.0,
    )

    assert with_borrow["econ_avg_borrow_bps_applied"] > 0.0
    assert with_borrow["econ_strategy_total_return"] < no_borrow["econ_strategy_total_return"] - 1e-9
