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
