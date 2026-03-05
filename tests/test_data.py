import pandas as pd

from frisk.data import (
    compute_log_returns_and_volume,
    load_macro_features,
    build_macro_features_from_market_data,
    load_static_edges,
    load_sec_fundamentals,
)


def test_compute_log_returns_and_volume_shapes():
    prices = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "price": [100.0, 50.0, 101.0, 49.0],
            "volume": [1000, 2000, 1100, 1900],
        }
    )
    returns, volume = compute_log_returns_and_volume(prices)
    assert returns.shape == (2, 2)
    assert volume is not None
    assert volume.shape == (2, 2)


def test_load_macro_features_supports_long_format(tmp_path):
    macro_csv = tmp_path / "macro_prices.csv"
    df = pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "QQQ",
                "TLT",
                "HYG",
                "VXX",
                "SPY",
                "QQQ",
                "TLT",
                "HYG",
                "VXX",
                "SPY",
                "QQQ",
                "TLT",
                "HYG",
                "VXX",
            ],
            "time": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
                "2020-01-03",
                "2020-01-03",
                "2020-01-03",
            ],
            "close": [
                100.0,
                50.0,
                120.0,
                90.0,
                40.0,
                101.0,
                49.5,
                120.5,
                90.6,
                41.5,
                100.5,
                50.2,
                121.2,
                91.1,
                43.0,
            ],
            "volume": [1000, 1500, 1200, 1300, 800, 1100, 1400, 1250, 1320, 820, 1200, 1600, 1300, 1350, 840],
        }
    )
    df.to_csv(macro_csv, index=False)
    macro = load_macro_features(macro_csv)
    assert not macro.empty
    assert "macro_mkt_ret_eqw" in macro.columns
    assert "macro_ret_SPY" in macro.columns
    assert "macro_rates_proxy_tlt" in macro.columns
    assert "macro_credit_term_hyg_tlt" in macro.columns
    assert "macro_vol_term_vxx_spy" in macro.columns


def test_build_macro_features_from_market_data_has_expected_columns():
    idx = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.00, -0.01, 0.02, 0.00, -0.02],
            "BBB": [0.00, 0.01, 0.00, -0.01, 0.02, 0.00],
        },
        index=idx,
    )
    volume = pd.DataFrame(
        {
            "AAA": [100, 110, 105, 115, 112, 118],
            "BBB": [200, 195, 205, 198, 210, 215],
        },
        index=idx,
    )
    macro = build_macro_features_from_market_data(returns=returns, volume=volume, short_window=3, long_window=5)
    assert not macro.empty
    assert "macro_mkt_ret_eqw" in macro.columns
    assert "macro_mkt_vol_ratio" in macro.columns
    assert "macro_log_volume" in macro.columns


def test_load_static_edges_normalizes_source_target_columns(tmp_path):
    edges_csv = tmp_path / "static_edges.csv"
    df = pd.DataFrame(
        {
            "source": ["aaa", "bbb"],
            "target": ["ccc", "ddd"],
            "strength": [0.8, 1.2],
            "is_directed": [1, 0],
        }
    )
    df.to_csv(edges_csv, index=False)
    edges = load_static_edges(edges_csv)
    assert list(edges.columns) == ["src", "dst", "weight", "directed"]
    assert edges["src"].tolist() == ["AAA", "BBB"]
    assert edges["dst"].tolist() == ["CCC", "DDD"]
    assert edges["weight"].tolist() == [0.8, 1.2]
    assert edges["directed"].tolist() == [True, False]


def test_load_sec_fundamentals_builds_expected_features(tmp_path):
    companyfacts_csv = tmp_path / "sec_companyfacts_selected.csv"
    submissions_csv = tmp_path / "sec_submissions_entities.csv"

    pd.DataFrame(
        {
            "cik": ["1", "1", "1", "2"],
            "ticker": ["AAA", "AAA", "AAA", ""],
            "entity_name": ["A", "A", "A", "B"],
            "taxonomy": ["us-gaap", "us-gaap", "us-gaap", "us-gaap"],
            "tag": ["Revenues", "NetIncomeLoss", "Assets", "Liabilities"],
            "unit": ["USD", "USD", "USD", "USD"],
            "end": ["2020-03-31", "2020-03-31", "2020-03-31", "2020-03-31"],
            "val": [100.0, 10.0, 200.0, 50.0],
            "filed": ["2020-04-20", "2020-04-20", "2020-04-20", "2020-04-21"],
            "form": ["10-Q", "10-Q", "10-Q", "10-Q"],
            "fy": [2020, 2020, 2020, 2020],
            "fp": ["Q1", "Q1", "Q1", "Q1"],
            "accn": ["x", "x", "x", "y"],
            "frame": ["f", "f", "f", "f"],
            "source_file": ["a", "a", "a", "b"],
        }
    ).to_csv(companyfacts_csv, index=False)

    pd.DataFrame(
        {
            "cik": ["1", "2"],
            "ticker": ["AAA", "BBB"],
            "sic": [1234, 5678],
            "recent_filings_count": [9, 5],
        }
    ).to_csv(submissions_csv, index=False)

    out = load_sec_fundamentals(companyfacts_csv, submissions_path=submissions_csv)
    assert not out.empty
    assert {"date", "ticker", "sec_revenues", "sec_net_income", "sec_assets", "sec_net_margin"}.issubset(
        out.columns
    )
    aaa = out[out["ticker"] == "AAA"].iloc[-1]
    assert aaa["sec_net_margin"] == 0.1
    assert aaa["sec_sic"] == 1234
    assert aaa["sec_recent_filings_count"] == 9
