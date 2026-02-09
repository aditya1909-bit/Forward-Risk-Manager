import pandas as pd

from frisk.data import compute_log_returns_and_volume


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
