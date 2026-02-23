from __future__ import annotations

import csv
from pathlib import Path

import pytest

from frisk import targets as target_mod


def _write_prices_csv(path: Path) -> None:
    rows = [
        ("2024-01-01", "AAA", 100.0),
        ("2024-01-02", "AAA", 101.0),
        ("2024-01-03", "AAA", 99.0),
        ("2024-01-04", "AAA", 103.0),
        ("2024-01-05", "AAA", 104.0),
        ("2024-01-06", "AAA", 106.0),
        ("2024-01-07", "AAA", 105.0),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "adj_close"])
        for row in rows:
            w.writerow(row)


def test_targets_cached_deterministic_outputs(tmp_path: Path):
    prices = tmp_path / "prices.csv"
    _write_prices_csv(prices)
    dates = [f"2024-01-0{i}" for i in range(1, 8)]

    risk_a = target_mod.compute_risk_targets_cached(
        prices_path=str(prices),
        ticker="AAA",
        dates=dates,
        horizon=3,
        standardize=True,
        max_abs_logret=0.5,
        cache_dir=None,
        mem_cache=None,
    )
    risk_b = target_mod.compute_risk_targets_cached(
        prices_path=str(prices),
        ticker="AAA",
        dates=dates,
        horizon=3,
        standardize=True,
        max_abs_logret=0.5,
        cache_dir=None,
        mem_cache=None,
    )
    assert risk_a == risk_b

    fwd_a = target_mod.compute_forward_return_targets_cached(
        prices_path=str(prices),
        ticker="AAA",
        dates=dates,
        horizon=2,
        standardize=True,
        max_abs_logret=0.5,
        cache_dir=None,
        mem_cache=None,
    )
    fwd_b = target_mod.compute_forward_return_targets_cached(
        prices_path=str(prices),
        ticker="AAA",
        dates=dates,
        horizon=2,
        standardize=True,
        max_abs_logret=0.5,
        cache_dir=None,
        mem_cache=None,
    )
    assert fwd_a == fwd_b


def test_targets_cached_mem_cache_hit_avoids_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prices = tmp_path / "prices.csv"
    _write_prices_csv(prices)
    dates = [f"2024-01-0{i}" for i in range(1, 8)]
    mem_cache: dict[str, tuple[list[float | None], float, float]] = {}

    out_first = target_mod.compute_risk_targets_cached(
        prices_path=str(prices),
        ticker="AAA",
        dates=dates,
        horizon=3,
        standardize=True,
        max_abs_logret=0.5,
        cache_dir=None,
        mem_cache=mem_cache,
    )

    def _fail_read(*_args, **_kwargs):
        raise AssertionError("mem cache miss: recomputation was attempted")

    monkeypatch.setattr(target_mod, "_read_ticker_log_returns", _fail_read)
    out_second = target_mod.compute_risk_targets_cached(
        prices_path=str(prices),
        ticker="AAA",
        dates=dates,
        horizon=3,
        standardize=True,
        max_abs_logret=0.5,
        cache_dir=None,
        mem_cache=mem_cache,
    )
    assert out_first == out_second


def test_targets_cached_disk_cache_hit_avoids_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prices = tmp_path / "prices.csv"
    _write_prices_csv(prices)
    dates = [f"2024-01-0{i}" for i in range(1, 8)]
    cache_dir = tmp_path / "cache"

    out_first = target_mod.compute_forward_return_targets_cached(
        prices_path=str(prices),
        ticker="AAA",
        dates=dates,
        horizon=2,
        standardize=True,
        max_abs_logret=0.5,
        cache_dir=str(cache_dir),
        mem_cache=None,
    )

    def _fail_read(*_args, **_kwargs):
        raise AssertionError("disk cache miss: recomputation was attempted")

    monkeypatch.setattr(target_mod, "_read_ticker_log_returns", _fail_read)
    out_second = target_mod.compute_forward_return_targets_cached(
        prices_path=str(prices),
        ticker="AAA",
        dates=dates,
        horizon=2,
        standardize=True,
        max_abs_logret=0.5,
        cache_dir=str(cache_dir),
        mem_cache=None,
    )
    assert out_first == out_second
