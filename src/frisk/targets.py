from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import torch


def _file_signature(path: Path) -> str:
    try:
        st = path.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "missing"


def _build_cache_key(
    prices_file: Path,
    ticker: str,
    dates: list[str],
    horizon: int,
    standardize: bool,
    max_abs_logret: float,
    *,
    prefix: str = "",
) -> str:
    parts = [
        str(prices_file.resolve()),
        str(ticker).upper(),
        str(int(horizon)),
        str(int(bool(standardize))),
        f"{float(max_abs_logret):.8f}",
        _file_signature(prices_file),
        hashlib.sha1("\n".join(dates).encode("utf-8")).hexdigest(),
    ]
    if prefix:
        parts.insert(0, str(prefix))
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _read_ticker_log_returns(
    prices_path: str | Path,
    ticker: str,
    max_abs_logret: float,
) -> tuple[list[str], list[float]]:
    prices_by_date: dict[str, list[float]] = {}
    ticker_norm = str(ticker).upper()
    with Path(prices_path).open() as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("prices.csv missing header")
        price_col = "adj_close" if "adj_close" in r.fieldnames else "close"
        for row in r:
            if str(row.get("ticker", "")).upper() != ticker_norm:
                continue
            date = row.get("date")
            if not date:
                continue
            val = row.get(price_col, "")
            if not val:
                continue
            try:
                price = float(val)
            except ValueError:
                continue
            if not math.isfinite(price) or price <= 0:
                continue
            prices_by_date.setdefault(date, []).append(price)

    prices: list[tuple[str, float]] = []
    for date, vals in prices_by_date.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        mid = len(vals_sorted) // 2
        if len(vals_sorted) % 2 == 1:
            px = vals_sorted[mid]
        else:
            px = 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
        prices.append((date, float(px)))

    if not prices:
        raise ValueError(f"No prices found for ticker {ticker} in {prices_path}")

    prices.sort(key=lambda x: x[0])
    date_list = [d for d, _ in prices]
    price_list = [p for _, p in prices]
    log_returns: list[float] = []
    clip = float(max_abs_logret)
    for i in range(len(price_list) - 1):
        if price_list[i] <= 0 or price_list[i + 1] <= 0:
            log_returns.append(0.0)
            continue
        ret = math.log(price_list[i + 1] / price_list[i])
        if clip > 0 and abs(ret) > clip:
            ret = math.copysign(clip, ret)
        log_returns.append(ret)
    return date_list, log_returns


def _load_cached_targets(cache_path: Path) -> tuple[list[float | None], float, float] | None:
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["targets"], float(payload["mean"]), float(payload["std"])
    except Exception:
        return None


def _save_cached_targets(
    cache_path: Path | None,
    targets: list[float | None],
    mean: float,
    std: float,
) -> None:
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"targets": targets, "mean": mean, "std": std}, cache_path)
    except Exception:
        return


def compute_risk_targets_cached(
    prices_path: str,
    ticker: str,
    dates: list[str],
    horizon: int,
    standardize: bool,
    max_abs_logret: float,
    cache_dir: str | None = "runs/cache",
    mem_cache: dict[str, tuple[list[float | None], float, float]] | None = None,
) -> tuple[list[float | None], float, float]:
    prices_file = Path(prices_path)
    cache_key = _build_cache_key(
        prices_file,
        ticker,
        dates,
        horizon,
        standardize,
        max_abs_logret,
    )

    if mem_cache is not None:
        cached = mem_cache.get(cache_key)
        if cached is not None:
            return cached

    cache_path: Path | None = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"risk_targets_{cache_key}.pt"
        cached_disk = _load_cached_targets(cache_path)
        if cached_disk is not None:
            if mem_cache is not None:
                mem_cache[cache_key] = cached_disk
            return cached_disk

    date_list, log_returns = _read_ticker_log_returns(
        prices_path=prices_path,
        ticker=ticker,
        max_abs_logret=max_abs_logret,
    )
    idx_map = {d: i for i, d in enumerate(date_list)}

    targets: list[float | None] = []
    horizon = max(1, int(horizon))
    for d in dates:
        idx = idx_map.get(d)
        if idx is None or idx + horizon > len(log_returns):
            targets.append(None)
            continue
        window = log_returns[idx : idx + horizon]
        if not window:
            targets.append(None)
            continue
        mean_w = sum(window) / len(window)
        var_w = sum((x - mean_w) ** 2 for x in window) / len(window)
        targets.append(math.sqrt(var_w))

    finite = [t for t in targets if t is not None]
    if not finite:
        result = (targets, 0.0, 1.0)
        if mem_cache is not None:
            mem_cache[cache_key] = result
        _save_cached_targets(cache_path, targets, 0.0, 1.0)
        return result

    mean = sum(finite) / len(finite)
    var = sum((x - mean) ** 2 for x in finite) / len(finite)
    std = math.sqrt(var) if var > 0 else 1.0
    if standardize:
        targets = [((t - mean) / (std + 1e-6)) if t is not None else None for t in targets]

    result = (targets, mean, std)
    if mem_cache is not None:
        mem_cache[cache_key] = result
    _save_cached_targets(cache_path, targets, mean, std)
    return result


def compute_forward_return_targets_cached(
    prices_path: str,
    ticker: str,
    dates: list[str],
    horizon: int,
    standardize: bool,
    max_abs_logret: float,
    cache_dir: str | None = "runs/cache",
    mem_cache: dict[str, tuple[list[float | None], float, float]] | None = None,
) -> tuple[list[float | None], float, float]:
    prices_file = Path(prices_path)
    cache_key = _build_cache_key(
        prices_file,
        ticker,
        dates,
        horizon,
        standardize,
        max_abs_logret,
        prefix="portfolio_targets",
    )

    if mem_cache is not None:
        cached = mem_cache.get(cache_key)
        if cached is not None:
            return cached

    cache_path: Path | None = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"portfolio_targets_{cache_key}.pt"
        cached_disk = _load_cached_targets(cache_path)
        if cached_disk is not None:
            if mem_cache is not None:
                mem_cache[cache_key] = cached_disk
            return cached_disk

    date_list, log_returns = _read_ticker_log_returns(
        prices_path=prices_path,
        ticker=ticker,
        max_abs_logret=max_abs_logret,
    )
    idx_map = {d: i for i, d in enumerate(date_list)}

    targets: list[float | None] = []
    horizon = max(1, int(horizon))
    for d in dates:
        idx = idx_map.get(d)
        if idx is None or idx + horizon > len(log_returns):
            targets.append(None)
            continue
        window = log_returns[idx : idx + horizon]
        if not window:
            targets.append(None)
            continue
        targets.append(float(math.exp(sum(window)) - 1.0))

    finite = [t for t in targets if t is not None]
    if not finite:
        result = (targets, 0.0, 1.0)
        if mem_cache is not None:
            mem_cache[cache_key] = result
        _save_cached_targets(cache_path, targets, 0.0, 1.0)
        return result

    mean = sum(finite) / len(finite)
    var = sum((x - mean) ** 2 for x in finite) / len(finite)
    std = math.sqrt(var) if var > 0 else 1.0
    if standardize:
        targets = [((t - mean) / (std + 1e-6)) if t is not None else None for t in targets]

    result = (targets, mean, std)
    if mem_cache is not None:
        mem_cache[cache_key] = result
    _save_cached_targets(cache_path, targets, mean, std)
    return result
