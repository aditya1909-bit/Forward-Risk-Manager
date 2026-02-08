from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_prices(path: Path, price_col: str = "adj_close") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("prices.csv must include a 'date' column")
    if "ticker" not in df.columns:
        raise ValueError("prices.csv must include a 'ticker' column")

    if price_col not in df.columns:
        if "close" in df.columns:
            price_col = "close"
        else:
            raise ValueError(f"prices.csv missing '{price_col}' (or 'close')")

    cols = ["date", "ticker", price_col]
    if "volume" in df.columns:
        cols.append("volume")
    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date", "ticker"])
    df = df.drop_duplicates(subset=["date", "ticker"])
    df = df.rename(columns={price_col: "price"})
    return df


def compute_log_returns_and_volume(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    pivot = prices.pivot(index="date", columns="ticker", values="price").sort_index()
    returns = np.log(pivot / pivot.shift(1))
    volume = None
    if "volume" in prices.columns:
        volume = prices.pivot(index="date", columns="ticker", values="volume").sort_index()
    return returns, volume


def load_constituents(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("constituents.csv must include a 'date' column")
    if "ticker" not in df.columns:
        raise ValueError("constituents.csv must include a 'ticker' column")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "is_member" in df.columns:
        df = df[df["is_member"] == 1]
    df = df.dropna(subset=["date", "ticker"])
    return df


def build_membership_map(
    constituents: pd.DataFrame,
    extra_tickers: List[str] | None = None,
) -> Dict[str, List[str]]:
    grouped = constituents.groupby("date")["ticker"].apply(list)
    membership = grouped.to_dict()
    if extra_tickers:
        extra = [t.upper().strip() for t in extra_tickers if t]
        for date, members in membership.items():
            seen = set(members)
            for t in extra:
                if t not in seen:
                    members.append(t)
                    seen.add(t)
    return membership


def _apply_extra_tickers(members: List[str], extra_tickers: List[str] | None) -> List[str]:
    if not extra_tickers:
        return members
    extra = [t.upper().strip() for t in extra_tickers if t]
    seen = set(members)
    for t in extra:
        if t not in seen:
            members.append(t)
            seen.add(t)
    return members


def build_membership_map_ffill(
    constituents: pd.DataFrame,
    dates: List[str],
    extra_tickers: List[str] | None = None,
    max_gap_days: int | None = None,
) -> tuple[Dict[str, List[str]], Dict[str, int]]:
    grouped = constituents.groupby("date")["ticker"].apply(list)
    members_by_date = grouped.to_dict()
    stats = {
        "source_dates": len(members_by_date),
        "filled_dates": 0,
        "gap_dropped": 0,
    }
    membership: Dict[str, List[str]] = {}
    current: List[str] | None = None
    last_date: str | None = None
    for date in dates:
        if date in members_by_date:
            members = list(dict.fromkeys(members_by_date[date]))
            members = _apply_extra_tickers(members, extra_tickers)
            membership[date] = members
            current = members
            last_date = date
            continue
        if current is None or last_date is None:
            continue
        if max_gap_days is not None:
            gap = (pd.to_datetime(date) - pd.to_datetime(last_date)).days
            if gap > max_gap_days:
                stats["gap_dropped"] += 1
                continue
        stats["filled_dates"] += 1
        membership[date] = list(current)
    return membership, stats


def build_membership_map_all(
    returns: pd.DataFrame,
    extra_tickers: List[str] | None = None,
) -> Dict[str, List[str]]:
    tickers = [t for t in returns.columns if t]
    if extra_tickers:
        extra = [t.upper().strip() for t in extra_tickers if t]
        for t in extra:
            if t not in tickers:
                tickers.append(t)
    members = tickers
    membership = {str(date): members for date in returns.index}
    return membership


def _parse_debt_equity(value: str) -> float | None:
    if not value or not isinstance(value, str):
        return None
    # format like "1Y:0.210245;3M:0.453478"
    parts = value.split(";")
    parsed = {}
    for part in parts:
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            parsed[k.strip()] = float(v)
        except ValueError:
            continue
    if "1Y" in parsed:
        return parsed["1Y"]
    if "3M" in parsed:
        return parsed["3M"]
    return None


def load_fundamentals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("fundamentals.csv must include a 'date' column")
    if "ticker" not in df.columns:
        raise ValueError("fundamentals.csv must include a 'ticker' column")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "sector_code" in df.columns:
        df["sector_code"] = pd.to_numeric(df["sector_code"], errors="coerce")
    for col in ("market_cap", "pe_ratio", "pb_ratio"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "debt_equity" in df.columns:
        df["debt_equity"] = df["debt_equity"].apply(_parse_debt_equity)
    df = df.dropna(subset=["date", "ticker"])
    return df
