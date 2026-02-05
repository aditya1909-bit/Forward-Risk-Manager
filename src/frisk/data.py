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
