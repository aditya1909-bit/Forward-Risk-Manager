"""Convert QuantConnect Research CSV exports into tidy CSVs.

Outputs:
  - prices.csv: date,ticker,open,high,low,close,adj_close,volume
  - constituents.csv: date,ticker,is_member,weight,sector,market_cap

The script tries to infer columns but allows overrides via CLI.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _normalize(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def _build_norm_map(columns: Iterable[str]) -> Dict[str, str]:
    return {_normalize(c): c for c in columns}


def _pick_col(norm_map: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        key = _normalize(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _load_csvs(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        frames = []
        for f in files:
            frames.append(pd.read_csv(f))
        return pd.concat(frames, ignore_index=True)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _coerce_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    # Normalize to date string for stable joins
    return dt.dt.strftime("%Y-%m-%d")


def _clean_ticker(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    # Extract first plausible ticker-like token
    s = s.str.extract(r"([A-Z0-9][A-Z0-9\.\-/]*)", expand=False)
    return s


def _infer_date_col(norm_map: Dict[str, str]) -> Optional[str]:
    return _pick_col(norm_map, ["date", "time", "timestamp", "datetime", "end", "period_end"])


def _infer_ticker_col(norm_map: Dict[str, str], df: pd.DataFrame) -> Optional[str]:
    # Prefer constituent-like columns if present
    preferred = _pick_col(norm_map, ["constituent", "holding", "member", "component", "constituent_symbol"])
    if preferred:
        return preferred

    candidate = _pick_col(norm_map, ["ticker", "symbol", "sid"])
    if not candidate:
        return None

    # Heuristic: if there's another symbol-like column with higher cardinality, use that
    symbol_like = []
    for key in ["ticker", "symbol", "constituent", "holding", "member", "component"]:
        col = _pick_col(norm_map, [key])
        if col:
            symbol_like.append(col)
    if len(symbol_like) > 1:
        best = max(symbol_like, key=lambda c: df[c].nunique(dropna=True))
        return best
    return candidate


def _standardize_prices(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    norm_map = _build_norm_map(df.columns)
    date_col = args.price_date_col or _infer_date_col(norm_map)
    ticker_col = args.price_ticker_col or _infer_ticker_col(norm_map, df)

    if not date_col or not ticker_col:
        raise ValueError(
            "Could not infer price date/ticker columns. "
            "Pass --price-date-col and --price-ticker-col."
        )

    col_map = {
        "open": _pick_col(norm_map, [args.open_col or "open"]),
        "high": _pick_col(norm_map, [args.high_col or "high"]),
        "low": _pick_col(norm_map, [args.low_col or "low"]),
        "close": _pick_col(norm_map, [args.close_col or "close"]),
        "adj_close": _pick_col(norm_map, [args.adj_close_col or "adj_close", "adjclose", "adjusted_close"]),
        "volume": _pick_col(norm_map, [args.volume_col or "volume"]),
    }

    out = pd.DataFrame()
    out["date"] = _coerce_date(df[date_col])
    out["ticker"] = _clean_ticker(df[ticker_col])

    for key, col in col_map.items():
        if col and col in df.columns:
            out[key] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[key] = pd.NA

    # Fallback for adj_close
    if out["adj_close"].isna().all() and "close" in out.columns:
        out["adj_close"] = out["close"]

    out = out.dropna(subset=["date", "ticker"]).drop_duplicates()
    return out


def _standardize_constituents(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    norm_map = _build_norm_map(df.columns)
    date_col = args.constituent_date_col or _infer_date_col(norm_map)
    ticker_col = args.constituent_ticker_col or _infer_ticker_col(norm_map, df)

    if not date_col or not ticker_col:
        raise ValueError(
            "Could not infer constituent date/ticker columns. "
            "Pass --constituent-date-col and --constituent-ticker-col."
        )

    weight_col = _pick_col(norm_map, [args.weight_col or "weight", "percentage", "value"])
    sector_col = _pick_col(norm_map, [args.sector_col or "sector", "industry", "gics_sector"])
    member_col = _pick_col(norm_map, [args.member_col or "is_member", "member", "included"])

    out = pd.DataFrame()
    out["date"] = _coerce_date(df[date_col])
    out["ticker"] = _clean_ticker(df[ticker_col])

    if member_col:
        out["is_member"] = pd.to_numeric(df[member_col], errors="coerce").fillna(1).astype(int)
    else:
        out["is_member"] = 1

    if weight_col:
        out["weight"] = pd.to_numeric(df[weight_col], errors="coerce")
    else:
        out["weight"] = pd.NA

    if sector_col:
        out["sector"] = df[sector_col].astype(str)
    else:
        out["sector"] = pd.NA

    out = out.dropna(subset=["date", "ticker"]).drop_duplicates()
    return out


def _standardize_coarse(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    norm_map = _build_norm_map(df.columns)
    date_col = args.coarse_date_col or _infer_date_col(norm_map)
    ticker_col = args.coarse_ticker_col or _infer_ticker_col(norm_map, df)

    if not date_col or not ticker_col:
        raise ValueError(
            "Could not infer coarse date/ticker columns. "
            "Pass --coarse-date-col and --coarse-ticker-col."
        )

    market_cap_col = _pick_col(norm_map, [args.market_cap_col or "market_cap", "marketcap", "mktcap"])
    if not market_cap_col:
        # Some coarse datasets have price + shares, but if market cap isn't present we skip
        return pd.DataFrame(columns=["date", "ticker", "market_cap"])

    out = pd.DataFrame()
    out["date"] = _coerce_date(df[date_col])
    out["ticker"] = _clean_ticker(df[ticker_col])
    out["market_cap"] = pd.to_numeric(df[market_cap_col], errors="coerce")
    out = out.dropna(subset=["date", "ticker"]).drop_duplicates()
    return out


def _merge_market_cap(constituents: pd.DataFrame, coarse: pd.DataFrame) -> pd.DataFrame:
    if coarse.empty:
        constituents["market_cap"] = pd.NA
        return constituents
    merged = constituents.merge(coarse, on=["date", "ticker"], how="left")
    if "market_cap" not in merged.columns:
        merged["market_cap"] = pd.NA
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert QuantConnect Research CSV exports into tidy CSVs.")
    parser.add_argument("--prices", required=True, help="CSV file or directory of price exports")
    parser.add_argument("--constituents", required=True, help="CSV file or directory of constituents exports")
    parser.add_argument("--coarse", help="Optional CSV file or directory of coarse universe exports")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory")

    # Optional overrides for price columns
    parser.add_argument("--price-date-col")
    parser.add_argument("--price-ticker-col")
    parser.add_argument("--open-col")
    parser.add_argument("--high-col")
    parser.add_argument("--low-col")
    parser.add_argument("--close-col")
    parser.add_argument("--adj-close-col")
    parser.add_argument("--volume-col")

    # Optional overrides for constituent columns
    parser.add_argument("--constituent-date-col")
    parser.add_argument("--constituent-ticker-col")
    parser.add_argument("--weight-col")
    parser.add_argument("--sector-col")
    parser.add_argument("--member-col")

    # Optional overrides for coarse columns
    parser.add_argument("--coarse-date-col")
    parser.add_argument("--coarse-ticker-col")
    parser.add_argument("--market-cap-col")

    args = parser.parse_args()

    prices_path = Path(args.prices)
    constituents_path = Path(args.constituents)
    coarse_path = Path(args.coarse) if args.coarse else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices_raw = _load_csvs(prices_path)
    constituents_raw = _load_csvs(constituents_path)
    prices = _standardize_prices(prices_raw, args)
    constituents = _standardize_constituents(constituents_raw, args)

    if coarse_path:
        coarse_raw = _load_csvs(coarse_path)
        coarse = _standardize_coarse(coarse_raw, args)
    else:
        coarse = pd.DataFrame(columns=["date", "ticker", "market_cap"])

    constituents = _merge_market_cap(constituents, coarse)

    prices_out = out_dir / "prices.csv"
    constituents_out = out_dir / "constituents.csv"

    prices.to_csv(prices_out, index=False)
    constituents.to_csv(constituents_out, index=False)

    print(f"Wrote {prices_out} ({len(prices):,} rows)")
    print(f"Wrote {constituents_out} ({len(constituents):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
