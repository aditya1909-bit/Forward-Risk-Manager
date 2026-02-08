#!/usr/bin/env python3
from __future__ import annotations

"""Merge year-bucketed QC exports into single CSVs.

Outputs cleaned, merged files in an output directory:
  - prices.csv
  - constituents.csv
  - fundamentals.csv
  - macro_prices.csv
"""

import argparse
from pathlib import Path
import re

import pandas as pd


def _clean_ticker(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    # Strip QC SID or extra tokens ("TICKER SID")
    s = s.str.split().str[0]
    s = s.str.extract(r"([A-Z0-9][A-Z0-9\.\-/]*)", expand=False)
    return s


def _read_all(root: Path, filename: str) -> pd.DataFrame:
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"No {filename} files found under {root}")
    frames = []
    for f in files:
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge year-bucketed QC exports.")
    parser.add_argument("--raw-root", default="data/raw", help="Root folder with year subfolders")
    parser.add_argument("--out-dir", default="data/raw_merged", help="Output directory")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)

    prices = _read_all(raw_root, "prices.csv")
    if "ticker" not in prices.columns:
        raise ValueError("prices.csv missing 'ticker' column")
    prices["ticker"] = _clean_ticker(prices["ticker"])
    if "date" in prices.columns:
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    prices = prices.dropna(subset=["date", "ticker"]).drop_duplicates()
    _write(prices, out_dir / "prices.csv")

    constituents = _read_all(raw_root, "constituents.csv")
    # Normalize to columns expected by pipeline
    if "constituent_symbol" in constituents.columns and "ticker" not in constituents.columns:
        constituents = constituents.rename(columns={"constituent_symbol": "ticker"})
    if "ticker" not in constituents.columns:
        raise ValueError("constituents.csv missing ticker/constituent_symbol column")
    constituents["ticker"] = _clean_ticker(constituents["ticker"])
    if "date" in constituents.columns:
        constituents["date"] = pd.to_datetime(constituents["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    constituents = constituents.dropna(subset=["date", "ticker"]).drop_duplicates()
    _write(constituents, out_dir / "constituents.csv")

    fundamentals = _read_all(raw_root, "fundamentals.csv")
    if "ticker" in fundamentals.columns:
        fundamentals["ticker"] = _clean_ticker(fundamentals["ticker"])
    if "date" in fundamentals.columns:
        fundamentals["date"] = pd.to_datetime(fundamentals["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fundamentals = fundamentals.dropna(subset=["date", "ticker"]).drop_duplicates()
    _write(fundamentals, out_dir / "fundamentals.csv")

    macro = _read_all(raw_root, "macro_prices.csv")
    if "ticker" in macro.columns:
        macro["ticker"] = _clean_ticker(macro["ticker"])
    if "time" in macro.columns:
        macro["time"] = pd.to_datetime(macro["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    macro = macro.dropna(subset=["time", "ticker"]).drop_duplicates()
    _write(macro, out_dir / "macro_prices.csv")

    print(f"Wrote merged files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
