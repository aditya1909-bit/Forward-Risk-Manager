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
import math

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


def _dedupe_by_key(df: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.drop_duplicates(subset=keys, keep="last")
    return out, before - len(out)


def _resolve_ticker_price_path(df_ticker: pd.DataFrame, price_col: str) -> pd.DataFrame:
    # Choose one row per date that minimizes day-over-day log-price jumps.
    grouped = list(df_ticker.groupby("date", sort=True))
    if not grouped:
        return df_ticker.iloc[0:0]

    candidates: list[pd.DataFrame] = []
    for _, g in grouped:
        g = g.copy()
        g[price_col] = pd.to_numeric(g[price_col], errors="coerce")
        g = g[g[price_col] > 0]
        if g.empty:
            continue
        candidates.append(g.reset_index(drop=True))
    if not candidates:
        return df_ticker.iloc[0:0]

    prices = [c[price_col].to_numpy(dtype=float) for c in candidates]
    backptrs: list[list[int]] = [[-1] * len(prices[0])]
    prev_cost = [0.0] * len(prices[0])
    for t in range(1, len(prices)):
        cur_prices = prices[t]
        prev_prices = prices[t - 1]
        cur_cost = [0.0] * len(cur_prices)
        cur_ptr = [0] * len(cur_prices)
        for j, p in enumerate(cur_prices):
            best_cost = float("inf")
            best_idx = 0
            for i, pp in enumerate(prev_prices):
                step = abs(math.log(p / pp))
                cost = prev_cost[i] + step
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
            cur_cost[j] = best_cost
            cur_ptr[j] = best_idx
        prev_cost = cur_cost
        backptrs.append(cur_ptr)

    last_choice = min(range(len(prev_cost)), key=lambda i: prev_cost[i])
    choices = [0] * len(candidates)
    choices[-1] = last_choice
    for t in range(len(candidates) - 1, 0, -1):
        choices[t - 1] = backptrs[t][choices[t]]

    out_rows = [candidates[t].iloc[[choices[t]]] for t in range(len(candidates))]
    return pd.concat(out_rows, ignore_index=True)


def _resolve_price_duplicates(prices: pd.DataFrame, price_col: str) -> tuple[pd.DataFrame, int]:
    dup_mask = prices.duplicated(subset=["date", "ticker"], keep=False)
    if not dup_mask.any():
        return prices.drop_duplicates(subset=["date", "ticker"], keep="last"), 0

    resolved = []
    changed = 0
    for _, g in prices.groupby("ticker", sort=False):
        if not g.duplicated(subset=["date"], keep=False).any():
            resolved.append(g.drop_duplicates(subset=["date"], keep="last"))
            continue
        pre_rows = len(g)
        g_resolved = _resolve_ticker_price_path(g, price_col=price_col)
        changed += pre_rows - len(g_resolved)
        resolved.append(g_resolved)
    out = pd.concat(resolved, ignore_index=True)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out, changed


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
    prices = prices.dropna(subset=["date", "ticker"]).copy()
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    if price_col not in prices.columns:
        raise ValueError("prices.csv missing both 'close' and 'adj_close' columns")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices, resolved_dups = _resolve_price_duplicates(prices, price_col=price_col)
    prices = prices.dropna(subset=[price_col])
    prices, dropped_dups = _dedupe_by_key(prices, ["date", "ticker"])
    _write(prices, out_dir / "prices.csv")
    print(
        "prices:",
        f"rows={len(prices)}",
        f"resolved_duplicates={resolved_dups}",
        f"dropped_duplicates={dropped_dups}",
        f"price_col={price_col}",
    )

    constituents = _read_all(raw_root, "constituents.csv")
    # Normalize to columns expected by pipeline
    if "constituent_symbol" in constituents.columns and "ticker" not in constituents.columns:
        constituents = constituents.rename(columns={"constituent_symbol": "ticker"})
    if "ticker" not in constituents.columns:
        raise ValueError("constituents.csv missing ticker/constituent_symbol column")
    constituents["ticker"] = _clean_ticker(constituents["ticker"])
    if "date" in constituents.columns:
        constituents["date"] = pd.to_datetime(constituents["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    constituents = constituents.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])
    constituents, dropped_const = _dedupe_by_key(constituents, ["date", "ticker"])
    _write(constituents, out_dir / "constituents.csv")
    print("constituents:", f"rows={len(constituents)}", f"dropped_duplicates={dropped_const}")

    fundamentals = _read_all(raw_root, "fundamentals.csv")
    if "ticker" in fundamentals.columns:
        fundamentals["ticker"] = _clean_ticker(fundamentals["ticker"])
    if "date" in fundamentals.columns:
        fundamentals["date"] = pd.to_datetime(fundamentals["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fundamentals = fundamentals.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])
    fundamentals, dropped_fund = _dedupe_by_key(fundamentals, ["date", "ticker"])
    _write(fundamentals, out_dir / "fundamentals.csv")
    print("fundamentals:", f"rows={len(fundamentals)}", f"dropped_duplicates={dropped_fund}")

    macro = _read_all(raw_root, "macro_prices.csv")
    if "ticker" in macro.columns:
        macro["ticker"] = _clean_ticker(macro["ticker"])
    if "time" in macro.columns:
        macro["time"] = pd.to_datetime(macro["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    macro = macro.dropna(subset=["time", "ticker"]).sort_values(["time", "ticker"])
    macro, dropped_macro = _dedupe_by_key(macro, ["time", "ticker"])
    _write(macro, out_dir / "macro_prices.csv")
    print("macro_prices:", f"rows={len(macro)}", f"dropped_duplicates={dropped_macro}")

    print(f"Wrote merged files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
