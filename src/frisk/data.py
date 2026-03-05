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


def _safe_divide_num(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return pd.to_numeric(a, errors="coerce") / denom


def load_sec_fundamentals(
    companyfacts_path: Path,
    submissions_path: Path | None = None,
) -> pd.DataFrame:
    if not companyfacts_path.exists():
        raise FileNotFoundError(f"SEC companyfacts CSV not found: {companyfacts_path}")

    usecols = ["cik", "ticker", "tag", "end", "filed", "val"]
    cf = pd.read_csv(companyfacts_path, usecols=usecols, low_memory=False)
    cf = cf.copy()
    cf["ticker"] = cf["ticker"].astype(str).str.upper().str.strip()
    cf["cik"] = cf["cik"].astype(str).str.strip()
    cf["tag"] = cf["tag"].astype(str).str.strip()
    cf["val"] = pd.to_numeric(cf["val"], errors="coerce")
    cf["date"] = pd.to_datetime(cf["filed"], errors="coerce")
    missing = cf["date"].isna()
    if missing.any():
        cf.loc[missing, "date"] = pd.to_datetime(cf.loc[missing, "end"], errors="coerce")
    cf["date"] = cf["date"].dt.strftime("%Y-%m-%d")
    cf = cf.dropna(subset=["date", "tag", "val"])

    sub = None
    if submissions_path is not None and submissions_path.exists():
        sub_usecols = ["cik", "ticker", "sic", "recent_filings_count"]
        sub = pd.read_csv(submissions_path, usecols=sub_usecols, low_memory=False)
        sub = sub.copy()
        sub["ticker"] = sub["ticker"].astype(str).str.upper().str.strip()
        sub["cik"] = sub["cik"].astype(str).str.strip()
        sub["sic"] = pd.to_numeric(sub["sic"], errors="coerce")
        sub["recent_filings_count"] = pd.to_numeric(sub["recent_filings_count"], errors="coerce")
        cik_to_ticker = (
            sub.dropna(subset=["cik", "ticker"])
            .drop_duplicates(subset=["cik"], keep="last")
            .set_index("cik")["ticker"]
            .to_dict()
        )
        missing_ticker = cf["ticker"].eq("") | cf["ticker"].isna()
        if missing_ticker.any():
            cf.loc[missing_ticker, "ticker"] = cf.loc[missing_ticker, "cik"].map(cik_to_ticker).fillna("")

    cf = cf[cf["ticker"] != ""]
    if cf.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    pivot = (
        cf.pivot_table(index=["date", "ticker"], columns="tag", values="val", aggfunc="last")
        .sort_index()
    )
    pivot.columns = [str(c) for c in pivot.columns]
    out = pivot.reset_index()

    tag_map = {
        "Assets": "sec_assets",
        "Liabilities": "sec_liabilities",
        "StockholdersEquity": "sec_stockholders_equity",
        "Revenues": "sec_revenues",
        "NetIncomeLoss": "sec_net_income",
        "OperatingIncomeLoss": "sec_operating_income",
        "GrossProfit": "sec_gross_profit",
        "LongTermDebtNoncurrent": "sec_long_term_debt",
        "CashAndCashEquivalentsAtCarryingValue": "sec_cash",
        "EntityCommonStockSharesOutstanding": "sec_shares_outstanding",
        "CommonStockSharesOutstanding": "sec_common_shares_outstanding",
        "EarningsPerShareBasic": "sec_eps_basic",
        "EarningsPerShareDiluted": "sec_eps_diluted",
    }
    rename_cols = {c: tag_map[c] for c in out.columns if c in tag_map}
    out = out.rename(columns=rename_cols)
    if "sec_shares_outstanding" not in out.columns and "sec_common_shares_outstanding" in out.columns:
        out["sec_shares_outstanding"] = out["sec_common_shares_outstanding"]

    if "sec_long_term_debt" in out.columns and "sec_stockholders_equity" in out.columns:
        out["sec_debt_to_equity"] = _safe_divide_num(out["sec_long_term_debt"], out["sec_stockholders_equity"])
    if "sec_net_income" in out.columns and "sec_revenues" in out.columns:
        out["sec_net_margin"] = _safe_divide_num(out["sec_net_income"], out["sec_revenues"])
    if "sec_operating_income" in out.columns and "sec_revenues" in out.columns:
        out["sec_operating_margin"] = _safe_divide_num(out["sec_operating_income"], out["sec_revenues"])
    if "sec_gross_profit" in out.columns and "sec_revenues" in out.columns:
        out["sec_gross_margin"] = _safe_divide_num(out["sec_gross_profit"], out["sec_revenues"])
    if "sec_cash" in out.columns and "sec_assets" in out.columns:
        out["sec_cash_to_assets"] = _safe_divide_num(out["sec_cash"], out["sec_assets"])
    if "sec_stockholders_equity" in out.columns and "sec_shares_outstanding" in out.columns:
        out["sec_book_value_per_share"] = _safe_divide_num(
            out["sec_stockholders_equity"], out["sec_shares_outstanding"]
        )

    if sub is not None:
        sub_ticker = (
            sub[sub["ticker"] != ""]
            .drop_duplicates(subset=["ticker"], keep="last")[["ticker", "sic", "recent_filings_count"]]
            .rename(
                columns={
                    "sic": "sec_sic",
                    "recent_filings_count": "sec_recent_filings_count",
                }
            )
        )
        out = out.merge(sub_ticker, on="ticker", how="left")

    for col in out.columns:
        if col in {"date", "ticker"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])
    return out


def load_macro_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"])
        value_cols = [c for c in df.columns if c != "date"]
        if not value_cols:
            raise ValueError("macro.csv must include at least one feature column besides date")
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        macro = df.groupby("date", as_index=True)[value_cols].mean().sort_index()
        return macro.replace([np.inf, -np.inf], np.nan)

    # Fallback for long-format macro price files:
    # ticker,time,close/high/low/open,volume
    if "time" in df.columns and "ticker" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df.dropna(subset=["date", "ticker"])
        price_col = "adj_close" if "adj_close" in df.columns else ("close" if "close" in df.columns else "")
        if not price_col:
            raise ValueError(
                "Long macro price CSV must include one of: adj_close, close."
            )
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        px = (
            df.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="mean")
            .sort_index()
        )
        log_ret = np.log(px / px.shift(1))

        feats: dict[str, pd.Series] = {}
        for ticker in log_ret.columns:
            s = pd.to_numeric(log_ret[ticker], errors="coerce")
            feats[f"macro_ret_{ticker}"] = s
            feats[f"macro_vol21_{ticker}"] = s.rolling(21, min_periods=5).std()

        # Term-structure style macro proxies when common ETF symbols are present.
        # This keeps integration feasible with local macro_prices universes that
        # do not contain full treasury/credit index histories.
        def _ticker_ret(symbol: str) -> pd.Series | None:
            if symbol in log_ret.columns:
                return pd.to_numeric(log_ret[symbol], errors="coerce")
            return None

        spy = _ticker_ret("SPY")
        tlt = _ticker_ret("TLT")
        hyg = _ticker_ret("HYG")
        vxx = _ticker_ret("VXX")
        gld = _ticker_ret("GLD")
        uup = _ticker_ret("UUP")

        if tlt is not None:
            feats["macro_rates_proxy_tlt"] = tlt
            feats["macro_rates_proxy_tlt_trend_21_63"] = (
                tlt.rolling(21, min_periods=5).mean()
                - tlt.rolling(63, min_periods=10).mean()
            )
        if hyg is not None:
            feats["macro_credit_proxy_hyg"] = hyg
            feats["macro_credit_proxy_hyg_trend_21_63"] = (
                hyg.rolling(21, min_periods=5).mean()
                - hyg.rolling(63, min_periods=10).mean()
            )
        if vxx is not None:
            feats["macro_vol_proxy_vxx"] = vxx
            feats["macro_vol_proxy_vxx_vol21"] = vxx.rolling(21, min_periods=5).std()

        if hyg is not None and tlt is not None:
            feats["macro_credit_term_hyg_tlt"] = hyg - tlt
        elif hyg is not None and spy is not None:
            feats["macro_credit_term_hyg_spy"] = hyg - spy

        if vxx is not None and spy is not None:
            feats["macro_vol_term_vxx_spy"] = vxx - spy

        if tlt is not None and spy is not None:
            feats["macro_duration_equity_term_tlt_spy"] = tlt - spy

        if gld is not None and uup is not None:
            feats["macro_real_asset_fx_term_gld_uup"] = gld - uup

        mkt_ret = log_ret.mean(axis=1)
        feats["macro_mkt_ret_eqw"] = mkt_ret
        feats["macro_mkt_vol21"] = mkt_ret.rolling(21, min_periods=5).std()
        feats["macro_mkt_vol63"] = mkt_ret.rolling(63, min_periods=10).std()
        feats["macro_cs_dispersion"] = log_ret.std(axis=1)

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            vol = (
                df.pivot_table(index="date", columns="ticker", values="volume", aggfunc="mean")
                .sort_index()
            )
            vol_mean = vol.mean(axis=1)
            feats["macro_log_volume"] = np.log1p(vol_mean.clip(lower=0.0))
            vol_base = vol_mean.rolling(21, min_periods=5).mean()
            feats["macro_volume_shock_21"] = (
                np.divide(vol_mean, vol_base + 1e-8) - 1.0
            )

        macro = pd.DataFrame(feats).sort_index()
        macro = macro.replace([np.inf, -np.inf], np.nan)
        if macro.shape[1] == 0:
            raise ValueError("Failed to build macro features from long-format macro prices CSV.")
        return macro

    raise ValueError(
        "macro.csv must include either a 'date' column (wide format) or "
        "'time' + 'ticker' columns (long format)."
    )


def load_static_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["src", "dst", "weight", "directed"])

    norm = {str(c).strip().lower().replace("-", "_").replace(" ", "_"): c for c in df.columns}

    def _pick(cands: list[str]) -> str | None:
        for cand in cands:
            if cand in norm:
                return norm[cand]
        return None

    src_col = _pick(["src", "source", "from", "from_ticker", "ticker_from", "u"])
    dst_col = _pick(["dst", "target", "to", "to_ticker", "ticker_to", "v"])
    if src_col is None or dst_col is None:
        raise ValueError(
            "static_edges.csv must include source/destination columns "
            "(e.g., src,dst or source,target)."
        )
    weight_col = _pick(["weight", "edge_weight", "strength", "w"])
    directed_col = _pick(["directed", "is_directed"])

    out = pd.DataFrame()
    out["src"] = df[src_col].astype(str).str.upper().str.strip()
    out["dst"] = df[dst_col].astype(str).str.upper().str.strip()
    if weight_col is not None:
        out["weight"] = pd.to_numeric(df[weight_col], errors="coerce")
    else:
        out["weight"] = 1.0
    if directed_col is not None:
        raw = df[directed_col]
        if raw.dtype == bool:
            out["directed"] = raw.astype(bool)
        else:
            txt = raw.astype(str).str.strip().str.lower()
            out["directed"] = txt.isin({"1", "true", "t", "yes", "y"})
    else:
        out["directed"] = False

    out = out.dropna(subset=["src", "dst", "weight"])
    out = out[(out["src"] != "") & (out["dst"] != "")]
    out["weight"] = out["weight"].astype(float)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["weight"])
    return out


def build_macro_features_from_market_data(
    returns: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    short_window: int = 21,
    long_window: int = 63,
) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    short_w = max(2, int(short_window))
    long_w = max(short_w + 1, int(long_window))

    rets = returns.sort_index()
    mkt_ret = rets.mean(axis=1)
    abs_ret = mkt_ret.abs()
    cs_disp = rets.std(axis=1)
    q90 = rets.quantile(0.9, axis=1)
    q10 = rets.quantile(0.1, axis=1)

    macro = pd.DataFrame(index=rets.index)
    macro["macro_mkt_ret_eqw"] = mkt_ret
    macro["macro_mkt_abs_ret"] = abs_ret
    macro["macro_mkt_vol_short"] = mkt_ret.rolling(short_w, min_periods=max(2, short_w // 3)).std()
    macro["macro_mkt_vol_long"] = mkt_ret.rolling(long_w, min_periods=max(3, long_w // 3)).std()
    macro["macro_mkt_vol_ratio"] = (
        macro["macro_mkt_vol_short"] / (macro["macro_mkt_vol_long"] + 1e-8)
    )
    macro["macro_mkt_trend_short"] = mkt_ret.rolling(short_w, min_periods=max(2, short_w // 3)).mean()
    macro["macro_mkt_trend_long"] = mkt_ret.rolling(long_w, min_periods=max(3, long_w // 3)).mean()
    macro["macro_cs_dispersion"] = cs_disp
    macro["macro_cs_tail_spread"] = q90 - q10

    if volume is not None and not volume.empty:
        vol = volume.sort_index()
        vol_mean = vol.mean(axis=1)
        vol_base_short = vol_mean.rolling(short_w, min_periods=max(2, short_w // 3)).mean()
        vol_base_long = vol_mean.rolling(long_w, min_periods=max(3, long_w // 3)).mean()
        macro["macro_log_volume"] = np.log1p(vol_mean.clip(lower=0.0))
        macro["macro_volume_shock_short"] = np.divide(vol_mean, vol_base_short + 1e-8) - 1.0
        macro["macro_volume_shock_long"] = np.divide(vol_mean, vol_base_long + 1e-8) - 1.0

    macro = macro.replace([np.inf, -np.inf], np.nan)
    return macro
