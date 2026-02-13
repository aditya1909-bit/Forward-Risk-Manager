from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm

FUND_COLS = ["sector_code", "market_cap", "pe_ratio", "debt_equity", "pb_ratio"]


@dataclass
class GraphBuildConfig:
    window: int = 20
    step: int = 1
    top_k: int | None = 10
    corr_threshold: float | None = None
    min_nodes: int = 50
    feature_mode: str = "window"  # "window", "last", "window_plus_summary", "window_plus_summary_fund"
    normalize: bool = True
    symmetric: bool = True
    rsi_period: int = 14
    mdy_ticker: str = "AUTO"
    edge_norm: bool = True
    edge_weight_mode: str = "raw"
    cross_sectional_norm: bool = False
    corr_method: str = "pearson"  # "pearson", "partial"
    partial_corr_ridge: float = 1e-3
    edge_select_mode: str = "top_k"  # "top_k", "threshold", "significance"
    significance_alpha: float = 0.05
    edge_node_weighting: str = "none"  # "none", "volume", "market_cap", "volume_market_cap"
    edge_node_weight_power: float = 0.5
    # Leakage controls: build graph inputs at t using only data up to t-lag.
    corr_lag_days: int = 0
    feature_lag_days: int = 0
    membership_lag_days: int = 0


def _select_edges(
    corr: np.ndarray,
    top_k: int | None,
    corr_threshold: float | None,
    symmetric: bool,
    mode: str = "top_k",
    significance_alpha: float = 0.05,
    n_obs: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = corr.shape[0]
    corr = corr.copy()
    np.fill_diagonal(corr, 0.0)

    src: List[int] = []
    dst: List[int] = []
    w: List[float] = []
    seen = set()

    mode_norm = str(mode).strip().lower()
    if mode_norm in ("", "auto"):
        mode_norm = "top_k" if top_k is not None else "threshold"

    if mode_norm == "top_k" and top_k is not None:
        k = max(1, min(top_k, n - 1))
        for i in range(n):
            row = corr[i]
            idx = np.argpartition(-np.abs(row), k)[:k]
            for j in idx:
                if i == j:
                    continue
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                src.append(i)
                dst.append(j)
                w.append(float(row[j]))
                if symmetric and (j, i) not in seen:
                    seen.add((j, i))
                    src.append(j)
                    dst.append(i)
                    w.append(float(row[j]))
    elif mode_norm == "threshold" and corr_threshold is not None:
        mask = np.abs(corr) >= corr_threshold
        np.fill_diagonal(mask, False)
        idx = np.argwhere(mask)
        for i, j in idx:
            if (i, j) in seen:
                continue
            seen.add((i, j))
            src.append(int(i))
            dst.append(int(j))
            w.append(float(corr[i, j]))
            if symmetric and (j, i) not in seen:
                seen.add((j, i))
                src.append(int(j))
                dst.append(int(i))
                w.append(float(corr[i, j]))
    elif mode_norm == "significance":
        if n_obs <= 3:
            return np.array(src), np.array(dst), np.array(w)
        alpha = float(np.clip(significance_alpha, 1e-8, 0.5))
        p = torch.tensor(1.0 - alpha / 2.0, dtype=torch.float64)
        z_crit = float(np.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0).item())
        z_stat = np.abs(np.arctanh(np.clip(corr, -0.999999, 0.999999))) * np.sqrt(n_obs - 3.0)
        mask = z_stat >= z_crit
        np.fill_diagonal(mask, False)
        if corr_threshold is not None:
            mask &= np.abs(corr) >= float(corr_threshold)

        if top_k is not None:
            k = max(1, min(int(top_k), n - 1))
            for i in range(n):
                row_idx = np.where(mask[i])[0]
                if row_idx.size == 0:
                    continue
                row_vals = np.abs(corr[i, row_idx])
                ord_idx = row_idx[np.argsort(-row_vals)[:k]]
                for j in ord_idx.tolist():
                    if i == j or (i, j) in seen:
                        continue
                    seen.add((i, j))
                    src.append(i)
                    dst.append(j)
                    w.append(float(corr[i, j]))
                    if symmetric and (j, i) not in seen:
                        seen.add((j, i))
                        src.append(j)
                        dst.append(i)
                        w.append(float(corr[i, j]))
        else:
            idx = np.argwhere(mask)
            for i, j in idx:
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                src.append(int(i))
                dst.append(int(j))
                w.append(float(corr[i, j]))
                if symmetric and (j, i) not in seen:
                    seen.add((j, i))
                    src.append(int(j))
                    dst.append(int(i))
                    w.append(float(corr[i, j]))
    else:
        raise ValueError(
            "Invalid edge selection settings. "
            f"mode={mode!r}, top_k={top_k}, corr_threshold={corr_threshold}"
        )

    return np.array(src), np.array(dst), np.array(w)


def _compute_rsi(returns: np.ndarray, period: int) -> np.ndarray:
    if returns.shape[1] == 0:
        return np.zeros(returns.shape[0])
    p = min(period, returns.shape[1])
    window = returns[:, -p:]
    gains = np.maximum(window, 0.0)
    losses = np.maximum(-window, 0.0)
    avg_gain = gains.mean(axis=1)
    avg_loss = losses.mean(axis=1) + 1e-8
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _signed_gcn_norm(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Preserve signed edge weights while normalizing by absolute weighted degree.
    row = edge_index[0]
    col = edge_index[1]
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    deg.scatter_add_(0, row, edge_weight.abs())
    deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, edge_weight


def _compute_summary_features(
    window_returns: pd.DataFrame,
    window_volume: pd.DataFrame | None,
    mdy_ticker: str,
    rsi_period: int,
) -> np.ndarray:
    rets = window_returns.to_numpy().T
    n = rets.shape[0]
    vol = np.nanstd(rets, axis=1)
    momentum = np.nansum(rets, axis=1)

    if window_volume is not None:
        vols = window_volume.to_numpy(dtype=float).T
        last = vols[:, -1].astype(float, copy=False)
        mean = np.nanmean(vols, axis=1)
        vol_shock = np.divide(last, mean, out=np.ones(last.shape, dtype=float), where=mean > 0)
    else:
        vol_shock = np.ones(n)

    tickers = list(window_returns.columns)
    ref = str(mdy_ticker or "").upper().strip()
    use_explicit = ref not in {"", "AUTO", "AUTO_DETECT", "AUTO-DETECT"} and ref in tickers

    if use_explicit:
        ref_idx = tickers.index(ref)
        market = rets[ref_idx]
    else:
        # Fallback when explicit market ticker is unavailable: equal-weight market proxy.
        market = np.nanmean(rets, axis=0)
        ref_idx = -1

    market_c = market - np.nanmean(market)
    market_std = np.nanstd(market_c) + 1e-8
    beta = np.zeros(n)
    for i in range(n):
        if use_explicit and i == ref_idx:
            beta[i] = 1.0
            continue
        xi = rets[i]
        xi_c = xi - np.nanmean(xi)
        denom = (np.nanstd(xi_c) + 1e-8) * market_std
        cov = np.nanmean(xi_c * market_c)
        beta[i] = cov / denom if denom > 0 else 0.0

    rsi = _compute_rsi(rets, rsi_period)
    summary = np.stack([vol, momentum, vol_shock, beta, rsi], axis=1)
    return summary


def _safe_corr_matrix(window_df: pd.DataFrame) -> np.ndarray:
    x = window_df.to_numpy()
    x = x - np.nanmean(x, axis=0, keepdims=True)
    cov = np.nan_to_num((x.T @ x) / max(1, x.shape[0] - 1))
    std = np.nanstd(x, axis=0) + 1e-8
    denom = std[:, None] * std[None, :]
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    return corr


def _safe_partial_corr_matrix(window_df: pd.DataFrame, ridge: float = 1e-3) -> np.ndarray:
    x = window_df.to_numpy(dtype=float)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    cov = np.nan_to_num((x.T @ x) / max(1, x.shape[0] - 1))
    n = cov.shape[0]
    if n == 0:
        return cov
    ridge_scale = float(max(0.0, ridge))
    diag_scale = float(np.nanmean(np.diag(cov))) if np.isfinite(cov).any() else 1.0
    if not np.isfinite(diag_scale) or diag_scale <= 0:
        diag_scale = 1.0
    cov_reg = cov + (ridge_scale * diag_scale) * np.eye(n, dtype=cov.dtype)
    precision = np.linalg.pinv(cov_reg)
    d = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    denom = d[:, None] * d[None, :]
    partial = np.divide(-precision, denom, out=np.zeros_like(precision), where=denom > 0)
    np.fill_diagonal(partial, 0.0)
    partial = np.nan_to_num(partial, nan=0.0, posinf=0.0, neginf=0.0)
    return partial


def _prepare_fundamentals_panel(
    fundamentals: pd.DataFrame | None,
    dates: List[str],
) -> tuple[pd.DataFrame | None, List[str]]:
    if fundamentals is None or fundamentals.empty:
        return None, []
    cols = [c for c in FUND_COLS if c in fundamentals.columns]
    if not cols:
        return None, []
    df = fundamentals[["date", "ticker"] + cols].dropna(subset=["date", "ticker"]).copy()
    df = df.sort_values(["ticker", "date"])
    frames = []
    for ticker, g in df.groupby("ticker"):
        g = g.drop_duplicates("date").set_index("date").sort_index()
        g = g.reindex(dates, method="ffill")
        g["ticker"] = ticker
        frames.append(g.reset_index())
    if not frames:
        return None, cols
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.set_index(["date", "ticker"]).sort_index()
    # Ensure unique index (date, ticker)
    panel = panel[~panel.index.duplicated(keep="last")]
    return panel, cols


def _prepare_macro_panel(
    macro: pd.DataFrame | None,
    dates: List[str],
) -> tuple[pd.DataFrame | None, List[str]]:
    if macro is None or macro.empty:
        return None, []
    panel = macro.copy()
    if "date" in panel.columns:
        panel = panel.set_index("date")
    panel.index = panel.index.astype(str)
    cols = [c for c in panel.columns]
    if not cols:
        return None, []
    panel = panel[cols].apply(pd.to_numeric, errors="coerce")
    panel = panel.reindex(dates).ffill()
    panel = panel.fillna(0.0)
    return panel, cols


def _build_node_features(
    window_df: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    feature_mode: str,
    normalize: bool,
    cross_sectional_norm: bool,
    mdy_ticker: str,
    rsi_period: int,
    macro_features: np.ndarray | None = None,
    fund_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if feature_mode == "window":
        values = window_df.to_numpy().T
    elif feature_mode == "last":
        values = window_df.iloc[-1].to_numpy()[:, None]
    elif feature_mode in ("window_plus_summary", "window_plus_summary_fund"):
        values = window_df.to_numpy().T
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    if cross_sectional_norm and values.size:
        cs_mean = np.nanmean(values, axis=0, keepdims=True)
        cs_std = np.nanstd(values, axis=0, keepdims=True) + 1e-8
        values = (values - cs_mean) / cs_std

    ret_mean = None
    ret_std = None
    if normalize and feature_mode in ("window", "window_plus_summary", "window_plus_summary_fund"):
        ret_mean = np.nanmean(values, axis=1, keepdims=True)
        ret_std = np.nanstd(values, axis=1, keepdims=True) + 1e-8
        values = (values - ret_mean) / ret_std

    if feature_mode in ("window_plus_summary", "window_plus_summary_fund"):
        summary = _compute_summary_features(window_df, volume_df, mdy_ticker, rsi_period)
        if normalize:
            s_mean = np.nanmean(summary, axis=0, keepdims=True)
            s_std = np.nanstd(summary, axis=0, keepdims=True) + 1e-8
            summary = (summary - s_mean) / s_std
        values = np.concatenate([values, summary], axis=1)

    if macro_features is not None and macro_features.size:
        macro = macro_features.copy()
        if normalize:
            m_mean = np.nanmean(macro, axis=0, keepdims=True)
            m_std = np.nanstd(macro, axis=0, keepdims=True) + 1e-8
            macro = (macro - m_mean) / m_std
        values = np.concatenate([values, macro], axis=1)

    if feature_mode == "window_plus_summary_fund" and fund_features is not None:
        fund = fund_features.copy()
        # Fill missing fundamentals with per-column median (or zero if all missing)
        med = np.nanmedian(fund, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        inds = np.where(np.isnan(fund))
        if inds[0].size:
            fund[inds] = np.take(med, inds[1])
        if normalize:
            f_mean = np.nanmean(fund, axis=0, keepdims=True)
            f_std = np.nanstd(fund, axis=0, keepdims=True) + 1e-8
            fund = (fund - f_mean) / f_std
        values = np.concatenate([values, fund], axis=1)

    return values, ret_mean, ret_std


def _window_to_graph_data(
    end_idx: int,
    dates: List[str],
    returns: pd.DataFrame,
    volume: pd.DataFrame | None,
    membership_map: Dict[str, List[str]],
    config: GraphBuildConfig,
    fund_panel: pd.DataFrame | None,
    fund_cols: List[str],
    macro_panel: pd.DataFrame | None,
    macro_cols: List[str],
):
    end_date = dates[end_idx]
    corr_end_idx = end_idx - max(0, int(config.corr_lag_days))
    feature_end_idx = end_idx - max(0, int(config.feature_lag_days))
    member_end_idx = end_idx - max(0, int(config.membership_lag_days))
    if min(corr_end_idx, feature_end_idx, member_end_idx) < 0:
        return None, "lag_history"

    corr_start_idx = corr_end_idx - config.window + 1
    feature_start_idx = feature_end_idx - config.window + 1
    if corr_start_idx < 0 or feature_start_idx < 0:
        return None, "lag_history"

    member_date = dates[member_end_idx]
    members = membership_map.get(member_date)
    if not members:
        return None, "no_members"

    corr_window_df = returns.iloc[corr_start_idx : corr_end_idx + 1]
    window_df = returns.iloc[feature_start_idx : feature_end_idx + 1]
    window_volume = None
    if volume is not None:
        window_volume = volume.iloc[feature_start_idx : feature_end_idx + 1]

    cols = [t for t in members if t in window_df.columns and t in corr_window_df.columns]
    if not cols:
        return None, "no_cols"
    corr_window_df = corr_window_df[cols]
    window_df = window_df[cols]

    valid_cols = corr_window_df.notna().all(axis=0) & window_df.notna().all(axis=0)
    cols = [c for c in cols if bool(valid_cols.get(c, False))]
    if not cols:
        return None, "no_cols"
    corr_window_df = corr_window_df[cols]
    window_df = window_df[cols]

    if window_volume is not None:
        window_volume = window_volume[cols]
        vol_cols = window_volume.notna().all(axis=0)
        cols = [c for c in cols if bool(vol_cols.get(c, False))]
        if not cols:
            return None, "no_cols"
        corr_window_df = corr_window_df[cols]
        window_df = window_df[cols]
        window_volume = window_volume[cols]
    # Use the post-dropna columns for all downstream alignment
    cols = list(window_df.columns)
    if window_df.shape[1] < config.min_nodes:
        return None, "min_nodes"

    corr_method = str(config.corr_method).strip().lower()
    if corr_method == "partial":
        corr = _safe_partial_corr_matrix(corr_window_df, ridge=float(config.partial_corr_ridge))
    else:
        corr = _safe_corr_matrix(corr_window_df)
    src, dst, w = _select_edges(
        corr,
        config.top_k,
        config.corr_threshold,
        config.symmetric,
        mode=config.edge_select_mode,
        significance_alpha=float(config.significance_alpha),
        n_obs=int(corr_window_df.shape[0]),
    )
    if len(src) == 0:
        return None, "no_edges"

    fund_features = None
    fund_slice = None
    if config.feature_mode == "window_plus_summary_fund" and fund_panel is not None:
        feature_end_date = dates[feature_end_idx]
        try:
            fund_slice = fund_panel.loc[feature_end_date]
        except KeyError:
            fund_slice = None
        if fund_slice is not None and fund_cols:
            if isinstance(fund_slice, pd.Series):
                fund_slice = fund_slice.to_frame().T
            if fund_slice.index.has_duplicates:
                fund_slice = fund_slice.groupby(level=0).last()
            fund_slice = fund_slice.reindex(cols, axis=0)
            try:
                fund_features = fund_slice[fund_cols].to_numpy()
            except KeyError:
                fund_features = None
            if fund_features is not None and fund_features.shape[0] != len(cols):
                # Fallback: skip fundamentals for this window if misaligned
                fund_features = None

    macro_features = None
    if macro_panel is not None and macro_cols:
        feature_end_date = dates[feature_end_idx]
        if feature_end_date in macro_panel.index:
            macro_vec = macro_panel.loc[feature_end_date, macro_cols].to_numpy(dtype=float)
            macro_features = np.tile(macro_vec[None, :], (len(cols), 1))

    node_weight_mode = str(config.edge_node_weighting).strip().lower()
    if node_weight_mode not in {"none", "volume", "market_cap", "volume_market_cap"}:
        node_weight_mode = "none"
    if node_weight_mode != "none":
        node_w = np.ones(len(cols), dtype=float)
        if node_weight_mode in {"volume", "volume_market_cap"} and window_volume is not None:
            vol_w = window_volume.mean(axis=0).to_numpy(dtype=float)
            vol_w = np.nan_to_num(vol_w, nan=1.0, posinf=1.0, neginf=1.0)
            vol_w = np.clip(vol_w, 1e-8, None)
            node_w *= vol_w
        if node_weight_mode in {"market_cap", "volume_market_cap"}:
            cap_w = np.ones(len(cols), dtype=float)
            if fund_slice is not None and "market_cap" in getattr(fund_slice, "columns", []):
                cap_s = fund_slice.reindex(cols, axis=0)["market_cap"]
                cap_w = cap_s.to_numpy(dtype=float)
                cap_w = np.nan_to_num(cap_w, nan=1.0, posinf=1.0, neginf=1.0)
                cap_w = np.clip(cap_w, 1e-8, None)
            node_w *= cap_w

        med = float(np.nanmedian(node_w)) if np.isfinite(node_w).any() else 1.0
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        node_w = np.clip(node_w / med, 1e-3, 1e3)
        power = float(max(0.0, config.edge_node_weight_power))
        edge_scale = np.power(node_w[src] * node_w[dst], power)
        w = w * edge_scale

    x, ret_mean, ret_std = _build_node_features(
        window_df,
        window_volume,
        config.feature_mode,
        config.normalize,
        config.cross_sectional_norm,
        config.mdy_ticker,
        config.rsi_period,
        macro_features,
        fund_features,
    )
    return (end_date, list(window_df.columns), src, dst, w, x, ret_mean, ret_std), "ok"


def build_rolling_corr_graphs(
    returns: pd.DataFrame,
    volume: pd.DataFrame | None,
    membership_map: Dict[str, List[str]],
    config: GraphBuildConfig,
    fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    num_workers: int = 1,
    parallel_backend: str | None = "threadpool",
    joblib_prefer: str = "threads",
    joblib_n_jobs: int | None = None,
    progress: bool = True,
) -> Tuple[List[Data], List[str], List[List[str]], Dict[str, int]]:
    dates = list(returns.index)
    fund_panel, fund_cols = _prepare_fundamentals_panel(fundamentals, dates)
    macro_panel, macro_cols = _prepare_macro_panel(macro, dates)
    graphs: List[Data] = []
    graph_dates: List[str] = []
    node_tickers: List[List[str]] = []
    stats = {
        "total_windows": 0,
        "skipped_lag_history": 0,
        "skipped_no_members": 0,
        "skipped_no_cols": 0,
        "skipped_min_nodes": 0,
        "skipped_no_edges": 0,
        "built": 0,
    }

    end_indices = list(range(config.window - 1, len(dates), config.step))

    def _task(end_idx: int):
        return _window_to_graph_data(
            end_idx,
            dates,
            returns,
            volume,
            membership_map,
            config,
            fund_panel,
            fund_cols,
            macro_panel,
            macro_cols,
        )

    backend = (parallel_backend or "threadpool").lower()
    use_parallel = num_workers is not None and num_workers > 1 and backend not in ("none", "serial")

    if use_parallel and backend in ("joblib", "loky"):
        try:
            from joblib import Parallel, delayed  # type: ignore

            n_jobs = joblib_n_jobs if joblib_n_jobs is not None else num_workers
            if progress and joblib_prefer == "threads":
                pbar = tqdm(
                    total=len(end_indices),
                    desc="Building graphs",
                    unit="win",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )

                def _task_pbar(end_idx: int):
                    res = _task(end_idx)
                    pbar.update(1)
                    return res

                results = Parallel(
                    n_jobs=n_jobs,
                    prefer=joblib_prefer,
                    batch_size="auto",
                )(delayed(_task_pbar)(end_idx) for end_idx in end_indices)
                pbar.close()
            else:
                results = Parallel(
                    n_jobs=n_jobs,
                    prefer=joblib_prefer,
                    batch_size="auto",
                )(delayed(_task)(end_idx) for end_idx in end_indices)
        except Exception as exc:
            print(f"joblib parallel failed ({exc}); falling back to ThreadPoolExecutor")
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                it = executor.map(_task, end_indices)
                if progress:
                    it = tqdm(
                        it,
                        total=len(end_indices),
                        desc="Building graphs",
                        unit="win",
                        dynamic_ncols=True,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    )
                results = list(it)
    elif use_parallel:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            it = executor.map(_task, end_indices)
            if progress:
                it = tqdm(
                    it,
                    total=len(end_indices),
                    desc="Building graphs",
                    unit="win",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )
            results = list(it)
    else:
        it = end_indices
        if progress:
            it = tqdm(
                it,
                total=len(end_indices),
                desc="Building graphs",
                unit="win",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        results = [_task(end_idx) for end_idx in it]

    stats["total_windows"] = len(results)
    for result, reason in results:
        if result is None:
            if reason == "lag_history":
                stats["skipped_lag_history"] += 1
            elif reason == "no_members":
                stats["skipped_no_members"] += 1
            elif reason == "no_cols":
                stats["skipped_no_cols"] += 1
            elif reason == "min_nodes":
                stats["skipped_min_nodes"] += 1
            elif reason == "no_edges":
                stats["skipped_no_edges"] += 1
            continue
        end_date, tickers, src, dst, w, x, ret_mean, ret_std = result
        edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        edge_attr = torch.from_numpy(w).float().unsqueeze(-1)
        edge_weight = edge_attr.squeeze(-1)
        if config.edge_weight_mode == "abs":
            edge_weight = edge_weight.abs()
        elif config.edge_weight_mode == "ones":
            edge_weight = torch.ones_like(edge_weight)
        if config.edge_norm:
            if config.edge_weight_mode == "raw":
                edge_index, edge_weight = _signed_gcn_norm(
                    edge_index, edge_weight, num_nodes=len(tickers)
                )
            else:
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, num_nodes=len(tickers), add_self_loops=False
                )

        x_tensor = torch.from_numpy(x).float()
        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(tickers))
        data.edge_weight = edge_weight
        if ret_mean is not None and ret_std is not None:
            data.ret_mean = torch.tensor(ret_mean.squeeze(1), dtype=torch.float32)
            data.ret_std = torch.tensor(ret_std.squeeze(1), dtype=torch.float32)

        graphs.append(data)
        graph_dates.append(end_date)
        node_tickers.append(tickers)
        stats["built"] += 1

    return graphs, graph_dates, node_tickers, stats
