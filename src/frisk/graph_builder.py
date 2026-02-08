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
    mdy_ticker: str = "MDY"
    edge_norm: bool = True
    edge_weight_mode: str = "abs"


def _select_edges(
    corr: np.ndarray,
    top_k: int | None,
    corr_threshold: float | None,
    symmetric: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = corr.shape[0]
    corr = corr.copy()
    np.fill_diagonal(corr, 0.0)

    src: List[int] = []
    dst: List[int] = []
    w: List[float] = []
    seen = set()

    if top_k is not None:
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
    elif corr_threshold is not None:
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
    else:
        raise ValueError("Either top_k or corr_threshold must be set")

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
        vols = window_volume.to_numpy().T
        last = vols[:, -1]
        mean = np.nanmean(vols, axis=1)
        vol_shock = np.divide(last, mean, out=np.ones_like(last), where=mean > 0)
    else:
        vol_shock = np.ones(n)

    tickers = list(window_returns.columns)
    if mdy_ticker in tickers:
        mdy_idx = tickers.index(mdy_ticker)
        mdy = rets[mdy_idx]
        mdy_c = mdy - np.nanmean(mdy)
        mdy_std = np.nanstd(mdy_c) + 1e-8
        beta = np.zeros(n)
        for i in range(n):
            if i == mdy_idx:
                beta[i] = 1.0
                continue
            xi = rets[i]
            xi_c = xi - np.nanmean(xi)
            denom = (np.nanstd(xi_c) + 1e-8) * mdy_std
            cov = np.nanmean(xi_c * mdy_c)
            beta[i] = cov / denom if denom > 0 else 0.0
    else:
        beta = np.zeros(n)

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


def _build_node_features(
    window_df: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    feature_mode: str,
    normalize: bool,
    mdy_ticker: str,
    rsi_period: int,
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

    ret_mean = None
    ret_std = None
    if normalize and feature_mode in ("window", "window_plus_summary"):
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
):
    end_date = dates[end_idx]
    members = membership_map.get(end_date)
    if not members:
        return None, "no_members"

    window_df = returns.iloc[end_idx - config.window + 1 : end_idx + 1]
    window_volume = None
    if volume is not None:
        window_volume = volume.iloc[end_idx - config.window + 1 : end_idx + 1]
    cols = [t for t in members if t in window_df.columns]
    if not cols:
        return None, "no_cols"
    window_df = window_df[cols].dropna(axis=1, how="any")
    if window_volume is not None:
        window_volume = window_volume[window_df.columns]
    # Use the post-dropna columns for all downstream alignment
    cols = list(window_df.columns)
    if window_df.shape[1] < config.min_nodes:
        return None, "min_nodes"

    corr = _safe_corr_matrix(window_df)
    src, dst, w = _select_edges(corr, config.top_k, config.corr_threshold, config.symmetric)
    if len(src) == 0:
        return None, "no_edges"

    fund_features = None
    if config.feature_mode == "window_plus_summary_fund" and fund_panel is not None:
        try:
            fund_slice = fund_panel.loc[end_date]
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

    x, ret_mean, ret_std = _build_node_features(
        window_df,
        window_volume,
        config.feature_mode,
        config.normalize,
        config.mdy_ticker,
        config.rsi_period,
        fund_features,
    )
    return (end_date, list(window_df.columns), src, dst, w, x, ret_mean, ret_std), "ok"


def build_rolling_corr_graphs(
    returns: pd.DataFrame,
    volume: pd.DataFrame | None,
    membership_map: Dict[str, List[str]],
    config: GraphBuildConfig,
    fundamentals: pd.DataFrame | None = None,
    num_workers: int = 1,
    parallel_backend: str | None = "threadpool",
    joblib_prefer: str = "threads",
    joblib_n_jobs: int | None = None,
    progress: bool = True,
) -> Tuple[List[Data], List[str], List[List[str]], Dict[str, int]]:
    dates = list(returns.index)
    fund_panel, fund_cols = _prepare_fundamentals_panel(fundamentals, dates)
    graphs: List[Data] = []
    graph_dates: List[str] = []
    node_tickers: List[List[str]] = []
    stats = {
        "total_windows": 0,
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
            if reason == "no_members":
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
