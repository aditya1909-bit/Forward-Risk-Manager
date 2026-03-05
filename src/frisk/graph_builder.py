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
EDGE_REL_CORR_POS = 1
EDGE_REL_CORR_NEG = 2
EDGE_REL_LEAD_LAG = 4
EDGE_REL_SECTOR_STATIC = 8
EDGE_REL_STATIC_OVERLAY = 16
EDGE_REL_UNKNOWN = 0


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
    lead_lag_enabled: bool = False
    lead_lag_max_lag: int = 3
    lead_lag_top_k: int | None = 2
    lead_lag_threshold: float | None = None
    lead_lag_weight: float = 0.5
    lead_lag_mode: str = "top_k"  # "top_k", "threshold", "significance"
    sector_static_enabled: bool = False
    sector_static_weight: float = 0.25
    sector_static_top_k: int | None = 4
    static_edge_weight: float = 0.25
    # Leakage controls: build graph inputs at t using only data up to t-lag.
    corr_lag_days: int = 0
    feature_lag_days: int = 0
    membership_lag_days: int = 0
    macro_lag_days: int = 0


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


def _safe_lagged_corr_matrix(window_df: pd.DataFrame, lag: int) -> np.ndarray:
    x = window_df.to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        return np.zeros((0, 0), dtype=float)
    lag_n = max(1, int(lag))
    if x.shape[0] <= lag_n:
        return np.zeros((x.shape[1], x.shape[1]), dtype=float)
    x_src = x[:-lag_n]
    x_dst = x[lag_n:]
    x_src = x_src - np.nanmean(x_src, axis=0, keepdims=True)
    x_dst = x_dst - np.nanmean(x_dst, axis=0, keepdims=True)
    src_std = np.nanstd(x_src, axis=0) + 1e-8
    dst_std = np.nanstd(x_dst, axis=0) + 1e-8
    src_z = np.divide(x_src, src_std[None, :], out=np.zeros_like(x_src), where=src_std[None, :] > 0)
    dst_z = np.divide(x_dst, dst_std[None, :], out=np.zeros_like(x_dst), where=dst_std[None, :] > 0)
    corr = np.nan_to_num((src_z.T @ dst_z) / max(1, src_z.shape[0] - 1))
    np.fill_diagonal(corr, 0.0)
    return corr


def _merge_edges(
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
    rel_mask: np.ndarray,
    rel_lag: np.ndarray,
    extra_edges: list[tuple[int, int, float, int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not extra_edges:
        return src, dst, w, rel_mask, rel_lag
    edge_map: dict[tuple[int, int], tuple[float, int, float]] = {}
    for si, di, wi, rmi, rli in zip(
        src.tolist(),
        dst.tolist(),
        w.tolist(),
        rel_mask.tolist(),
        rel_lag.tolist(),
    ):
        edge_map[(int(si), int(di))] = (float(wi), int(rmi), float(rli))
    for si, di, wi, rel_i, lag_i in extra_edges:
        key = (int(si), int(di))
        if key in edge_map:
            prev_w, prev_rel, prev_lag = edge_map[key]
            edge_map[key] = (
                float(prev_w + float(wi)),
                int(prev_rel | int(rel_i)),
                float(max(prev_lag, float(lag_i))),
            )
        else:
            edge_map[key] = (float(wi), int(rel_i), float(lag_i))
    keys = sorted(edge_map.keys())
    if not keys:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
        )
    src_out = []
    dst_out = []
    w_out = []
    rel_out = []
    lag_out = []
    for key in keys:
        val, rel_i, lag_i = edge_map[key]
        if not np.isfinite(val) or abs(val) < 1e-12:
            continue
        src_out.append(key[0])
        dst_out.append(key[1])
        w_out.append(val)
        rel_out.append(int(rel_i))
        lag_out.append(float(max(0.0, lag_i)))
    return (
        np.array(src_out, dtype=int),
        np.array(dst_out, dtype=int),
        np.array(w_out, dtype=float),
        np.array(rel_out, dtype=np.int64),
        np.array(lag_out, dtype=float),
    )


def _build_lead_lag_edges(
    corr_window_df: pd.DataFrame,
    config: GraphBuildConfig,
) -> list[tuple[int, int, float, int, float]]:
    if not bool(config.lead_lag_enabled):
        return []
    max_lag = max(1, int(config.lead_lag_max_lag))
    mode = str(config.lead_lag_mode or "top_k").strip().lower()
    if mode in {"", "auto"}:
        mode = "top_k" if config.lead_lag_top_k is not None else "threshold"
    if mode not in {"top_k", "threshold", "significance"}:
        mode = "top_k"
    if mode == "top_k" and config.lead_lag_top_k is None and config.lead_lag_threshold is not None:
        mode = "threshold"
    if mode == "threshold" and config.lead_lag_threshold is None and config.lead_lag_top_k is not None:
        mode = "top_k"

    lag_edge_map: dict[tuple[int, int], tuple[float, int]] = {}
    for lag in range(1, max_lag + 1):
        if corr_window_df.shape[0] <= lag + 1:
            break
        lag_corr = _safe_lagged_corr_matrix(corr_window_df, lag=lag)
        src_l, dst_l, w_l = _select_edges(
            lag_corr,
            config.lead_lag_top_k,
            config.lead_lag_threshold,
            symmetric=False,
            mode=mode,
            significance_alpha=float(config.significance_alpha),
            n_obs=int(corr_window_df.shape[0] - lag),
        )
        for si, di, wi in zip(src_l.tolist(), dst_l.tolist(), w_l.tolist()):
            key = (int(si), int(di))
            val = float(wi)
            if key not in lag_edge_map or abs(val) > abs(lag_edge_map[key][0]):
                lag_edge_map[key] = (val, int(lag))

    scale = float(max(0.0, config.lead_lag_weight))
    if scale == 0.0:
        return []
    out: list[tuple[int, int, float, int, float]] = []
    for (si, di), (wi, lag_i) in lag_edge_map.items():
        out.append(
            (
                si,
                di,
                scale * float(wi),
                EDGE_REL_LEAD_LAG,
                float(max(0, int(lag_i))),
            )
        )
    return out


def _edge_primary_type_from_mask(rel_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(rel_mask.shape, dtype=np.int64)
    for i, raw in enumerate(rel_mask.tolist()):
        mask = int(raw)
        if mask == 0:
            out[i] = 0
        elif (mask & EDGE_REL_LEAD_LAG) and (mask & (mask - 1)) == 0:
            out[i] = 3
        elif (mask & EDGE_REL_STATIC_OVERLAY) and (mask & (mask - 1)) == 0:
            out[i] = 5
        elif (mask & EDGE_REL_SECTOR_STATIC) and (mask & (mask - 1)) == 0:
            out[i] = 4
        elif (mask & EDGE_REL_CORR_NEG) and (mask & (mask - 1)) == 0:
            out[i] = 2
        elif (mask & EDGE_REL_CORR_POS) and (mask & (mask - 1)) == 0:
            out[i] = 1
        else:
            # Multi-source overlap: keep a dedicated relation id.
            out[i] = 6
    return out


def _prepare_static_edge_map(
    static_edges: pd.DataFrame | None,
) -> Dict[str, List[tuple[str, float, bool]]]:
    if static_edges is None or static_edges.empty:
        return {}
    df = static_edges.copy()
    if not {"src", "dst"}.issubset(df.columns):
        return {}
    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "directed" not in df.columns:
        df["directed"] = False
    df["src"] = df["src"].astype(str).str.upper().str.strip()
    df["dst"] = df["dst"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    if df["directed"].dtype == bool:
        df["directed"] = df["directed"].astype(bool)
    else:
        d_txt = df["directed"].astype(str).str.strip().str.lower()
        df["directed"] = d_txt.isin({"1", "true", "t", "yes", "y"})
    df = df[(df["src"] != "") & (df["dst"] != "")]
    edge_map: Dict[str, List[tuple[str, float, bool]]] = {}
    for row in df.itertuples(index=False):
        edge_map.setdefault(str(row.src), []).append(
            (str(row.dst), float(row.weight), bool(row.directed))
        )
    return edge_map


def _prepare_fundamentals_panel(
    fundamentals: pd.DataFrame | None,
    dates: List[str],
) -> tuple[pd.DataFrame | None, List[str]]:
    if fundamentals is None or fundamentals.empty:
        return None, []
    cols: List[str] = []
    for c in fundamentals.columns:
        if c in {"date", "ticker"}:
            continue
        s = pd.to_numeric(fundamentals[c], errors="coerce")
        if s.notna().any():
            cols.append(c)
    if not cols:
        return None, []
    df = fundamentals[["date", "ticker"] + cols].dropna(subset=["date", "ticker"]).copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
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
            m_std = np.nanstd(macro, axis=0, keepdims=True)
            # Macro vectors are often global (same value repeated for each node).
            # In that case cross-node std is ~0 and z-scoring would zero out the signal.
            if np.nanmax(m_std) > 1e-8:
                macro = (macro - m_mean) / (m_std + 1e-8)
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
    static_edge_map: Dict[str, List[tuple[str, float, bool]]],
):
    end_date = dates[end_idx]
    corr_end_idx = end_idx - max(0, int(config.corr_lag_days))
    feature_end_idx = end_idx - max(0, int(config.feature_lag_days))
    member_end_idx = end_idx - max(0, int(config.membership_lag_days))
    macro_end_idx = feature_end_idx - max(0, int(config.macro_lag_days))
    if min(corr_end_idx, feature_end_idx, member_end_idx, macro_end_idx) < 0:
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
    edge_mode = str(config.edge_select_mode or "top_k").strip().lower()
    if edge_mode in {"", "auto"}:
        edge_mode = "top_k" if config.top_k is not None else "threshold"
    if edge_mode == "top_k" and config.top_k is None and config.corr_threshold is not None:
        edge_mode = "threshold"
    if edge_mode == "threshold" and config.corr_threshold is None and config.top_k is not None:
        edge_mode = "top_k"
    src, dst, w = _select_edges(
        corr,
        config.top_k,
        config.corr_threshold,
        config.symmetric,
        mode=edge_mode,
        significance_alpha=float(config.significance_alpha),
        n_obs=int(corr_window_df.shape[0]),
    )
    rel_mask = np.where(w >= 0.0, EDGE_REL_CORR_POS, EDGE_REL_CORR_NEG).astype(np.int64)
    rel_lag = np.zeros_like(w, dtype=float)

    extra_edges: list[tuple[int, int, float, int, float]] = []
    lead_lag_edges = _build_lead_lag_edges(corr_window_df, config)
    if lead_lag_edges:
        extra_edges.extend(lead_lag_edges)

    fund_features = None
    fund_slice = None
    if fund_panel is not None:
        feature_end_date = dates[feature_end_idx]
        try:
            fund_slice = fund_panel.loc[feature_end_date]
        except KeyError:
            fund_slice = None
        if fund_slice is not None:
            if isinstance(fund_slice, pd.Series):
                fund_slice = fund_slice.to_frame().T
            if fund_slice.index.has_duplicates:
                fund_slice = fund_slice.groupby(level=0).last()
            fund_slice = fund_slice.reindex(cols, axis=0)
        if config.feature_mode == "window_plus_summary_fund" and fund_slice is not None and fund_cols:
            try:
                fund_features = fund_slice[fund_cols].to_numpy()
            except KeyError:
                fund_features = None
            if fund_features is not None and fund_features.shape[0] != len(cols):
                # Fallback: skip fundamentals for this window if misaligned
                fund_features = None

    if bool(config.sector_static_enabled) and fund_slice is not None and "sector_code" in getattr(
        fund_slice, "columns", []
    ):
        sector_weight = float(max(0.0, config.sector_static_weight))
        sector_top_k = config.sector_static_top_k
        if sector_weight > 0:
            sector_codes = pd.to_numeric(
                fund_slice.reindex(cols, axis=0)["sector_code"],
                errors="coerce",
            )
            groups: Dict[int, List[int]] = {}
            for idx_i, sec in enumerate(sector_codes.tolist()):
                if not np.isfinite(sec):
                    continue
                groups.setdefault(int(sec), []).append(idx_i)
            for idxs in groups.values():
                if len(idxs) < 2:
                    continue
                per_node = len(idxs) - 1
                if sector_top_k is not None:
                    per_node = min(per_node, max(1, int(sector_top_k)))
                for i in idxs:
                    peers = [j for j in idxs if j != i]
                    if not peers:
                        continue
                    if per_node < len(peers):
                        peer_strength = np.abs(corr[i, peers])
                        order = np.argsort(-peer_strength)[:per_node]
                        peers = [peers[k] for k in order.tolist()]
                    for j in peers:
                        extra_edges.append(
                            (int(i), int(j), sector_weight, EDGE_REL_SECTOR_STATIC, 0.0)
                        )

    if static_edge_map:
        ticker_to_idx = {t: i for i, t in enumerate(cols)}
        static_scale = float(max(0.0, config.static_edge_weight))
        if static_scale > 0:
            for src_ticker, src_i in ticker_to_idx.items():
                for dst_ticker, edge_weight_raw, directed in static_edge_map.get(src_ticker, []):
                    dst_i = ticker_to_idx.get(dst_ticker)
                    if dst_i is None or dst_i == src_i:
                        continue
                    ew = static_scale * float(edge_weight_raw)
                    if not np.isfinite(ew) or ew == 0.0:
                        continue
                    extra_edges.append(
                        (
                            int(src_i),
                            int(dst_i),
                            ew,
                            EDGE_REL_STATIC_OVERLAY,
                            0.0,
                        )
                    )
                    if not directed:
                        extra_edges.append(
                            (
                                int(dst_i),
                                int(src_i),
                                ew,
                                EDGE_REL_STATIC_OVERLAY,
                                0.0,
                            )
                        )

    if extra_edges:
        src, dst, w, rel_mask, rel_lag = _merge_edges(src, dst, w, rel_mask, rel_lag, extra_edges)
    if len(src) == 0:
        return None, "no_edges"

    macro_features = None
    if macro_panel is not None and macro_cols:
        macro_end_date = dates[macro_end_idx]
        if macro_end_date in macro_panel.index:
            macro_vec = macro_panel.loc[macro_end_date, macro_cols].to_numpy(dtype=float)
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
    return (
        end_date,
        list(window_df.columns),
        src,
        dst,
        w,
        rel_mask,
        rel_lag,
        x,
        ret_mean,
        ret_std,
    ), "ok"


def build_rolling_corr_graphs(
    returns: pd.DataFrame,
    volume: pd.DataFrame | None,
    membership_map: Dict[str, List[str]],
    config: GraphBuildConfig,
    fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    static_edges: pd.DataFrame | None = None,
    num_workers: int = 1,
    parallel_backend: str | None = "threadpool",
    joblib_prefer: str = "threads",
    joblib_n_jobs: int | None = None,
    progress: bool = True,
) -> Tuple[List[Data], List[str], List[List[str]], Dict[str, int]]:
    dates = list(returns.index)
    fund_panel, fund_cols = _prepare_fundamentals_panel(fundamentals, dates)
    macro_panel, macro_cols = _prepare_macro_panel(macro, dates)
    static_edge_map = _prepare_static_edge_map(static_edges)
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

    raw_end_indices = list(range(config.window - 1, len(dates), config.step))
    eligible_end_indices: List[int] = []
    for end_idx in raw_end_indices:
        corr_end_idx = end_idx - max(0, int(config.corr_lag_days))
        feature_end_idx = end_idx - max(0, int(config.feature_lag_days))
        member_end_idx = end_idx - max(0, int(config.membership_lag_days))
        macro_end_idx = feature_end_idx - max(0, int(config.macro_lag_days))
        if min(corr_end_idx, feature_end_idx, member_end_idx, macro_end_idx) < 0:
            stats["skipped_lag_history"] += 1
            continue
        corr_start_idx = corr_end_idx - config.window + 1
        feature_start_idx = feature_end_idx - config.window + 1
        if corr_start_idx < 0 or feature_start_idx < 0:
            stats["skipped_lag_history"] += 1
            continue
        member_date = dates[member_end_idx]
        if not membership_map.get(member_date):
            stats["skipped_no_members"] += 1
            continue
        eligible_end_indices.append(end_idx)

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
            static_edge_map,
        )

    backend = (parallel_backend or "threadpool").lower()
    use_parallel = num_workers is not None and num_workers > 1 and backend not in ("none", "serial")

    if use_parallel and backend in ("joblib", "loky"):
        try:
            from joblib import Parallel, delayed  # type: ignore

            n_jobs = joblib_n_jobs if joblib_n_jobs is not None else num_workers
            if progress and joblib_prefer == "threads":
                pbar = tqdm(
                    total=len(eligible_end_indices),
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
                )(delayed(_task_pbar)(end_idx) for end_idx in eligible_end_indices)
                pbar.close()
            else:
                results = Parallel(
                    n_jobs=n_jobs,
                    prefer=joblib_prefer,
                    batch_size="auto",
                )(delayed(_task)(end_idx) for end_idx in eligible_end_indices)
        except Exception as exc:
            print(f"joblib parallel failed ({exc}); falling back to ThreadPoolExecutor")
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                it = executor.map(_task, eligible_end_indices)
                if progress:
                    it = tqdm(
                        it,
                        total=len(eligible_end_indices),
                        desc="Building graphs",
                        unit="win",
                        dynamic_ncols=True,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    )
                results = list(it)
    elif use_parallel:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            it = executor.map(_task, eligible_end_indices)
            if progress:
                it = tqdm(
                    it,
                    total=len(eligible_end_indices),
                    desc="Building graphs",
                    unit="win",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )
            results = list(it)
    else:
        it = eligible_end_indices
        if progress:
            it = tqdm(
                it,
                total=len(eligible_end_indices),
                desc="Building graphs",
                unit="win",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        results = [_task(end_idx) for end_idx in it]

    stats["total_windows"] = len(raw_end_indices)
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
        end_date, tickers, src, dst, w, rel_mask, rel_lag, x, ret_mean, ret_std = result
        edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        edge_attr = torch.from_numpy(w).float().unsqueeze(-1)
        edge_weight = edge_attr.squeeze(-1)
        edge_relation_mask = torch.from_numpy(np.array(rel_mask, copy=True)).long()
        edge_lag_days = torch.from_numpy(np.array(rel_lag, copy=True)).float()
        edge_type_primary = torch.from_numpy(_edge_primary_type_from_mask(rel_mask)).long()
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

        # Ensure writable backing memory before converting to torch tensor.
        x_tensor = torch.from_numpy(np.array(x, copy=True)).float()
        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(tickers))
        data.edge_weight = edge_weight
        data.edge_relation_mask = edge_relation_mask
        data.edge_lag_days = edge_lag_days
        data.edge_type = edge_type_primary
        if ret_mean is not None and ret_std is not None:
            data.ret_mean = torch.tensor(ret_mean.squeeze(1), dtype=torch.float32)
            data.ret_std = torch.tensor(ret_std.squeeze(1), dtype=torch.float32)

        graphs.append(data)
        graph_dates.append(end_date)
        node_tickers.append(tickers)
        stats["built"] += 1

    return graphs, graph_dates, node_tickers, stats
