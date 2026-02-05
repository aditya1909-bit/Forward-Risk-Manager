from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data


@dataclass
class GraphBuildConfig:
    window: int = 20
    step: int = 1
    top_k: int | None = 10
    corr_threshold: float | None = None
    min_nodes: int = 50
    feature_mode: str = "window"  # "window" or "last"
    normalize: bool = True
    symmetric: bool = True


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


def _build_node_features(window_df: pd.DataFrame, feature_mode: str, normalize: bool) -> np.ndarray:
    if feature_mode == "window":
        values = window_df.to_numpy().T  # shape: (nodes, window)
    elif feature_mode == "last":
        values = window_df.iloc[-1].to_numpy()[:, None]
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    if normalize:
        mean = np.nanmean(values, axis=1, keepdims=True)
        std = np.nanstd(values, axis=1, keepdims=True) + 1e-8
        values = (values - mean) / std
    return values


def _window_to_graph_data(
    end_idx: int,
    dates: List[str],
    returns: pd.DataFrame,
    membership_map: Dict[str, List[str]],
    config: GraphBuildConfig,
):
    end_date = dates[end_idx]
    members = membership_map.get(end_date)
    if not members:
        return None

    window_df = returns.iloc[end_idx - config.window + 1 : end_idx + 1]
    cols = [t for t in members if t in window_df.columns]
    if not cols:
        return None
    window_df = window_df[cols].dropna(axis=1, how="any")
    if window_df.shape[1] < config.min_nodes:
        return None

    corr = window_df.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    src, dst, w = _select_edges(corr, config.top_k, config.corr_threshold, config.symmetric)
    if len(src) == 0:
        return None

    x = _build_node_features(window_df, config.feature_mode, config.normalize)
    return end_date, list(window_df.columns), src, dst, w, x


def build_rolling_corr_graphs(
    returns: pd.DataFrame,
    membership_map: Dict[str, List[str]],
    config: GraphBuildConfig,
    num_workers: int = 1,
) -> Tuple[List[Data], List[str], List[List[str]]]:
    dates = list(returns.index)
    graphs: List[Data] = []
    graph_dates: List[str] = []
    node_tickers: List[List[str]] = []

    end_indices = list(range(config.window - 1, len(dates), config.step))

    if num_workers and num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        def _task(end_idx: int):
            return _window_to_graph_data(end_idx, dates, returns, membership_map, config)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_task, end_indices))
    else:
        results = [
            _window_to_graph_data(end_idx, dates, returns, membership_map, config)
            for end_idx in end_indices
        ]

    for result in results:
        if result is None:
            continue
        end_date, tickers, src, dst, w, x = result
        edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        edge_attr = torch.from_numpy(w).float().unsqueeze(-1)
        x_tensor = torch.from_numpy(x).float()
        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(tickers))

        graphs.append(data)
        graph_dates.append(end_date)
        node_tickers.append(tickers)

    return graphs, graph_dates, node_tickers
