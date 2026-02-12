from __future__ import annotations

import math
import numpy as np


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    i = 0
    while i < order.size:
        j = i
        vi = values[order[i]]
        while j + 1 < order.size and values[order[j + 1]] == vi:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    y = labels.astype(np.int32, copy=False)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average_ties(scores.astype(float, copy=False))
    sum_pos = float(ranks[y == 1].sum())
    u = sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def binary_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    y = labels.astype(np.int32, copy=False)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    ap = 0.0
    prev_recall = 0.0
    for i in range(y_sorted.size):
        if y_sorted[i] != 1:
            continue
        recall = float(tp[i] / n_pos)
        precision = float(tp[i] / (i + 1))
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


def binary_brier_and_ece(
    scores: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> tuple[float, float]:
    y = labels.astype(float, copy=False)
    s = np.asarray(scores, dtype=float)
    s = np.clip(s, -50.0, 50.0)
    probs = 1.0 / (1.0 + np.exp(-s))
    brier = float(np.mean((probs - y) ** 2))

    nb = max(2, int(bins))
    edges = np.linspace(0.0, 1.0, nb + 1)
    ece = 0.0
    n = probs.size
    if n == 0:
        return brier, float("nan")
    for bi in range(nb):
        lo = edges[bi]
        hi = edges[bi + 1]
        if bi == nb - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(y[mask].mean())
        ece += (cnt / n) * abs(conf - acc)
    return brier, float(ece)


def ff_binary_metrics(
    g_pos: np.ndarray,
    g_neg: np.ndarray,
    threshold: float,
    ece_bins: int = 10,
) -> dict[str, float]:
    if g_pos.size == 0 or g_neg.size == 0:
        return {
            "eval_auroc": float("nan"),
            "eval_auprc": float("nan"),
            "eval_brier": float("nan"),
            "eval_ece": float("nan"),
        }
    labels = np.concatenate([np.ones_like(g_pos), np.zeros_like(g_neg)]).astype(int)
    scores = np.concatenate([g_pos, g_neg]) - float(threshold)
    auroc = binary_auroc(scores, labels)
    auprc = binary_auprc(scores, labels)
    brier, ece = binary_brier_and_ece(scores, labels, bins=ece_bins)
    if not math.isfinite(auroc):
        auroc = float("nan")
    if not math.isfinite(auprc):
        auprc = float("nan")
    return {
        "eval_auroc": float(auroc),
        "eval_auprc": float(auprc),
        "eval_brier": float(brier),
        "eval_ece": float(ece),
    }
