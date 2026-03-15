from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def is_walk_forward_mode(split_mode: str) -> bool:
    mode = str(split_mode).strip().lower().replace("-", "_")
    return mode in {
        "walk_forward",
        "walkforward",
        "expanding",
        "expanding_window",
        "expandingwindow",
    }


def simple_train_eval_split(
    items: Sequence[T],
    eval_frac: float = 0.2,
    seed: int = 7,
    split_mode: str = "chronological",
) -> tuple[list[T], list[T]]:
    train_idx, eval_idx = simple_split_indices(
        len(items),
        eval_frac=eval_frac,
        seed=seed,
        split_mode=split_mode,
    )
    train = [items[i] for i in train_idx]
    evals = [items[i] for i in eval_idx]
    return train, evals


def simple_split_indices(
    n_items: int,
    eval_frac: float = 0.2,
    seed: int = 7,
    split_mode: str = "chronological",
) -> tuple[list[int], list[int]]:
    n = int(n_items)
    if n < 2:
        raise ValueError("Need at least 2 items to create train/eval splits.")
    cut = int(n * (1 - float(eval_frac)))
    cut = max(1, min(n - 1, cut))

    mode = str(split_mode).strip().lower()
    if mode in ("chronological", "chrono", "time"):
        return list(range(cut)), list(range(cut, n))
    if mode in ("random", "shuffle"):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        return idx[:cut], idx[cut:]
    raise ValueError(
        f"Unknown split_mode: {split_mode}. Expected chronological or random."
    )


def walk_forward_splits(
    items: Sequence[T],
    train_frac: float = 0.6,
    eval_frac: float = 0.2,
    step_frac: float | None = None,
    min_train_size: int = 64,
    min_eval_size: int = 16,
    max_folds: int = 0,
) -> list[dict[str, object]]:
    n = len(items)
    if n < 2:
        raise ValueError("Need at least 2 items for walk-forward splits.")

    train_size = max(int(round(n * float(train_frac))), int(min_train_size))
    train_size = min(max(1, train_size), n - 1)

    eval_size = max(int(round(n * float(eval_frac))), int(min_eval_size))
    eval_size = min(max(1, eval_size), n - train_size)
    if eval_size <= 0:
        raise ValueError("walk-forward eval_size resolved to zero.")

    if step_frac is None or float(step_frac) <= 0:
        step_size = eval_size
    else:
        step_size = max(1, int(round(n * float(step_frac))))

    splits: list[dict[str, object]] = []
    eval_start = train_size
    while eval_start + eval_size <= n:
        eval_end = eval_start + eval_size
        train_items = list(items[:eval_start])
        eval_items = list(items[eval_start:eval_end])
        if train_items and eval_items:
            splits.append(
                {
                    "fold_id": len(splits),
                    "train_start": 0,
                    "train_end": eval_start,
                    "eval_start": eval_start,
                    "eval_end": eval_end,
                    "train_items": train_items,
                    "eval_items": eval_items,
                }
            )
            if max_folds > 0 and len(splits) >= int(max_folds):
                break
        eval_start += step_size

    if not splits:
        raise ValueError(
            "No valid walk-forward folds were generated. Adjust train/eval fractions or min sizes."
        )
    return splits


def walk_forward_split_indices(
    n_items: int,
    train_frac: float = 0.6,
    eval_frac: float = 0.2,
    step_frac: float | None = None,
    min_train_size: int = 64,
    min_eval_size: int = 16,
    max_folds: int = 0,
) -> list[dict[str, object]]:
    n = int(n_items)
    if n < 2:
        raise ValueError("Need at least 2 items for walk-forward splits.")

    train_size = max(int(round(n * float(train_frac))), int(min_train_size))
    train_size = min(max(1, train_size), n - 1)

    eval_size = max(int(round(n * float(eval_frac))), int(min_eval_size))
    eval_size = min(max(1, eval_size), n - train_size)
    if eval_size <= 0:
        raise ValueError("walk-forward eval_size resolved to zero.")

    if step_frac is None or float(step_frac) <= 0:
        step_size = eval_size
    else:
        step_size = max(1, int(round(n * float(step_frac))))

    splits: list[dict[str, object]] = []
    eval_start = train_size
    while eval_start + eval_size <= n:
        eval_end = eval_start + eval_size
        train_idx = list(range(0, eval_start))
        eval_idx = list(range(eval_start, eval_end))
        if train_idx and eval_idx:
            splits.append(
                {
                    "fold_id": len(splits),
                    "train_start": 0,
                    "train_end": eval_start,
                    "eval_start": eval_start,
                    "eval_end": eval_end,
                    "train_idx": train_idx,
                    "eval_idx": eval_idx,
                }
            )
            if max_folds > 0 and len(splits) >= int(max_folds):
                break
        eval_start += step_size

    if not splits:
        raise ValueError(
            "No valid walk-forward folds were generated. Adjust train/eval fractions or min sizes."
        )
    return splits
