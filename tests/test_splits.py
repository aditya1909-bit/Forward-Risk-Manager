from frisk.splits import (
    is_walk_forward_mode,
    simple_split_indices,
    simple_train_eval_split,
    walk_forward_splits,
)


def test_simple_split_chronological_order():
    items = list(range(10))
    train, evals = simple_train_eval_split(items, eval_frac=0.3, split_mode="chronological")
    assert train == list(range(7))
    assert evals == list(range(7, 10))


def test_simple_split_random_keeps_partition():
    items = list(range(20))
    train, evals = simple_train_eval_split(items, eval_frac=0.25, split_mode="random", seed=3)
    assert len(train) == 15
    assert len(evals) == 5
    assert sorted(train + evals) == items


def test_simple_split_indices_chronological():
    train_idx, eval_idx = simple_split_indices(10, eval_frac=0.3, split_mode="chronological")
    assert train_idx == list(range(7))
    assert eval_idx == [7, 8, 9]


def test_walk_forward_splits_expanding_windows():
    items = list(range(30))
    folds = walk_forward_splits(
        items,
        train_frac=0.5,
        eval_frac=0.2,
        step_frac=0.1,
        min_train_size=5,
        min_eval_size=3,
    )
    assert len(folds) == 4
    assert folds[0]["train_items"] == list(range(15))
    assert folds[0]["eval_items"] == list(range(15, 21))
    assert folds[1]["train_items"] == list(range(18))
    assert folds[1]["eval_items"] == list(range(18, 24))
    assert folds[0]["train_end"] < folds[1]["train_end"]


def test_is_walk_forward_mode_aliases():
    assert is_walk_forward_mode("walk_forward")
    assert is_walk_forward_mode("walk-forward")
    assert is_walk_forward_mode("expanding")
    assert not is_walk_forward_mode("chronological")
