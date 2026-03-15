from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from frisk.graph_artifact import (
    GraphIndexSequence,
    load_graph_artifact,
    resolve_graph_artifact_path,
    save_graph_artifact,
)
from frisk.splits import walk_forward_split_indices


def _toy_graph(graph_idx: int) -> Data:
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=3)
    data.edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    data.graph_idx = graph_idx
    return data


def test_save_and_load_sharded_graph_artifact(tmp_path: Path):
    graphs = [_toy_graph(i) for i in range(5)]
    dates = [f"2024-01-0{i + 1}" for i in range(5)]
    tickers = [["AAA", "BBB", "CCC"] for _ in range(5)]
    out = tmp_path / "graphs.pt.sharded"

    save_graph_artifact(
        out,
        graphs=graphs,
        dates=dates,
        tickers=tickers,
        config={"window": 20},
        stats={"built": 5},
        artifact_format="sharded",
        shard_size=2,
    )

    artifact = load_graph_artifact(out, include_tickers=True, prefer_lazy=True, prefer_sharded=False)
    assert artifact.format == "sharded"
    assert len(artifact.graphs) == 5
    assert artifact.dates == dates
    assert artifact.stats["built"] == 5
    assert artifact.graphs[3].graph_idx == 3
    assert list(artifact.tickers[3]) == tickers[3]


def test_resolve_graph_artifact_prefers_sharded_sidecar(tmp_path: Path):
    packed = tmp_path / "graphs.pt"
    save_graph_artifact(
        packed,
        graphs=[_toy_graph(0)],
        dates=["2024-01-01"],
        tickers=[["AAA"]],
        config={},
        stats={},
        artifact_format="packed",
    )
    sharded = Path(str(packed) + ".sharded")
    save_graph_artifact(
        sharded,
        graphs=[_toy_graph(0)],
        dates=["2024-01-01"],
        tickers=[["AAA"]],
        config={},
        stats={},
        artifact_format="sharded",
        shard_size=1,
    )

    assert resolve_graph_artifact_path(packed) == sharded
    assert resolve_graph_artifact_path(packed, prefer_sharded=False) == packed


def test_graph_index_sequence_preserves_base_indices():
    base = [_toy_graph(i) for i in range(6)]
    subset = GraphIndexSequence(base, [1, 3, 5])
    assert len(subset) == 3
    assert subset[0].graph_idx == 1
    assert subset[2].graph_idx == 5


def test_walk_forward_split_indices_do_not_materialize_items():
    folds = walk_forward_split_indices(
        10,
        train_frac=0.5,
        eval_frac=0.2,
        step_frac=0.1,
        min_train_size=3,
        min_eval_size=2,
        max_folds=0,
    )
    assert folds
    assert "train_idx" in folds[0]
    assert "eval_idx" in folds[0]
    assert "train_items" not in folds[0]
    assert "eval_items" not in folds[0]
