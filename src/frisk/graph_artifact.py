from __future__ import annotations

import bisect
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


_MANIFEST_NAME = "manifest.json"
_DEFAULT_SHARDED_SUFFIX = ".sharded"


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _ensure_graph_idx(graph, graph_idx: int):
    if getattr(graph, "graph_idx", None) != int(graph_idx):
        setattr(graph, "graph_idx", int(graph_idx))
    return graph


class GraphSequence(Sequence):
    def __iter__(self) -> Iterator:
        for idx in range(len(self)):
            yield self[idx]


class InMemoryGraphSequence(GraphSequence):
    def __init__(self, graphs: Sequence):
        self._graphs = list(graphs)

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        graph_idx = int(index)
        return _ensure_graph_idx(self._graphs[graph_idx], graph_idx)


class GraphIndexSequence(GraphSequence):
    def __init__(self, graphs: Sequence, indices: Sequence[int]):
        self._graphs = graphs
        self._indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        base_index = self._indices[int(index)]
        return self._graphs[base_index]

    @property
    def indices(self) -> list[int]:
        return list(self._indices)


class ShardedMetaSequence(Sequence):
    def __init__(self, root: Path, manifest: dict[str, Any], key: str):
        self._root = root
        self._key = key
        self._ranges = [int(row["end"]) for row in manifest.get("shards", [])]
        self._rows = list(manifest.get("shards", []))
        self._cache_idx: int | None = None
        self._cache_values: list[Any] | None = None

    def __len__(self) -> int:
        if not self._rows:
            return 0
        return int(self._rows[-1]["end"])

    def _load_shard_values(self, shard_idx: int) -> list[Any]:
        if self._cache_idx == shard_idx and self._cache_values is not None:
            return self._cache_values
        row = self._rows[shard_idx]
        payload = _torch_load(self._root / row["meta_relpath"])
        values = list(payload.get(self._key, []))
        self._cache_idx = shard_idx
        self._cache_values = values
        return values

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        idx = int(index)
        shard_idx = bisect.bisect_right(self._ranges, idx)
        if shard_idx >= len(self._rows):
            raise IndexError(idx)
        row = self._rows[shard_idx]
        local_idx = idx - int(row["start"])
        values = self._load_shard_values(shard_idx)
        return values[local_idx]


class ShardedGraphSequence(GraphSequence):
    def __init__(self, root: Path, manifest: dict[str, Any]):
        self._root = root
        self._rows = list(manifest.get("shards", []))
        self._ranges = [int(row["end"]) for row in self._rows]
        self._cache_idx: int | None = None
        self._cache_graphs: list[Any] | None = None

    def __len__(self) -> int:
        if not self._rows:
            return 0
        return int(self._rows[-1]["end"])

    def _load_shard_graphs(self, shard_idx: int) -> list[Any]:
        if self._cache_idx == shard_idx and self._cache_graphs is not None:
            return self._cache_graphs
        row = self._rows[shard_idx]
        payload = _torch_load(self._root / row["graphs_relpath"])
        graphs = list(payload.get("graphs", payload))
        self._cache_idx = shard_idx
        self._cache_graphs = graphs
        return graphs

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        idx = int(index)
        shard_idx = bisect.bisect_right(self._ranges, idx)
        if shard_idx >= len(self._rows):
            raise IndexError(idx)
        row = self._rows[shard_idx]
        local_idx = idx - int(row["start"])
        graphs = self._load_shard_graphs(shard_idx)
        return _ensure_graph_idx(graphs[local_idx], idx)


@dataclass(frozen=True)
class GraphArtifact:
    path: Path
    format: str
    graphs: Sequence
    dates: list[str]
    tickers: Sequence | None
    config: dict[str, Any]
    stats: dict[str, Any]


def resolve_graph_artifact_path(path: str | Path, *, prefer_sharded: bool = True) -> Path:
    base = Path(path)
    if base.is_dir():
        return base
    sharded = Path(str(base) + _DEFAULT_SHARDED_SUFFIX)
    if prefer_sharded and sharded.is_dir() and (sharded / _MANIFEST_NAME).exists():
        return sharded
    return base


def load_graph_artifact(
    path: str | Path,
    *,
    include_tickers: bool = True,
    prefer_lazy: bool = True,
    prefer_sharded: bool = True,
) -> GraphArtifact:
    resolved = resolve_graph_artifact_path(path, prefer_sharded=prefer_sharded)
    if resolved.is_dir():
        manifest_path = resolved / _MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        graphs: Sequence
        if prefer_lazy:
            graphs = ShardedGraphSequence(resolved, manifest)
        else:
            all_graphs = []
            for row in manifest.get("shards", []):
                payload = _torch_load(resolved / row["graphs_relpath"])
                all_graphs.extend(list(payload.get("graphs", payload)))
            graphs = InMemoryGraphSequence(all_graphs)
        tickers = ShardedMetaSequence(resolved, manifest, "tickers") if include_tickers else None
        return GraphArtifact(
            path=resolved,
            format="sharded",
            graphs=graphs,
            dates=list(manifest.get("dates", [])),
            tickers=tickers,
            config=dict(manifest.get("config", {})),
            stats=dict(manifest.get("stats", {})),
        )

    payload = _torch_load(resolved)
    graphs_raw = payload["graphs"] if isinstance(payload, dict) and "graphs" in payload else payload
    graphs = graphs_raw if isinstance(graphs_raw, GraphSequence) else InMemoryGraphSequence(graphs_raw)
    return GraphArtifact(
        path=resolved,
        format="packed",
        graphs=graphs,
        dates=list(payload.get("dates", [])) if isinstance(payload, dict) else [],
        tickers=(payload.get("tickers", []) if isinstance(payload, dict) and include_tickers else None),
        config=dict(payload.get("config", {})) if isinstance(payload, dict) else {},
        stats=dict(payload.get("stats", {})) if isinstance(payload, dict) else {},
    )


def save_graph_artifact(
    path: str | Path,
    *,
    graphs: Sequence,
    dates: Sequence[str],
    tickers: Sequence[Sequence[str]],
    config: dict[str, Any],
    stats: dict[str, Any],
    artifact_format: str = "packed",
    shard_size: int = 256,
    progress: bool = False,
) -> Path:
    out_path = Path(path)
    fmt = str(artifact_format).strip().lower()
    if fmt not in {"packed", "sharded"}:
        raise ValueError(f"Unknown artifact_format: {artifact_format}")

    payload = {
        "graphs": list(graphs),
        "dates": list(dates),
        "tickers": list(tickers),
        "config": dict(config),
        "stats": dict(stats),
    }
    if fmt == "packed":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
        return out_path

    shard_n = max(1, int(shard_size))
    out_path.mkdir(parents=True, exist_ok=True)
    shard_rows: list[dict[str, Any]] = []
    graphs_dir = out_path / "shards"
    meta_dir = out_path / "meta"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    total = len(payload["graphs"])
    shard_starts = list(range(0, total, shard_n))
    shard_iter = shard_starts
    if progress:
        shard_iter = tqdm(
            shard_starts,
            total=len(shard_starts),
            desc="Writing shards",
            unit="shard",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    for shard_id, start in enumerate(shard_iter):
        end = min(total, start + shard_n)
        graphs_rel = Path("shards") / f"shard_{shard_id:05d}_graphs.pt"
        meta_rel = Path("meta") / f"shard_{shard_id:05d}_meta.pt"
        torch.save({"graphs": payload["graphs"][start:end]}, out_path / graphs_rel)
        torch.save(
            {
                "dates": payload["dates"][start:end],
                "tickers": payload["tickers"][start:end],
            },
            out_path / meta_rel,
        )
        shard_rows.append(
            {
                "shard_id": shard_id,
                "start": start,
                "end": end,
                "graphs_relpath": graphs_rel.as_posix(),
                "meta_relpath": meta_rel.as_posix(),
            }
        )

    manifest = {
        "version": 1,
        "format": "sharded",
        "num_graphs": total,
        "shard_size": shard_n,
        "num_shards": int(math.ceil(total / shard_n)) if shard_n > 0 else 0,
        "dates": payload["dates"],
        "config": payload["config"],
        "stats": payload["stats"],
        "shards": shard_rows,
    }
    (out_path / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path
