#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.graph_artifact import load_graph_artifact, save_graph_artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a packed graph artifact into a sharded directory for lazy loading."
    )
    parser.add_argument("--graphs", required=True, help="Packed graph artifact path (.pt).")
    parser.add_argument(
        "--out",
        default="",
        help="Output sharded artifact directory. Defaults to <graphs>.sharded",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help="Graphs per shard.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=0,
        help="Optional torch CPU thread count for deserialization/serialization. 0 keeps current setting.",
    )
    args = parser.parse_args()

    graphs_path = Path(args.graphs)
    out_path = Path(args.out) if str(args.out).strip() else Path(str(graphs_path) + ".sharded")
    if int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    print(f"torch_num_threads={torch.get_num_threads()}")
    if graphs_path.exists():
        print(f"source={graphs_path} size_gb={graphs_path.stat().st_size / (1024 ** 3):.2f}")
    print("Loading packed graph artifact into CPU memory...")
    t0 = time.perf_counter()
    try:
        artifact = load_graph_artifact(
            graphs_path,
            include_tickers=True,
            prefer_lazy=False,
            prefer_sharded=False,
        )
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "") or "unknown"
        raise RuntimeError(
            "Failed to load packed graph artifact because a required Python module "
            f"is missing during deserialization: {missing_name}. "
            "Install the repo graph dependencies first, especially torch-geometric."
        ) from exc
    load_s = time.perf_counter() - t0
    print(
        f"loaded format={artifact.format} graphs={len(artifact.graphs)} "
        f"in {load_s / 60.0:.2f} min"
    )
    print(
        f"Writing sharded artifact to {out_path} "
        f"(shard_size={max(1, int(args.shard_size))})..."
    )
    t1 = time.perf_counter()
    saved = save_graph_artifact(
        out_path,
        graphs=artifact.graphs,
        dates=artifact.dates,
        tickers=artifact.tickers or [],
        config=artifact.config,
        stats=artifact.stats,
        artifact_format="sharded",
        shard_size=max(1, int(args.shard_size)),
        progress=True,
    )
    write_s = time.perf_counter() - t1
    total_s = time.perf_counter() - t0
    print(f"Wrote sharded artifact: {saved}")
    print(
        f"timing: load={load_s / 60.0:.2f} min | "
        f"write={write_s / 60.0:.2f} min | total={total_s / 60.0:.2f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
