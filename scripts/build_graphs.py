#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.data import load_prices, compute_log_returns, load_constituents, build_membership_map
from frisk.graph_builder import GraphBuildConfig, build_rolling_corr_graphs


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _get_setting(args: argparse.Namespace, section: dict, key: str, default):
    if hasattr(args, key):
        return getattr(args, key)
    if key in section:
        return section[key]
    return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Build rolling correlation graphs from tidy CSVs.")
    parser.add_argument("--config", help="Path to TOML config")
    parser.add_argument("--prices", help="Path to data/processed/prices.csv", default=argparse.SUPPRESS)
    parser.add_argument(
        "--constituents", help="Path to data/processed/constituents.csv", default=argparse.SUPPRESS
    )
    parser.add_argument("--out", help="Output .pt file", default=argparse.SUPPRESS)
    parser.add_argument("--window", type=int, help="Rolling window size in days", default=argparse.SUPPRESS)
    parser.add_argument("--step", type=int, help="Step size between windows", default=argparse.SUPPRESS)
    parser.add_argument("--top-k", type=int, help="Top-k correlations per node", default=argparse.SUPPRESS)
    parser.add_argument(
        "--corr-threshold", type=float, help="Correlation threshold", default=argparse.SUPPRESS
    )
    parser.add_argument("--min-nodes", type=int, help="Minimum nodes per graph", default=argparse.SUPPRESS)
    parser.add_argument("--feature-mode", choices=["window", "last"], default=argparse.SUPPRESS)
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable per-node z-score normalization"
    )
    parser.add_argument("--no-symmetric", action="store_true", help="Disable symmetric edge mirroring")
    parser.add_argument(
        "--include-tickers",
        help="Comma-separated tickers to force-include in every graph (e.g., MDY)",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker threads for building windows (use >1 for parallelism)",
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    section = cfg.get("build_graphs", {})

    prices_path = _get_setting(args, section, "prices", None)
    constituents_path = _get_setting(args, section, "constituents", None)
    out_path = _get_setting(args, section, "out", "data/processed/graphs.pt")

    if not prices_path or not constituents_path:
        raise ValueError("Provide --prices and --constituents (or set them in config).")

    window = _get_setting(args, section, "window", 20)
    step = _get_setting(args, section, "step", 1)
    top_k = _get_setting(args, section, "top_k", 10)
    corr_threshold = _get_setting(args, section, "corr_threshold", None)
    min_nodes = _get_setting(args, section, "min_nodes", 50)
    feature_mode = _get_setting(args, section, "feature_mode", "window")
    normalize = _get_setting(args, section, "normalize", True)
    symmetric = _get_setting(args, section, "symmetric", True)
    include_tickers = _get_setting(args, section, "include_tickers", [])
    workers = _get_setting(args, section, "workers", 1)

    if isinstance(include_tickers, str):
        include_tickers = [t.strip() for t in include_tickers.split(",") if t.strip()]

    if corr_threshold in ("", "none", "null"):
        corr_threshold = None

    if getattr(args, "no_normalize", False):
        normalize = False
    if getattr(args, "no_symmetric", False):
        symmetric = False

    if corr_threshold is not None:
        top_k = None

    prices = load_prices(Path(prices_path))
    returns = compute_log_returns(prices)
    constituents = load_constituents(Path(constituents_path))
    membership_map = build_membership_map(constituents, extra_tickers=include_tickers)

    cfg = GraphBuildConfig(
        window=window,
        step=step,
        top_k=top_k,
        corr_threshold=corr_threshold,
        min_nodes=min_nodes,
        feature_mode=feature_mode,
        normalize=normalize,
        symmetric=symmetric,
    )

    graphs, dates, tickers = build_rolling_corr_graphs(returns, membership_map, cfg, num_workers=workers)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "graphs": graphs,
        "dates": dates,
        "tickers": tickers,
        "config": cfg.__dict__,
    }
    torch.save(payload, out_path)
    print(f"Wrote {out_path} with {len(graphs)} graphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
