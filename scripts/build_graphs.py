#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.data import (
    load_prices,
    compute_log_returns_and_volume,
    load_constituents,
    build_membership_map,
    build_membership_map_ffill,
    build_membership_map_all,
    load_fundamentals,
)
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
    parser.add_argument(
        "--fundamentals", help="Optional fundamentals CSV", default=argparse.SUPPRESS
    )
    parser.add_argument("--out", help="Output .pt file", default=argparse.SUPPRESS)
    parser.add_argument("--window", type=int, help="Rolling window size in days", default=argparse.SUPPRESS)
    parser.add_argument("--step", type=int, help="Step size between windows", default=argparse.SUPPRESS)
    parser.add_argument("--top-k", type=int, help="Top-k correlations per node", default=argparse.SUPPRESS)
    parser.add_argument(
        "--corr-threshold", type=float, help="Correlation threshold", default=argparse.SUPPRESS
    )
    parser.add_argument("--min-nodes", type=int, help="Minimum nodes per graph", default=argparse.SUPPRESS)
    parser.add_argument(
        "--membership-mode",
        choices=["constituents", "all"],
        default=argparse.SUPPRESS,
        help="Use constituents membership or include all tickers per date",
    )
    parser.add_argument(
        "--membership-fill",
        choices=["none", "ffill"],
        default=argparse.SUPPRESS,
        help="Fill missing membership dates (ffill) when using constituents",
    )
    parser.add_argument(
        "--membership-max-gap-days",
        type=int,
        default=argparse.SUPPRESS,
        help="Max gap in days allowed for membership forward-fill",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["window", "last", "window_plus_summary", "window_plus_summary_fund"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--rsi-period", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mdy-ticker", default=argparse.SUPPRESS)
    parser.add_argument("--edge-norm", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--edge-weight-mode", choices=["abs", "raw", "ones"], default=argparse.SUPPRESS)
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
        "--start-date",
        default=argparse.SUPPRESS,
        help="Filter start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=argparse.SUPPRESS,
        help="Filter end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker threads for building windows (use >1 for parallelism)",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["threadpool", "joblib", "serial"],
        help="Parallel backend: threadpool (default), joblib, or serial",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--joblib-prefer",
        choices=["threads", "processes"],
        help="joblib backend preference",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--joblib-n-jobs",
        type=int,
        help="joblib n_jobs (defaults to workers)",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    section = cfg.get("build_graphs", {})

    prices_path = _get_setting(args, section, "prices", None)
    constituents_path = _get_setting(args, section, "constituents", None)
    fundamentals_path = _get_setting(args, section, "fundamentals", None)
    out_path = _get_setting(args, section, "out", "data/processed/graphs.pt")

    if not prices_path:
        raise ValueError("Provide --prices (or set it in config).")

    window = _get_setting(args, section, "window", 20)
    step = _get_setting(args, section, "step", 1)
    top_k = _get_setting(args, section, "top_k", 10)
    corr_threshold = _get_setting(args, section, "corr_threshold", None)
    min_nodes = _get_setting(args, section, "min_nodes", 50)
    feature_mode = _get_setting(args, section, "feature_mode", "window")
    rsi_period = _get_setting(args, section, "rsi_period", 14)
    mdy_ticker = _get_setting(args, section, "mdy_ticker", "MDY")
    edge_norm = _get_setting(args, section, "edge_norm", True)
    edge_weight_mode = _get_setting(args, section, "edge_weight_mode", "abs")
    normalize = _get_setting(args, section, "normalize", True)
    symmetric = _get_setting(args, section, "symmetric", True)
    membership_mode = _get_setting(args, section, "membership_mode", "constituents")
    membership_fill = _get_setting(args, section, "membership_fill", "none")
    membership_max_gap_days = _get_setting(args, section, "membership_max_gap_days", None)
    start_date = _get_setting(args, section, "start_date", "")
    end_date = _get_setting(args, section, "end_date", "")
    include_tickers = _get_setting(args, section, "include_tickers", [])
    workers = _get_setting(args, section, "workers", 1)
    parallel_backend = _get_setting(args, section, "parallel_backend", "threadpool")
    joblib_prefer = _get_setting(args, section, "joblib_prefer", "threads")
    joblib_n_jobs = _get_setting(args, section, "joblib_n_jobs", None)
    progress = _get_setting(args, section, "progress", True)
    if isinstance(joblib_n_jobs, str) and joblib_n_jobs.lower() in ("", "none", "null"):
        joblib_n_jobs = None
    if isinstance(joblib_n_jobs, int) and joblib_n_jobs <= 0:
        joblib_n_jobs = None
    if getattr(args, "no_progress", False):
        progress = False

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
    returns, volume = compute_log_returns_and_volume(prices)
    if start_date:
        returns = returns[returns.index >= start_date]
        if volume is not None:
            volume = volume[volume.index >= start_date]
    if end_date:
        returns = returns[returns.index <= end_date]
        if volume is not None:
            volume = volume[volume.index <= end_date]

    if membership_mode == "all":
        membership_map = build_membership_map_all(returns, extra_tickers=include_tickers)
    else:
        if not constituents_path:
            raise ValueError("Provide --constituents (or set it in config).")
        constituents = load_constituents(Path(constituents_path))
        if start_date:
            constituents = constituents[constituents["date"] >= start_date]
        if end_date:
            constituents = constituents[constituents["date"] <= end_date]
        if membership_fill == "ffill":
            membership_map, fill_stats = build_membership_map_ffill(
                constituents,
                list(returns.index),
                extra_tickers=include_tickers,
                max_gap_days=membership_max_gap_days,
            )
            print(
                "Membership ffill:",
                f"source_dates={fill_stats['source_dates']}",
                f"filled_dates={fill_stats['filled_dates']}",
                f"gap_dropped={fill_stats['gap_dropped']}",
                f"max_gap_days={membership_max_gap_days}",
            )
        else:
            membership_map = build_membership_map(constituents, extra_tickers=include_tickers)

    fundamentals = None
    if fundamentals_path:
        fundamentals = load_fundamentals(Path(fundamentals_path))
        if start_date:
            fundamentals = fundamentals[fundamentals["date"] >= start_date]
        if end_date:
            fundamentals = fundamentals[fundamentals["date"] <= end_date]

    cfg = GraphBuildConfig(
        window=window,
        step=step,
        top_k=top_k,
        corr_threshold=corr_threshold,
        min_nodes=min_nodes,
        feature_mode=feature_mode,
        normalize=normalize,
        symmetric=symmetric,
        rsi_period=rsi_period,
        mdy_ticker=mdy_ticker,
        edge_norm=edge_norm,
        edge_weight_mode=edge_weight_mode,
    )

    graphs, dates, tickers, stats = build_rolling_corr_graphs(
        returns,
        volume,
        membership_map,
        cfg,
        fundamentals=fundamentals,
        num_workers=workers,
        parallel_backend=parallel_backend,
        joblib_prefer=joblib_prefer,
        joblib_n_jobs=joblib_n_jobs,
        progress=progress,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "graphs": graphs,
        "dates": dates,
        "tickers": tickers,
        "config": cfg.__dict__,
        "stats": stats,
    }
    torch.save(payload, out_path)
    if dates:
        start_date = dates[0]
        end_date = dates[-1]
    else:
        start_date = "n/a"
        end_date = "n/a"
    print(f"Wrote {out_path} with {len(graphs)} graphs")
    print(
        f"Date range: {start_date} -> {end_date} | "
        f"windows: {stats.get('total_windows', 0)} | "
        f"built: {stats.get('built', 0)} | "
        f"skipped: members={stats.get('skipped_no_members', 0)}, "
        f"cols={stats.get('skipped_no_cols', 0)}, "
        f"min_nodes={stats.get('skipped_min_nodes', 0)}, "
        f"no_edges={stats.get('skipped_no_edges', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
