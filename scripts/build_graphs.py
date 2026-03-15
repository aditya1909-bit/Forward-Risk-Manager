#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
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
    load_sec_fundamentals,
    load_macro_features,
    load_static_edges,
    build_macro_features_from_market_data,
)
from frisk.graph_builder import GraphBuildConfig, build_rolling_corr_graphs
from frisk.graph_artifact import save_graph_artifact


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


def _optional_int(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


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
    parser.add_argument(
        "--sec-companyfacts",
        help="Optional SEC companyfacts CSV (consolidated_ff_local/sec_companyfacts_selected.csv).",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sec-submissions",
        help="Optional SEC submissions CSV (consolidated_ff_local/sec_submissions_entities.csv).",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sec-as-fundamentals",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Append SEC-derived fundamentals features from sec_companyfacts/submissions CSVs.",
    )
    parser.add_argument(
        "--no-sec-as-fundamentals",
        action="store_true",
        help="Disable SEC-derived fundamentals even if enabled in config.",
    )
    parser.add_argument(
        "--macro", help="Optional macro feature CSV (date + feature columns)", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--macro-auto",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Auto-generate macro features from the available price/volume panel when --macro is empty.",
    )
    parser.add_argument(
        "--macro-auto-short-window",
        type=int,
        default=argparse.SUPPRESS,
        help="Short rolling window used in auto-generated macro features.",
    )
    parser.add_argument(
        "--macro-auto-long-window",
        type=int,
        default=argparse.SUPPRESS,
        help="Long rolling window used in auto-generated macro features.",
    )
    parser.add_argument(
        "--static-edges",
        help="Optional static edge CSV (e.g., src,dst,weight,directed).",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--out", help="Output graph artifact path (.pt or sharded dir)", default=argparse.SUPPRESS)
    parser.add_argument(
        "--artifact-format",
        choices=["packed", "sharded"],
        default=argparse.SUPPRESS,
        help="Artifact format: packed .pt or sharded directory for lazy loading.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Graphs per shard when artifact-format=sharded.",
    )
    parser.add_argument("--window", type=int, help="Rolling window size in days", default=argparse.SUPPRESS)
    parser.add_argument("--step", type=int, help="Step size between windows", default=argparse.SUPPRESS)
    parser.add_argument(
        "--corr-lag-days",
        type=int,
        default=argparse.SUPPRESS,
        help="Lag (days) applied to correlation window end to prevent lookahead.",
    )
    parser.add_argument(
        "--feature-lag-days",
        type=int,
        default=argparse.SUPPRESS,
        help="Lag (days) applied to feature window end to prevent lookahead.",
    )
    parser.add_argument(
        "--membership-lag-days",
        type=int,
        default=argparse.SUPPRESS,
        help="Lag (days) applied to membership date lookup to prevent lookahead.",
    )
    parser.add_argument(
        "--macro-lag-days",
        type=int,
        default=argparse.SUPPRESS,
        help="Additional lag (days) applied to macro feature lookup to prevent lookahead.",
    )
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
        "--corr-method",
        choices=["pearson", "partial"],
        default=argparse.SUPPRESS,
        help="Correlation estimator to use for graph edges.",
    )
    parser.add_argument(
        "--partial-corr-ridge",
        type=float,
        default=argparse.SUPPRESS,
        help="Ridge regularization strength for partial correlation.",
    )
    parser.add_argument(
        "--edge-select-mode",
        choices=["top_k", "threshold", "significance"],
        default=argparse.SUPPRESS,
        help="Edge selection method.",
    )
    parser.add_argument(
        "--significance-alpha",
        type=float,
        default=argparse.SUPPRESS,
        help="Two-sided alpha for significance-based edge selection.",
    )
    parser.add_argument(
        "--cross-sectional-norm",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Apply cross-sectional z-score normalization across nodes for each date slice.",
    )
    parser.add_argument(
        "--edge-node-weighting",
        choices=["none", "volume", "market_cap", "volume_market_cap"],
        default=argparse.SUPPRESS,
        help="Optional node-derived weighting applied to edge strengths.",
    )
    parser.add_argument(
        "--edge-node-weight-power",
        type=float,
        default=argparse.SUPPRESS,
        help="Power for node-derived edge scaling.",
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable per-node z-score normalization"
    )
    parser.add_argument(
        "--lead-lag-enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Add directed lead-lag edges estimated from lagged cross-correlations.",
    )
    parser.add_argument(
        "--lead-lag-max-lag",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum lag (in days) for lead-lag edge search.",
    )
    parser.add_argument(
        "--lead-lag-top-k",
        type=int,
        default=argparse.SUPPRESS,
        help="Top-k lead-lag edges per node when lead_lag_mode=top_k.",
    )
    parser.add_argument(
        "--lead-lag-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Absolute lagged-correlation threshold when lead_lag_mode=threshold.",
    )
    parser.add_argument(
        "--lead-lag-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Global weight multiplier for lead-lag edges.",
    )
    parser.add_argument(
        "--lead-lag-mode",
        choices=["top_k", "threshold", "significance"],
        default=argparse.SUPPRESS,
        help="Lead-lag edge selection method.",
    )
    parser.add_argument(
        "--sector-static-enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Add permanent same-sector edges from fundamentals when available.",
    )
    parser.add_argument(
        "--sector-static-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight for same-sector static edges.",
    )
    parser.add_argument(
        "--sector-static-top-k",
        type=int,
        default=argparse.SUPPRESS,
        help="Max same-sector peers per node for static sector edges.",
    )
    parser.add_argument(
        "--static-edge-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Global multiplier for user-provided static edges.",
    )
    parser.add_argument("--no-symmetric", action="store_true", help="Disable symmetric edge mirroring")
    parser.add_argument(
        "--include-tickers",
        help="Comma-separated tickers to force-include in every graph",
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
    parser.add_argument(
        "--progress",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable progress bar output (overrides config).",
    )
    parser.add_argument(
        "--volume-complete-required",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Require complete volume history per window/ticker (stricter; may drop more windows).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    section = cfg.get("build_graphs", {})

    prices_path = _get_setting(args, section, "prices", None)
    constituents_path = _get_setting(args, section, "constituents", None)
    fundamentals_path = _get_setting(args, section, "fundamentals", None)
    sec_companyfacts_path = _get_setting(args, section, "sec_companyfacts", None)
    sec_submissions_path = _get_setting(args, section, "sec_submissions", None)
    sec_as_fundamentals = bool(_get_setting(args, section, "sec_as_fundamentals", True))
    if getattr(args, "no_sec_as_fundamentals", False):
        sec_as_fundamentals = False
    macro_path = _get_setting(args, section, "macro", None)
    macro_auto = bool(_get_setting(args, section, "macro_auto", False))
    macro_auto_short_window = int(_get_setting(args, section, "macro_auto_short_window", 21))
    macro_auto_long_window = int(_get_setting(args, section, "macro_auto_long_window", 63))
    static_edges_path = _get_setting(args, section, "static_edges", None)
    out_path = _get_setting(args, section, "out", "data/processed/graphs.pt")
    artifact_format = str(_get_setting(args, section, "artifact_format", "packed"))
    shard_size = int(_get_setting(args, section, "shard_size", 256))

    if not prices_path:
        raise ValueError("Provide --prices (or set it in config).")

    window = _get_setting(args, section, "window", 20)
    step = _get_setting(args, section, "step", 1)
    corr_lag_days = _get_setting(args, section, "corr_lag_days", 0)
    feature_lag_days = _get_setting(args, section, "feature_lag_days", 0)
    membership_lag_days = _get_setting(args, section, "membership_lag_days", 0)
    macro_lag_days = _get_setting(args, section, "macro_lag_days", 0)
    top_k = _get_setting(args, section, "top_k", 10)
    corr_threshold = _get_setting(args, section, "corr_threshold", None)
    min_nodes = _get_setting(args, section, "min_nodes", 50)
    feature_mode = _get_setting(args, section, "feature_mode", "window")
    rsi_period = _get_setting(args, section, "rsi_period", 14)
    mdy_ticker = _get_setting(args, section, "mdy_ticker", "AUTO")
    edge_norm = _get_setting(args, section, "edge_norm", True)
    edge_weight_mode = _get_setting(args, section, "edge_weight_mode", "raw")
    corr_method = _get_setting(args, section, "corr_method", "pearson")
    partial_corr_ridge = _get_setting(args, section, "partial_corr_ridge", 1e-3)
    edge_select_mode = _get_setting(args, section, "edge_select_mode", "top_k")
    significance_alpha = _get_setting(args, section, "significance_alpha", 0.05)
    cross_sectional_norm = _get_setting(args, section, "cross_sectional_norm", False)
    edge_node_weighting = _get_setting(args, section, "edge_node_weighting", "none")
    edge_node_weight_power = _get_setting(args, section, "edge_node_weight_power", 0.5)
    lead_lag_enabled = bool(_get_setting(args, section, "lead_lag_enabled", False))
    lead_lag_max_lag = int(_get_setting(args, section, "lead_lag_max_lag", 3))
    lead_lag_top_k = _get_setting(args, section, "lead_lag_top_k", 2)
    lead_lag_threshold = _get_setting(args, section, "lead_lag_threshold", None)
    lead_lag_weight = float(_get_setting(args, section, "lead_lag_weight", 0.5))
    lead_lag_mode = str(_get_setting(args, section, "lead_lag_mode", "top_k"))
    sector_static_enabled = bool(_get_setting(args, section, "sector_static_enabled", False))
    sector_static_weight = float(_get_setting(args, section, "sector_static_weight", 0.25))
    sector_static_top_k = _get_setting(args, section, "sector_static_top_k", 4)
    static_edge_weight = float(_get_setting(args, section, "static_edge_weight", 0.25))
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
    volume_complete_required = bool(_get_setting(args, section, "volume_complete_required", False))
    if isinstance(joblib_n_jobs, str) and joblib_n_jobs.lower() in ("", "none", "null"):
        joblib_n_jobs = None
    if isinstance(joblib_n_jobs, int) and joblib_n_jobs <= 0:
        joblib_n_jobs = None
    if getattr(args, "no_progress", False):
        progress = False
    elif getattr(args, "progress", False):
        progress = True

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

    prices_parent = Path(prices_path).expanduser().resolve().parent
    if not sec_companyfacts_path:
        auto_sec_companyfacts = prices_parent / "sec_companyfacts_selected.csv"
        if auto_sec_companyfacts.exists():
            sec_companyfacts_path = str(auto_sec_companyfacts)
    if not sec_submissions_path:
        auto_sec_submissions = prices_parent / "sec_submissions_entities.csv"
        if auto_sec_submissions.exists():
            sec_submissions_path = str(auto_sec_submissions)

    if sec_as_fundamentals and sec_companyfacts_path:
        sec_fund = load_sec_fundamentals(
            Path(sec_companyfacts_path),
            submissions_path=Path(sec_submissions_path) if sec_submissions_path else None,
        )
        if start_date:
            sec_fund = sec_fund[sec_fund["date"] >= start_date]
        if end_date:
            sec_fund = sec_fund[sec_fund["date"] <= end_date]
        fundamentals = (
            sec_fund
            if fundamentals is None or fundamentals.empty
            else pd.concat([fundamentals, sec_fund], ignore_index=True, sort=False)
        )
        if fundamentals is not None and not fundamentals.empty:
            fundamentals = (
                fundamentals.sort_values(["date", "ticker"])
                .groupby(["date", "ticker"], as_index=False)
                .last()
            )
        print(
            "SEC fundamentals:",
            f"companyfacts={sec_companyfacts_path}",
            f"submissions={sec_submissions_path or 'none'}",
            f"rows={0 if fundamentals is None else len(fundamentals)}",
        )

    macro = None
    macro_source = "none"
    if macro_path:
        try:
            macro = load_macro_features(Path(macro_path))
            macro_source = "file"
        except Exception as exc:
            if not macro_auto:
                raise
            print(f"warning: failed to load macro CSV ({exc}); using auto-generated macro features.")
            macro = build_macro_features_from_market_data(
                returns=returns,
                volume=volume,
                short_window=macro_auto_short_window,
                long_window=macro_auto_long_window,
            )
            macro_source = "auto_fallback"
    elif macro_auto:
        macro = build_macro_features_from_market_data(
            returns=returns,
            volume=volume,
            short_window=macro_auto_short_window,
            long_window=macro_auto_long_window,
        )
        macro_source = "auto"

    if macro is not None and not macro.empty:
        if start_date:
            macro = macro[macro.index >= start_date]
        if end_date:
            macro = macro[macro.index <= end_date]
        if macro_source != "none":
            print(f"Macro features: source={macro_source} cols={macro.shape[1]} rows={macro.shape[0]}")

    static_edges = None
    if static_edges_path:
        static_edges = load_static_edges(Path(static_edges_path))
        print(f"Static edges loaded: rows={len(static_edges)} path={static_edges_path}")

    cfg = GraphBuildConfig(
        window=window,
        step=step,
        corr_lag_days=max(0, int(corr_lag_days)),
        feature_lag_days=max(0, int(feature_lag_days)),
        membership_lag_days=max(0, int(membership_lag_days)),
        macro_lag_days=max(0, int(macro_lag_days)),
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
        cross_sectional_norm=bool(cross_sectional_norm),
        corr_method=str(corr_method),
        partial_corr_ridge=float(partial_corr_ridge),
        edge_select_mode=str(edge_select_mode),
        significance_alpha=float(significance_alpha),
        edge_node_weighting=str(edge_node_weighting),
        edge_node_weight_power=float(edge_node_weight_power),
        lead_lag_enabled=bool(lead_lag_enabled),
        lead_lag_max_lag=max(1, int(lead_lag_max_lag)),
        lead_lag_top_k=_optional_int(lead_lag_top_k),
        lead_lag_threshold=_optional_float(lead_lag_threshold),
        lead_lag_weight=float(lead_lag_weight),
        lead_lag_mode=str(lead_lag_mode),
        sector_static_enabled=bool(sector_static_enabled),
        sector_static_weight=float(sector_static_weight),
        sector_static_top_k=_optional_int(sector_static_top_k),
        static_edge_weight=float(static_edge_weight),
        volume_complete_required=bool(volume_complete_required),
    )

    graphs, dates, tickers, stats = build_rolling_corr_graphs(
        returns,
        volume,
        membership_map,
        cfg,
        fundamentals=fundamentals,
        macro=macro,
        static_edges=static_edges,
        num_workers=workers,
        parallel_backend=parallel_backend,
        joblib_prefer=joblib_prefer,
        joblib_n_jobs=joblib_n_jobs,
        progress=progress,
    )
    out_path = save_graph_artifact(
        out_path,
        graphs=graphs,
        dates=dates,
        tickers=tickers,
        config=cfg.__dict__,
        stats=stats,
        artifact_format=artifact_format,
        shard_size=shard_size,
    )
    if dates:
        start_date = dates[0]
        end_date = dates[-1]
    else:
        start_date = "n/a"
        end_date = "n/a"
    print(
        f"Wrote {out_path} with {len(graphs)} graphs "
        f"(format={artifact_format}, shard_size={shard_size if artifact_format == 'sharded' else 'n/a'})"
    )
    print(
        f"Date range: {start_date} -> {end_date} | "
        f"windows: {stats.get('total_windows', 0)} | "
        f"built: {stats.get('built', 0)} | "
        f"skipped_lag={stats.get('skipped_lag_history', 0)}, "
        f"skipped: members={stats.get('skipped_no_members', 0)}, "
        f"cols={stats.get('skipped_no_cols', 0)}, "
        f"min_nodes={stats.get('skipped_min_nodes', 0)}, "
        f"no_edges={stats.get('skipped_no_edges', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
