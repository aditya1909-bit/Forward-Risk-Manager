#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_LABELS = {
    "ff_accurate": "FF Accurate",
    "ff_financial": "FF Financial",
    "ff_bootstrap_rank": "FF Bootstrap Rank",
    "backprop_supervised_return": "Supervised Return",
    "ff_e2e_core": "FF Core",
    "backprop_contrastive_core": "Backprop Core",
}

METRIC_LABELS = {
    "primary_eval_metric": "Primary score",
    "econ_ls_oos_sharpe_uplift_min": "Long/short worst-fold Sharpe uplift",
    "econ_ls_sharpe_uplift": "Long/short Sharpe uplift",
    "econ_exposure_adjusted_sharpe_uplift": "Exposure-adjusted Sharpe uplift",
    "econ_oos_sharpe_uplift_min": "Timing worst-fold Sharpe uplift",
    "econ_sharpe_uplift": "Full buy-hold Sharpe uplift",
    "graphs_per_s": "Graphs/sec",
    "avg_epoch_s": "Seconds/epoch",
    "time_tracked_step_s": "Tracked seconds/step",
}

PALETTE = ["#35638f", "#c76f4b", "#4f8f6a", "#9a6fb0", "#d0a63d", "#6f6f6f"]


def _label_mode(value: object) -> str:
    raw = str(value)
    return MODE_LABELS.get(raw, raw.replace("_", " ").title())


def _label_metric(value: object) -> str:
    raw = str(value)
    return METRIC_LABELS.get(raw, raw.replace("_", " ").title())


def _finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "row_type" in df.columns:
        agg = df[df["row_type"].astype(str).eq("aggregate")].copy()
        if not agg.empty:
            return agg
    if "seed_run" in df.columns:
        agg = df[df["seed_run"].astype(str).eq("ALL")].copy()
        if not agg.empty:
            return agg
    return df.copy()


def _prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["mode"].map(_label_mode) if "mode" in out.columns else ""
    if "primary_eval_metric_name" in out.columns:
        out["primary_metric"] = out["primary_eval_metric_name"].map(_label_metric)
    else:
        out["primary_metric"] = "Primary score"
    return out


def _style_axis(ax, title: str, xlabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#d8dde6", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#b6bfca")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_overview(agg: pd.DataFrame, out_path: Path) -> None:
    data = agg.copy()
    data["score"] = _finite_series(data, "primary_eval_metric")
    data = data.dropna(subset=["score"]).sort_values("score", ascending=True)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    ax.barh(data["model"], data["score"], color=colors, height=0.58)
    ax.axvline(0, color="#222222", linewidth=1.0)
    for y, (_, row) in enumerate(data.iterrows()):
        val = float(row["score"])
        ha = "left"
        offset = 0.008
        ax.text(val + offset, y, f"{val:+.3f}", va="center", ha=ha, fontsize=9)
    xmin = min(0.0, float(data["score"].min()))
    xmax = max(0.0, float(data["score"].max()))
    pad = max(0.02, (xmax - xmin) * 0.12)
    ax.set_xlim(xmin - pad, xmax + pad)
    metric_names = sorted(str(x) for x in data.get("primary_metric", pd.Series()).dropna().unique())
    subtitle = metric_names[0] if len(metric_names) == 1 else "Mixed primary metrics"
    _style_axis(ax, f"Model Ranking ({subtitle})", "Sharpe uplift")
    _save(fig, out_path)


def _plot_economics(agg: pd.DataFrame, out_path: Path) -> None:
    columns = [
        "econ_ls_oos_sharpe_uplift_min",
        "econ_ls_sharpe_uplift",
        "econ_exposure_adjusted_sharpe_uplift",
        "econ_oos_sharpe_uplift_min",
        "econ_sharpe_uplift",
    ]
    available = [c for c in columns if c in agg.columns and _finite_series(agg, c).notna().any()]
    if not available:
        return
    data = agg.sort_values("primary_eval_metric", ascending=False, na_position="last").copy()
    x = np.arange(len(data))
    width = min(0.16, 0.75 / max(1, len(available)))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for idx, col in enumerate(available):
        values = _finite_series(data, col).to_numpy(dtype=float)
        ax.bar(
            x + (idx - (len(available) - 1) / 2) * width,
            values,
            width=width,
            label=_label_metric(col),
            color=PALETTE[idx % len(PALETTE)],
        )
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(data["model"], rotation=20, ha="right")
    ax.legend(frameon=False, ncols=2, fontsize=9)
    _style_axis(ax, "Financial Metrics", "Sharpe uplift")
    ax.set_xlabel("")
    ax.set_ylabel("Sharpe uplift")
    _save(fig, out_path)


def _plot_speed(agg: pd.DataFrame, out_path: Path) -> None:
    parts = [
        ("time_forward_pos_s", "Positive pass"),
        ("time_forward_neg_s", "Negative pass"),
        ("time_neg_gen_s", "Negative generation"),
        ("time_hallucinate_s", "Hallucination"),
        ("time_loss_terms_s", "Loss terms"),
        ("time_optimizer_s", "Optimizer"),
    ]
    available = [(c, label) for c, label in parts if c in agg.columns and _finite_series(agg, c).notna().any()]
    if not available:
        return
    data = agg.sort_values("time_tracked_step_s", ascending=True, na_position="last").copy()
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 5.4))
    left = np.zeros(len(data), dtype=float)
    for idx, (col, label) in enumerate(available):
        values = _finite_series(data, col).fillna(0).to_numpy(dtype=float)
        ax.barh(y, values, left=left, label=label, color=PALETTE[idx % len(PALETTE)], height=0.58)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(data["model"])
    ax.legend(frameon=False, ncols=2, fontsize=9)
    _style_axis(ax, "Per-Step Runtime Breakdown", "Seconds per tracked step")
    _save(fig, out_path)


def _plot_fold_heatmap(folds: pd.DataFrame, out_path: Path) -> None:
    if folds.empty or "fold_id" not in folds.columns:
        return
    data = _prepare_summary(folds)
    data["score"] = _finite_series(data, "primary_eval_metric")
    grouped = data.groupby(["model", "fold_id"], dropna=False)["score"].mean().reset_index()
    if grouped["score"].notna().sum() == 0:
        return
    pivot = grouped.pivot(index="model", columns="fold_id", values="score")
    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.6 * len(pivot) + 1.6)))
    values = pivot.to_numpy(dtype=float)
    limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    im = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"Fold {c}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Primary score")
    ax.set_title("Fold Stability", loc="left", fontsize=13, fontweight="bold", pad=10)
    _save(fig, out_path)


def _plot_graph_timing_comparison(agg: pd.DataFrame, out_path: Path) -> None:
    metric = "econ_exposure_adjusted_sharpe_uplift"
    if metric not in agg.columns or _finite_series(agg, metric).notna().sum() == 0:
        return
    data = agg.copy()
    data["score"] = _finite_series(data, metric)
    data = data.dropna(subset=["score"]).sort_values("score", ascending=True)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    ax.barh(data["model"], data["score"], color=colors, height=0.58)
    ax.axvline(0, color="#222222", linewidth=1.0)
    for y, (_, row) in enumerate(data.iterrows()):
        ax.text(float(row["score"]) + 0.008, y, f"{float(row['score']):+.3f}", va="center", fontsize=9)
    xmin = min(0.0, float(data["score"].min()))
    xmax = max(0.0, float(data["score"].max()))
    pad = max(0.02, (xmax - xmin) * 0.12)
    ax.set_xlim(xmin - pad, xmax + pad)
    _style_axis(ax, "Graph-Timing Comparison", "Exposure-adjusted Sharpe uplift")
    _save(fig, out_path)


def _paired_ff_vs_backprop(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = "econ_exposure_adjusted_sharpe_uplift"
    required = {"mode", "seed_run", "fold_id", metric}
    if folds.empty or not required.issubset(set(folds.columns)):
        return pd.DataFrame(), pd.DataFrame()
    data = folds.copy()
    data["score"] = _finite_series(data, metric)
    data = data.dropna(subset=["score"])
    bp = data[data["mode"].eq("backprop_supervised_return")][["seed_run", "fold_id", "score"]].rename(
        columns={"score": "backprop_score"}
    )
    ff = data[data["mode"].isin(["ff_accurate", "ff_financial", "ff_bootstrap_rank"])][
        ["mode", "seed_run", "fold_id", "score"]
    ].rename(columns={"score": "ff_score"})
    paired = ff.merge(bp, on=["seed_run", "fold_id"], how="inner")
    if paired.empty:
        return paired, pd.DataFrame()
    paired["ff_delta_vs_backprop"] = paired["ff_score"] - paired["backprop_score"]
    paired["model"] = paired["mode"].map(_label_mode)
    best = paired.loc[paired.groupby(["seed_run", "fold_id"])["ff_delta_vs_backprop"].idxmax()].copy()
    best["ff_beats_backprop"] = best["ff_delta_vs_backprop"] > 0
    return paired, best


def _plot_best_ff_vs_backprop(best: pd.DataFrame, out_path: Path) -> None:
    if best.empty:
        return
    data = best.sort_values(["seed_run", "fold_id"]).copy()
    data["slice"] = data.apply(lambda r: f"seed {int(r['seed_run'])}, fold {int(r['fold_id'])}", axis=1)
    colors = [PALETTE[2] if v > 0 else PALETTE[1] for v in data["ff_delta_vs_backprop"]]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(data))
    ax.bar(x, data["ff_delta_vs_backprop"], color=colors, width=0.68)
    ax.axhline(0, color="#222222", linewidth=1.0)
    for idx, (_, row) in enumerate(data.iterrows()):
        y = float(row["ff_delta_vs_backprop"])
        va = "bottom" if y >= 0 else "top"
        offset = 0.012 if y >= 0 else -0.012
        ax.text(idx, y + offset, str(row["model"]), ha="center", va=va, fontsize=8, rotation=90)
    wins = int((data["ff_delta_vs_backprop"] > 0).sum())
    title = f"Best FF vs Backprop by Paired Slice ({wins}/{len(data)} FF wins)"
    ax.set_xticks(x)
    ax.set_xticklabels(data["slice"], rotation=25, ha="right")
    ax.set_ylabel("FF minus backprop")
    _style_axis(ax, title, "Same seed/fold, exposure-adjusted Sharpe uplift")
    _save(fig, out_path)


def _plot_ff_win_rate(paired: pd.DataFrame, best: pd.DataFrame, out_path: Path) -> None:
    if paired.empty:
        return
    by_model = (
        paired.assign(win=paired["ff_delta_vs_backprop"] > 0)
        .groupby("model", as_index=False)
        .agg(win_rate=("win", "mean"), wins=("win", "sum"), n=("win", "size"), mean_delta=("ff_delta_vs_backprop", "mean"))
        .sort_values(["win_rate", "mean_delta"], ascending=True)
    )
    if not best.empty:
        best_row = pd.DataFrame(
            [
                {
                    "model": "Best FF per slice",
                    "win_rate": float((best["ff_delta_vs_backprop"] > 0).mean()),
                    "wins": int((best["ff_delta_vs_backprop"] > 0).sum()),
                    "n": int(len(best)),
                    "mean_delta": float(best["ff_delta_vs_backprop"].mean()),
                }
            ]
        )
        by_model = pd.concat([by_model, best_row], ignore_index=True)
    if by_model.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = [PALETTE[2] if v >= 0.5 else PALETTE[1] for v in by_model["win_rate"]]
    ax.barh(by_model["model"], by_model["win_rate"], color=colors, height=0.58)
    ax.axvline(0.5, color="#222222", linewidth=1.0)
    for y, (_, row) in enumerate(by_model.iterrows()):
        ax.text(
            min(0.98, float(row["win_rate"]) + 0.03),
            y,
            f"{int(row['wins'])}/{int(row['n'])}",
            va="center",
            fontsize=9,
        )
    if not best.empty:
        best_wins = int((best["ff_delta_vs_backprop"] > 0).sum())
        title = f"FF Win Rate vs Backprop (Best FF wins {best_wins}/{len(best)} paired slices)"
    else:
        title = "FF Win Rate vs Backprop"
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.linspace(0, 1, 6))
    _style_axis(ax, title, "Share of seed/fold slices where FF beats backprop")
    _save(fig, out_path)


def _plot_ff_delta_heatmap(paired: pd.DataFrame, out_path: Path) -> None:
    if paired.empty:
        return
    data = paired.copy()
    data["slice"] = data.apply(lambda r: f"s{int(r['seed_run'])}/f{int(r['fold_id'])}", axis=1)
    pivot = data.pivot(index="model", columns="slice", values="ff_delta_vs_backprop")
    order = sorted(pivot.columns, key=lambda x: tuple(int(part[1:]) for part in x.split("/")))
    pivot = pivot[order]
    values = pivot.to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return
    limit = max(0.05, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(10.5, max(3.8, 0.62 * len(pivot) + 1.6)))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.86, label="FF minus backprop")
    ax.set_title("FF Advantage Heatmap", loc="left", fontsize=13, fontweight="bold", pad=10)
    _save(fig, out_path)


def _plot_ff_vs_backprop_scatter(paired: pd.DataFrame, out_path: Path) -> None:
    if paired.empty:
        return
    data = paired.copy()
    if data[["ff_score", "backprop_score"]].dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    models = list(data["model"].dropna().unique())
    for idx, model in enumerate(models):
        sub = data[data["model"].eq(model)]
        ax.scatter(
            sub["backprop_score"],
            sub["ff_score"],
            s=70,
            alpha=0.82,
            label=model,
            color=PALETTE[idx % len(PALETTE)],
            edgecolor="white",
            linewidth=0.7,
        )
    vals = pd.concat([data["backprop_score"], data["ff_score"]]).dropna()
    lo = min(0.0, float(vals.min()))
    hi = max(0.0, float(vals.max()))
    pad = max(0.04, (hi - lo) * 0.12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#222222", linewidth=1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylabel("FF exposure-adjusted Sharpe uplift")
    _style_axis(ax, "FF vs Backprop Paired Scores", "Backprop exposure-adjusted Sharpe uplift")
    ax.grid(axis="both", color="#d8dde6", linewidth=0.8, alpha=0.75)
    _save(fig, out_path)


def _plot_best_ff_score_lift(best: pd.DataFrame, out_path: Path) -> None:
    if best.empty:
        return
    data = best.sort_values("ff_delta_vs_backprop", ascending=True).copy()
    data["slice"] = data.apply(lambda r: f"seed {int(r['seed_run'])}, fold {int(r['fold_id'])}", axis=1)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.hlines(y, data["backprop_score"], data["ff_score"], color="#aab4c0", linewidth=2.0)
    ax.scatter(data["backprop_score"], y, label="Backprop", color="#6f6f6f", s=52, zorder=3)
    ax.scatter(data["ff_score"], y, label="Best FF", color=PALETTE[2], s=64, zorder=3)
    ax.axvline(0, color="#222222", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(data["slice"])
    for yi, (_, row) in enumerate(data.iterrows()):
        ax.text(
            max(float(row["backprop_score"]), float(row["ff_score"])) + 0.02,
            yi,
            str(row["model"]),
            va="center",
            fontsize=8,
        )
    ax.legend(frameon=False, fontsize=9)
    _style_axis(ax, "Best FF Lifts Paired Score", "Exposure-adjusted Sharpe uplift")
    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write friendly benchmark summaries and report plots.")
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--fold-csv", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="benchmark")
    args = parser.parse_args()

    benchmark_csv = Path(args.benchmark_csv)
    out_dir = Path(args.out_dir)
    df = pd.read_csv(benchmark_csv)
    agg = _prepare_summary(_aggregate_rows(df))
    summary_cols = [
        c
        for c in [
            "model",
            "mode",
            "primary_metric",
            "primary_eval_metric",
            "econ_ls_oos_sharpe_uplift_min",
            "econ_ls_sharpe_uplift",
            "econ_exposure_adjusted_sharpe_uplift",
            "econ_oos_sharpe_uplift_min",
            "econ_sharpe_uplift",
            "graphs_per_s",
            "time_tracked_step_s",
            "avg_epoch_s",
            "resume_loaded_completed",
            "resume_schema_version",
            "status",
        ]
        if c in agg.columns
    ]
    summary_path = benchmark_csv.with_name(f"{args.prefix}_summary_named.csv")
    agg[summary_cols].sort_values("primary_eval_metric", ascending=False, na_position="last").to_csv(
        summary_path,
        index=False,
    )
    print(f"Wrote {summary_path}")

    _plot_overview(agg, out_dir / f"{args.prefix}_model_ranking.png")
    _plot_graph_timing_comparison(agg, out_dir / f"{args.prefix}_graph_timing_comparison.png")
    _plot_economics(agg, out_dir / f"{args.prefix}_financial_metrics.png")
    _plot_speed(agg, out_dir / f"{args.prefix}_speed_breakdown.png")
    if args.fold_csv:
        fold_path = Path(args.fold_csv)
        if fold_path.exists():
            folds = pd.read_csv(fold_path)
            _plot_fold_heatmap(folds, out_dir / f"{args.prefix}_fold_stability.png")
            paired, best = _paired_ff_vs_backprop(folds)
            if not paired.empty:
                paired_path = benchmark_csv.with_name(f"{args.prefix}_ff_vs_backprop_paired_slices.csv")
                paired.sort_values("ff_delta_vs_backprop", ascending=False).to_csv(paired_path, index=False)
                print(f"Wrote {paired_path}")
            if not best.empty:
                best_path = benchmark_csv.with_name(f"{args.prefix}_best_ff_vs_backprop_slices.csv")
                best.sort_values("ff_delta_vs_backprop", ascending=False).to_csv(best_path, index=False)
                print(f"Wrote {best_path}")
                _plot_best_ff_vs_backprop(best, out_dir / f"{args.prefix}_best_ff_vs_backprop_slices.png")
            if not paired.empty:
                _plot_ff_win_rate(paired, best, out_dir / f"{args.prefix}_ff_win_rate_vs_backprop.png")
                _plot_ff_delta_heatmap(paired, out_dir / f"{args.prefix}_ff_advantage_heatmap.png")
                _plot_ff_vs_backprop_scatter(paired, out_dir / f"{args.prefix}_ff_vs_backprop_scatter.png")
            if not best.empty:
                _plot_best_ff_score_lift(best, out_dir / f"{args.prefix}_best_ff_score_lift.png")


if __name__ == "__main__":
    main()
