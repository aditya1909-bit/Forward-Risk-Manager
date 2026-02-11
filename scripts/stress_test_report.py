#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np


def _load_rows(path: Path):
    with path.open() as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError("CSV is empty")
    return rows


def _infer_return_cols(rows):
    cols = [c for c in rows[0].keys() if c.startswith("r")]
    cols = sorted(cols, key=lambda x: int(x[1:]))
    return cols


def _calc_max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(dd.min())


def _var_cvar(x: np.ndarray, alpha: float = 0.95):
    # value-at-risk and conditional VaR on returns (more negative is worse)
    if x.size == 0:
        return 0.0, 0.0
    q = np.quantile(x, 1 - alpha)
    tail = x[x <= q]
    cvar = tail.mean() if tail.size > 0 else q
    return float(q), float(cvar)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate stress test report from hallucination windows.")
    parser.add_argument(
        "--csv",
        default="runs/experiments/manual/metrics/scenario_book.csv",
        help="Input CSV",
    )
    parser.add_argument(
        "--out-csv",
        default="runs/experiments/manual/metrics/stress_test_report.csv",
        help="Output CSV",
    )
    parser.add_argument(
        "--out-plot",
        default="runs/experiments/manual/plots/stress_test_report.png",
        help="Output PNG",
    )
    parser.add_argument("--target-ticker", default="", help="Optional target ticker for focused diagnostics.")
    args = parser.parse_args()
    target_ticker = args.target_ticker.strip().upper() if args.target_ticker else ""

    rows = _load_rows(Path(args.csv))
    ret_cols = _infer_return_cols(rows)

    scenario_col = "scenario_id" if "scenario_id" in rows[0] else "t"

    grouped = defaultdict(list)
    for row in rows:
        scenario = row.get(scenario_col, "0")
        series = row["series"]
        key = (scenario, series)
        ticker = row.get("ticker", "")
        rets = np.array([float(row[c]) for c in ret_cols], dtype=float)
        grouped[key].append((ticker, rets))

    metrics = []
    curves = defaultdict(list)
    scenarios = sorted({k[0] for k in grouped.keys()})
    scopes = ["all"]
    if target_ticker:
        scopes.extend(["target", "non_target"])

    for scenario in scenarios:
        for scope in scopes:
            for series in ("real", "halluc"):
                entries = grouped.get((scenario, series))
                if not entries:
                    continue
                if scope == "all":
                    selected = [rets for _, rets in entries]
                elif scope == "target":
                    selected = [rets for ticker, rets in entries if ticker.upper() == target_ticker]
                elif scope == "non_target":
                    selected = [rets for ticker, rets in entries if ticker.upper() != target_ticker]
                else:
                    selected = []
                if not selected:
                    continue

                rets = np.stack(selected, axis=0)  # [N, T]
                port_ret = rets.mean(axis=0)
                cum = np.exp(np.cumsum(port_ret))
                total_ret = float(cum[-1] - 1.0)
                max_dd = _calc_max_drawdown(cum)
                vol = float(np.std(port_ret))
                var95, cvar95 = _var_cvar(port_ret, alpha=0.95)
                metrics.append(
                    {
                        "scenario": scenario,
                        "scope": scope,
                        "series": series,
                        "num_tickers": int(rets.shape[0]),
                        "total_return": total_ret,
                        "max_drawdown": max_dd,
                        "volatility": vol,
                        "var_95": var95,
                        "cvar_95": cvar95,
                    }
                )
                curves[(scope, series)].append(cum)

    # Save metrics CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "scope",
                "series",
                "num_tickers",
                "total_return",
                "max_drawdown",
                "volatility",
                "var_95",
                "cvar_95",
            ],
        )
        w.writeheader()
        for row in metrics:
            w.writerow(row)

    print(f"Wrote {out_csv}")
    try:
        import matplotlib.pyplot as plt

        # Plot: mean curves + worst-case
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        if curves[("all", "real")]:
            real_mean = np.mean(np.stack(curves[("all", "real")], axis=0), axis=0)
            ax.plot(real_mean, label="real all (mean)")
        if curves[("all", "halluc")]:
            hall_mean = np.mean(np.stack(curves[("all", "halluc")], axis=0), axis=0)
            ax.plot(hall_mean, label="halluc all (mean)")
        if target_ticker:
            if curves[("target", "real")]:
                real_target = np.mean(np.stack(curves[("target", "real")], axis=0), axis=0)
                ax.plot(real_target, linestyle="--", label=f"real {target_ticker}")
            if curves[("target", "halluc")]:
                hall_target = np.mean(np.stack(curves[("target", "halluc")], axis=0), axis=0)
                ax.plot(hall_target, linestyle="--", label=f"halluc {target_ticker}")
        ax.set_title("Mean Portfolio Paths")
        ax.set_xlabel("Window step")
        ax.set_ylabel("Cumulative return")
        ax.legend()

        # Worst-case halluc scenario by max drawdown on target scope if available, else all.
        ax = axes[1]
        focus_scope = "target" if target_ticker and any(r["scope"] == "target" for r in metrics) else "all"
        worst = None
        for row in metrics:
            if row["series"] != "halluc" or row["scope"] != focus_scope:
                continue
            if worst is None or row["max_drawdown"] < worst["max_drawdown"]:
                worst = row
        if worst is not None:
            scenario = worst["scenario"]
            hall_entries = grouped[(scenario, "halluc")]
            if focus_scope == "target":
                hall_sel = [rets for ticker, rets in hall_entries if ticker.upper() == target_ticker]
            else:
                hall_sel = [rets for _, rets in hall_entries]
            hall_rets = np.stack(hall_sel, axis=0).mean(axis=0)
            hall_cum = np.exp(np.cumsum(hall_rets))
            ax.plot(hall_cum, label=f"halluc worst {focus_scope} (scn {scenario})")
            if (scenario, "real") in grouped:
                real_entries = grouped[(scenario, "real")]
                if focus_scope == "target":
                    real_sel = [rets for ticker, rets in real_entries if ticker.upper() == target_ticker]
                else:
                    real_sel = [rets for _, rets in real_entries]
                if real_sel:
                    real_rets = np.stack(real_sel, axis=0).mean(axis=0)
                    real_cum = np.exp(np.cumsum(real_rets))
                    ax.plot(real_cum, label=f"real {focus_scope} (scn {scenario})")
        ax.set_title(f"Worst Halluc Scenario ({focus_scope})")
        ax.set_xlabel("Window step")
        ax.set_ylabel("Cumulative return")
        ax.legend()

        fig.tight_layout()
        out_plot = Path(args.out_plot)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_plot}")
    except ModuleNotFoundError:
        print("matplotlib not available; skipping plot output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
