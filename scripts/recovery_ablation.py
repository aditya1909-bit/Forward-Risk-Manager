#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import tomllib
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
_BUILD_SIGNATURE_IGNORE = {
    "out",
    "workers",
    "parallel_backend",
    "joblib_prefer",
    "joblib_n_jobs",
    "progress",
}


def _load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isfinite(v):
            return repr(float(v))
        raise ValueError("non-finite float cannot be serialized to TOML")
    if isinstance(v, str):
        esc = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{esc}"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"Unsupported TOML value: {type(v)}")


def _write_section(lines: list[str], name: str, data: dict):
    lines.append(f"[{name}]")
    for k, v in data.items():
        if isinstance(v, dict):
            continue
        lines.append(f"{k} = {_toml_value(v)}")
    lines.append("")
    for k, v in data.items():
        if isinstance(v, dict):
            _write_section(lines, f"{name}.{k}", v)


def _write_config(path: Path, cfg: dict, sections: list[str]) -> None:
    lines: list[str] = []
    for section in sections:
        data = cfg.get(section, {})
        if not isinstance(data, dict):
            continue
        _write_section(lines, section, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _build_signature(build_cfg: dict) -> str:
    filtered = {}
    for key, value in build_cfg.items():
        if key in _BUILD_SIGNATURE_IGNORE:
            continue
        filtered[key] = value
    blob = json.dumps(filtered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _default_experiments() -> list[dict]:
    return [
        {
            "id": "baseline_updated",
            "family": "baseline",
            "description": "Current Updated Model settings from config.",
            "overrides": {},
        },
        {
            "id": "neg_shuffle_noise",
            "family": "negative_sampling",
            "description": "Replace mix negatives with simpler shuffle+noise.",
            "overrides": {
                "train": {"neg_mode": "shuffle+noise"},
                "benchmark": {"neg_mode": "shuffle+noise", "eval_neg_mode": "auto"},
            },
        },
        {
            "id": "neg_mix_gentle",
            "family": "negative_sampling",
            "description": "Keep mix mode but slow down and cap hallucination ratio.",
            "overrides": {
                "train": {"neg_mode": "mix", "neg_mix_end": 0.35, "neg_mix_ramp_epochs": 80},
                "benchmark": {"neg_mode": "mix", "eval_neg_mode": "auto"},
            },
        },
        {
            "id": "risk_head_off",
            "family": "risk_head",
            "description": "Disable risk-head to isolate FF objective behavior.",
            "overrides": {"train": {"risk_head_enabled": False}},
        },
        {
            "id": "risk_head_weight_low",
            "family": "risk_head",
            "description": "Keep risk-head but down-weight auxiliary loss.",
            "overrides": {"train": {"risk_head_enabled": True, "risk_loss_weight": 0.02}},
        },
        {
            "id": "risk_head_weight_lower",
            "family": "risk_head",
            "description": "Keep risk-head with lower auxiliary loss weight (0.01).",
            "overrides": {"train": {"risk_head_enabled": True, "risk_loss_weight": 0.01}},
        },
        {
            "id": "risk_head_weight_min",
            "family": "risk_head",
            "description": "Keep risk-head with minimal auxiliary loss weight (0.005).",
            "overrides": {"train": {"risk_head_enabled": True, "risk_loss_weight": 0.005}},
        },
        {
            "id": "graph_topk10_pearson_xsec",
            "family": "graph_params",
            "description": "Denser Pearson graph with top_k=10 and cross-sectional normalization.",
            "overrides": {
                "build_graphs": {
                    "top_k": 10,
                    "corr_method": "pearson",
                    "edge_select_mode": "top_k",
                    "corr_threshold": "",
                    "cross_sectional_norm": True,
                }
            },
        },
        {
            "id": "graph_topk10_partial",
            "family": "graph_params",
            "description": "Denser partial-correlation graph with top_k=10.",
            "overrides": {
                "build_graphs": {
                    "top_k": 10,
                    "corr_method": "partial",
                    "edge_select_mode": "top_k",
                    "corr_threshold": "",
                    "cross_sectional_norm": False,
                }
            },
        },
        {
            "id": "graph_threshold_pearson",
            "family": "graph_params",
            "description": "Threshold-based Pearson edges to test connectivity tradeoff.",
            "overrides": {
                "build_graphs": {
                    "corr_method": "pearson",
                    "edge_select_mode": "threshold",
                    "corr_threshold": 0.25,
                    "cross_sectional_norm": False,
                }
            },
        },
        {
            "id": "goodtemp_050_margin000",
            "family": "goodness_margin",
            "description": "Higher temperature without margin.",
            "overrides": {"train": {"goodness_temp": 0.5, "ff_margin": 0.0, "ff_margin_weight": 1.0}},
        },
        {
            "id": "goodtemp_025_margin010",
            "family": "goodness_margin",
            "description": "Current temperature with positive FF margin.",
            "overrides": {"train": {"goodness_temp": 0.25, "ff_margin": 0.1, "ff_margin_weight": 1.0}},
        },
        {
            "id": "goodtemp_050_margin010",
            "family": "goodness_margin",
            "description": "Higher temperature with positive FF margin.",
            "overrides": {"train": {"goodness_temp": 0.5, "ff_margin": 0.1, "ff_margin_weight": 1.0}},
        },
        {
            "id": "hall_stronger_steps_nodes",
            "family": "hallucination",
            "description": "Stronger hallucination schedule via more steps and node coverage.",
            "overrides": {"train": {"hallucinate_steps": 8, "hallucinate_node_fraction": 0.6}},
        },
        {
            "id": "hall_moment_penalties",
            "family": "hallucination",
            "description": "Reintroduce moment penalties for distributional realism.",
            "overrides": {
                "train": {
                    "hallucinate_steps": 8,
                    "hallucinate_node_fraction": 0.6,
                    "hallucinate_moment_mean": 0.02,
                    "hallucinate_moment_var": 0.02,
                    "hallucinate_moment_skew": 0.01,
                }
            },
        },
        {
            "id": "split_chronological",
            "family": "split_strategy",
            "description": "Chronological single-holdout benchmark split.",
            "overrides": {"benchmark": {"split_mode": "chronological"}},
        },
        {
            "id": "split_walk_forward_tighter_step",
            "family": "split_strategy",
            "description": "Walk-forward with smaller step for denser fold coverage.",
            "overrides": {"benchmark": {"split_mode": "walk_forward", "walk_forward_step_frac": 0.05}},
        },
    ]


def _read_csv_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _float_or_nan(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not math.isfinite(out):
        return float("nan")
    return out


def _add_baseline_deltas(rows: list[dict], baseline_id: str) -> None:
    baseline_by_mode: dict[str, dict] = {}
    for row in rows:
        if row.get("ablation_id") != baseline_id:
            continue
        if row.get("row_type") == "fold":
            continue
        mode = str(row.get("mode", "")).strip()
        if mode:
            baseline_by_mode[mode] = row

    tracked = ("eval_sep", "econ_sharpe_uplift", "econ_ann_return_uplift")
    for row in rows:
        mode = str(row.get("mode", "")).strip()
        base = baseline_by_mode.get(mode)
        for metric in tracked:
            key = f"delta_{metric}_vs_baseline"
            if base is None:
                row[key] = ""
                continue
            row_val = _float_or_nan(row.get(metric, "nan"))
            base_val = _float_or_nan(base.get(metric, "nan"))
            if math.isfinite(row_val) and math.isfinite(base_val):
                row[key] = f"{row_val - base_val:.10g}"
            else:
                row[key] = ""


def _family_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("row_type") == "fold":
            continue
        fam = str(row.get("ablation_family", ""))
        grouped.setdefault(fam, []).append(row)

    summary_rows: list[dict] = []
    for family, fam_rows in sorted(grouped.items()):
        best_sep = None
        best_sharpe = None
        for row in fam_rows:
            sep = _float_or_nan(row.get("eval_sep", "nan"))
            shp = _float_or_nan(row.get("econ_sharpe_uplift", "nan"))
            if math.isfinite(sep) and (best_sep is None or sep > best_sep[0]):
                best_sep = (sep, row)
            if math.isfinite(shp) and (best_sharpe is None or shp > best_sharpe[0]):
                best_sharpe = (shp, row)
        out = {"ablation_family": family}
        if best_sep is not None:
            out["best_sep_ablation_id"] = best_sep[1].get("ablation_id", "")
            out["best_sep_mode"] = best_sep[1].get("mode", "")
            out["best_sep_value"] = f"{best_sep[0]:.10g}"
        if best_sharpe is not None:
            out["best_sharpe_ablation_id"] = best_sharpe[1].get("ablation_id", "")
            out["best_sharpe_mode"] = best_sharpe[1].get("mode", "")
            out["best_sharpe_value"] = f"{best_sharpe[0]:.10g}"
        summary_rows.append(out)
    return summary_rows


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run controlled ablations targeting separation/economic regressions in the Updated Model."
    )
    parser.add_argument("--config", default="configs/default.toml", help="Base TOML config path.")
    parser.add_argument(
        "--run-dir",
        default="runs/experiments/recovery_ablation",
        help="Output directory for configs/graphs/metrics.",
    )
    parser.add_argument(
        "--modes",
        default="ff_e2e,ff_layerwise,backprop",
        help="Benchmark modes passed to benchmark_training.py.",
    )
    parser.add_argument(
        "--families",
        default="",
        help=(
            "Optional comma-separated subset of families: "
            "negative_sampling,risk_head,graph_params,goodness_margin,hallucination,split_strategy"
        ),
    )
    parser.add_argument("--benchmark-epochs", type=int, default=0, help="Override benchmark epochs when > 0.")
    parser.add_argument("--max-experiments", type=int, default=0, help="Run at most N experiments when > 0.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned matrix without running commands.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures and continue to remaining experiments.",
    )
    parser.add_argument(
        "--benchmark-batch-size",
        type=int,
        default=0,
        help="Override benchmark batch size when > 0 (useful for smaller Colab GPUs).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    base_cfg = _load_config(cfg_path)
    base_sections = {
        "build_graphs": copy.deepcopy(base_cfg.get("build_graphs", {})),
        "train": copy.deepcopy(base_cfg.get("train", {})),
        "benchmark": copy.deepcopy(base_cfg.get("benchmark", {})),
    }

    experiments = _default_experiments()
    family_filter = {x.strip() for x in args.families.split(",") if x.strip()}
    if family_filter:
        experiments = [
            e
            for e in experiments
            if e["family"] == "baseline" or e["family"] in family_filter
        ]
    if args.max_experiments > 0:
        experiments = experiments[: args.max_experiments]
    if not experiments:
        raise ValueError("No experiments selected.")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    graphs_dir = run_dir / "graphs"
    configs_dir = run_dir / "configs"
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    for d in (graphs_dir, configs_dir, metrics_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    graph_cache: dict[str, Path] = {}
    all_rows: list[dict] = []
    plan_rows: list[dict] = []

    pbar = tqdm(
        total=len(experiments),
        desc="RecoveryAblation",
        unit="exp",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    exp_durations: list[float] = []

    for exp in experiments:
        exp_t0 = time.perf_counter()
        exp_id = str(exp["id"])
        overrides = exp.get("overrides", {})
        cfg_exp = {
            "build_graphs": copy.deepcopy(base_sections["build_graphs"]),
            "train": copy.deepcopy(base_sections["train"]),
            "benchmark": copy.deepcopy(base_sections["benchmark"]),
        }
        _deep_update(cfg_exp, overrides)
        if args.benchmark_epochs > 0:
            cfg_exp["benchmark"]["epochs"] = int(args.benchmark_epochs)
        if args.benchmark_batch_size > 0:
            cfg_exp["benchmark"]["batch_size"] = int(args.benchmark_batch_size)

        build_sig = _build_signature(cfg_exp["build_graphs"])
        graph_path = graphs_dir / f"graphs_{build_sig}.pt"
        cfg_exp["build_graphs"]["out"] = str(graph_path)
        cfg_exp["train"]["graphs"] = str(graph_path)
        cfg_exp["benchmark"]["out_csv"] = str(metrics_dir / f"benchmark_{exp_id}.csv")
        cfg_exp["benchmark"]["walk_forward_out_csv"] = str(
            metrics_dir / f"benchmark_{exp_id}_folds.csv"
        )
        cfg_exp["benchmark"]["plot_path"] = str(plots_dir / f"benchmark_{exp_id}_speed_sep.png")
        cfg_exp["benchmark"]["bar_plot_path"] = str(plots_dir / f"benchmark_{exp_id}_bars.png")

        plan_rows.append(
            {
                "ablation_id": exp_id,
                "ablation_family": exp.get("family", ""),
                "ablation_description": exp.get("description", ""),
                "graph_signature": build_sig,
                "graph_path": str(graph_path),
                "overrides_json": json.dumps(overrides, sort_keys=True),
            }
        )

        if args.dry_run:
            pbar.set_postfix(exp=exp_id, status="planned")
            pbar.update(1)
            continue

        if build_sig not in graph_cache:
            build_cfg_path = configs_dir / f"build_{build_sig}.toml"
            _write_config(build_cfg_path, cfg_exp, sections=["build_graphs"])
            if not graph_path.exists():
                print(f"[build] {exp_id} -> {graph_path}")
                try:
                    _run_cmd(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "build_graphs.py"),
                            "--config",
                            str(build_cfg_path),
                        ],
                        cwd=ROOT,
                    )
                except Exception:
                    if args.continue_on_error:
                        all_rows.append(
                            {
                                "ablation_id": exp_id,
                                "ablation_family": exp.get("family", ""),
                                "ablation_description": exp.get("description", ""),
                                "mode": "",
                                "status": "build_failed",
                            }
                        )
                        pbar.set_postfix(exp=exp_id, status="build_failed")
                        pbar.update(1)
                        exp_durations.append(time.perf_counter() - exp_t0)
                        continue
                    pbar.set_postfix(exp=exp_id, status="build_failed")
                    pbar.update(1)
                    exp_durations.append(time.perf_counter() - exp_t0)
                    raise
            graph_cache[build_sig] = graph_path

        exp_cfg_path = configs_dir / f"{exp_id}.toml"
        _write_config(exp_cfg_path, cfg_exp, sections=["build_graphs", "train", "benchmark"])
        print(f"[benchmark] {exp_id}")
        try:
            _run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "benchmark_training.py"),
                    "--config",
                    str(exp_cfg_path),
                    "--modes",
                    str(args.modes),
                ],
                cwd=ROOT,
            )
        except Exception as exc:
            if not args.continue_on_error:
                pbar.set_postfix(exp=exp_id, status="benchmark_failed")
                pbar.update(1)
                exp_durations.append(time.perf_counter() - exp_t0)
                raise
            all_rows.append(
                {
                    "ablation_id": exp_id,
                    "ablation_family": exp.get("family", ""),
                    "ablation_description": exp.get("description", ""),
                    "mode": "",
                    "status": "benchmark_failed",
                    "error": str(exc),
                }
            )
            pbar.set_postfix(exp=exp_id, status="benchmark_failed")
            pbar.update(1)
            exp_durations.append(time.perf_counter() - exp_t0)
            continue

        bench_rows = _read_csv_rows(Path(cfg_exp["benchmark"]["out_csv"]))
        for row in bench_rows:
            row["ablation_id"] = exp_id
            row["ablation_family"] = exp.get("family", "")
            row["ablation_description"] = exp.get("description", "")
            row["graph_signature"] = build_sig
            row["graph_path"] = str(graph_path)
            row["modes_requested"] = str(args.modes)
            row["status"] = "ok"
            all_rows.append(row)
        pbar.set_postfix(exp=exp_id, status="ok")
        pbar.update(1)
        exp_durations.append(time.perf_counter() - exp_t0)

    pbar.close()
    if exp_durations:
        avg_exp = sum(exp_durations) / len(exp_durations)
        print(f"Avg wall time per ablation: {avg_exp:.1f}s")

    plan_out = metrics_dir / "recovery_ablation_plan.csv"
    if plan_rows:
        keys = sorted({k for r in plan_rows for k in r.keys()})
        with plan_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in plan_rows:
                w.writerow(row)
    print(f"Wrote {plan_out}")

    if args.dry_run:
        print("Dry-run complete. No graph builds or benchmarks were executed.")
        return 0

    if not all_rows:
        print("No benchmark rows produced.")
        return 0

    _add_baseline_deltas(all_rows, baseline_id="baseline_updated")
    out_csv = metrics_dir / "recovery_ablation.csv"
    keys = sorted({k for r in all_rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"Wrote {out_csv}")

    summary_rows = _family_summary(all_rows)
    summary_csv = metrics_dir / "recovery_ablation_summary.csv"
    if summary_rows:
        skeys = sorted({k for r in summary_rows for k in r.keys()})
        with summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=skeys)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
