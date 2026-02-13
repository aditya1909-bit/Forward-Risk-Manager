#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


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


def _build_temp_config(
    cfg: dict,
    section: str,
    candidate: dict,
    out_csv: Path,
) -> str:
    build_cfg = dict(cfg.get("build_graphs", {}))
    train_cfg = dict(cfg.get("train", {}))
    sweep_cfg = dict(cfg.get(section, {}))
    sweep_cfg.update(candidate)
    sweep_cfg["out_csv"] = str(out_csv)
    if "modes" not in sweep_cfg:
        sweep_cfg["modes"] = ["ff_e2e"]
    lines: list[str] = []
    _write_section(lines, "build_graphs", build_cfg)
    _write_section(lines, "train", train_cfg)
    _write_section(lines, section, sweep_cfg)
    return "\n".join(lines)


def _best_score_from_csv(path: Path) -> tuple[float, dict]:
    best_score = float("-inf")
    best_row: dict[str, str] = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                score = float(row.get("score", "nan"))
            except Exception:
                continue
            if not math.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_row = row
    return best_score, best_row


def _run_trial(cfg: dict, section: str, candidate: dict) -> tuple[float, dict]:
    with tempfile.TemporaryDirectory(prefix="ff_optuna_") as td:
        td_path = Path(td)
        out_csv = td_path / "sweep.csv"
        temp_cfg = td_path / "config.toml"
        temp_cfg.write_text(_build_temp_config(cfg, section, candidate, out_csv))
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "ff_sweep.py"),
            "--config",
            str(temp_cfg),
            "--section",
            section,
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        if not out_csv.exists():
            return float("-inf"), {}
        return _best_score_from_csv(out_csv)


def _sample_candidate(rng: random.Random) -> dict:
    return {
        "goodness_temp": rng.uniform(0.1, 0.6),
        "goodness_target": rng.uniform(0.8, 3.5),
        "neg_mix_end": rng.uniform(0.2, 0.9),
        "hall_steps": rng.randint(1, 10),
        "hall_lr": rng.uniform(0.01, 0.08),
        "hall_node_fraction": rng.uniform(0.1, 0.8),
        "ff_margin": rng.uniform(0.0, 0.5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bayesian (Optuna) search wrapper for ff_sweep.py with fallback random search."
    )
    parser.add_argument("--config", required=True, help="Path to project TOML config")
    parser.add_argument("--section", default="sweep", help="Sweep section to optimize")
    parser.add_argument("--trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-json",
        default="runs/experiments/default/metrics/ff_optuna_best.json",
        help="Where to store best parameters and summary.",
    )
    args = parser.parse_args()

    with Path(args.config).open("rb") as f:
        cfg = tomllib.load(f)

    rng = random.Random(args.seed)
    best_score = float("-inf")
    best_params: dict = {}
    best_row: dict = {}
    trials: list[dict] = []

    try:
        import optuna  # type: ignore

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial):
            candidate = {
                "goodness_temp": trial.suggest_float("goodness_temp", 0.1, 0.6),
                "goodness_target": trial.suggest_float("goodness_target", 0.8, 3.5),
                "neg_mix_end": trial.suggest_float("neg_mix_end", 0.2, 0.9),
                "hall_steps": trial.suggest_int("hall_steps", 1, 10),
                "hall_lr": trial.suggest_float("hall_lr", 0.01, 0.08),
                "hall_node_fraction": trial.suggest_float("hall_node_fraction", 0.1, 0.8),
                "ff_margin": trial.suggest_float("ff_margin", 0.0, 0.5),
            }
            score, row = _run_trial(cfg, args.section, candidate)
            trials.append({"candidate": candidate, "score": score})
            if row:
                trial.set_user_attr("row", row)
            return score

        study.optimize(_objective, n_trials=max(1, int(args.trials)))
        best_score = float(study.best_value)
        best_params = dict(study.best_params)
        best_row = dict(study.best_trial.user_attrs.get("row", {}))
        search_mode = "optuna"
    except Exception:
        for _ in range(max(1, int(args.trials))):
            candidate = _sample_candidate(rng)
            score, row = _run_trial(cfg, args.section, candidate)
            trials.append({"candidate": candidate, "score": score})
            if score > best_score:
                best_score = score
                best_params = candidate
                best_row = row
        search_mode = "random_fallback"

    out = {
        "search_mode": search_mode,
        "config": args.config,
        "section": args.section,
        "trials": int(args.trials),
        "best_score": best_score,
        "best_params": best_params,
        "best_row": best_row,
        "history": trials,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(f"best_score={best_score:.6f} best_params={best_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
