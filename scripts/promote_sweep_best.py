#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
import tomllib


def _load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    s = str(value).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _format_value(val):
    if isinstance(val, str):
        return f"\"{val}\""
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def _backup_config(path: Path, suffix: str) -> None:
    ts = time.strftime("%Y%m%d%H%M%S")
    backup = Path(f"{path}{suffix}.{ts}")
    backup.write_text(path.read_text())
    print(f"Wrote backup {backup}")


def _apply_to_config(path: Path, section: str, updates: dict) -> None:
    lines = path.read_text().splitlines(keepends=True)
    header = f"[{section}]"
    start = None
    end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if stripped == header:
                start = i
            elif start is not None:
                end = i
                break
    if start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        start = len(lines)
        lines.append(f"{header}\n")
        end = len(lines)
    if end is None:
        end = len(lines)

    key_to_idx = {}
    for i in range(start + 1, end):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in line:
            key = line.split("=", 1)[0].strip()
            key_to_idx[key] = i

    insert_at = end
    for key, val in updates.items():
        new_line = f"{key} = {_format_value(val)}\n"
        if key in key_to_idx:
            lines[key_to_idx[key]] = new_line
        else:
            lines.insert(insert_at, new_line)
            insert_at += 1
            end += 1

    path.write_text("".join(lines))


def _pick_rank_column(rows: list[dict], rank_by: str) -> str:
    if rank_by != "auto":
        return rank_by
    if any(_to_float(r.get("primary_eval_metric_robust")) is not None for r in rows):
        return "primary_eval_metric_robust"
    if any(_to_float(r.get("rank_value")) is not None for r in rows):
        return "rank_value"
    objectives = [str(r.get("eval_objective", "")).strip().lower() for r in rows]
    has_sc = any(obj == "self_contrastive" for obj in objectives)
    has_ff = any(obj == "ff" for obj in objectives)
    if has_sc and not has_ff:
        if any(_to_float(r.get("eval_sc_gap")) is not None for r in rows):
            return "eval_sc_gap"
    for key in ("score", "eval_sep", "eval_acc", "graphs_per_s"):
        if any(_to_float(r.get(key)) is not None for r in rows):
            return key
    raise ValueError(
        "No numeric ranking column found. Expected primary_eval_metric_robust/"
        "rank_value/eval_sc_gap/score/eval_sep/eval_acc/graphs_per_s."
    )


def _read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _coerce_with_type(value: str, cast):
    if cast is str:
        return value.strip()
    if cast is int:
        fv = _to_float(value)
        if fv is None:
            raise ValueError(f"Cannot parse integer value from {value!r}")
        return int(round(fv))
    if cast is float:
        fv = _to_float(value)
        if fv is None:
            raise ValueError(f"Cannot parse float value from {value!r}")
        return float(fv)
    raise TypeError(f"Unsupported cast type: {cast}")


def _build_updates(best: dict, apply_mode: bool) -> dict:
    mapping = {
        "goodness_target": ("goodness_target", float),
        "goodness_temp": ("goodness_temp", float),
        "neg_mode": ("neg_mode", str),
        "noise_std": ("noise_std", float),
        "neg_mix_start": ("neg_mix_start", float),
        "neg_mix_end": ("neg_mix_end", float),
        "neg_mix_ramp_epochs": ("neg_mix_ramp_epochs", int),
        "neg_gate_margin": ("neg_gate_margin", float),
        "batch_size": ("batch_size", int),
        "lr": ("lr", float),
        "hidden_dim": ("hidden_dim", int),
        "num_layers": ("num_layers", int),
        "dropout": ("dropout", float),
        "hall_steps": ("hallucinate_steps", int),
        "hall_lr": ("hallucinate_lr", float),
        "hall_l2": ("hallucinate_l2", float),
        "hall_mean": ("hallucinate_mean", float),
        "hall_std": ("hallucinate_std", float),
        "hall_corr": ("hallucinate_corr", float),
        "hall_clamp": ("hallucinate_clamp_std", float),
        "hall_node_fraction": ("hallucinate_node_fraction", float),
        "hall_node_min": ("hallucinate_node_min", int),
        "layerwise_neg_mode": ("layerwise_neg_mode", str),
        "layerwise_noise_std": ("layerwise_noise_std", float),
        "layerwise_hall_corr": ("layerwise_hall_corr", float),
        "layerwise_hall_mean": ("layerwise_hall_mean", float),
        "layerwise_hall_std": ("layerwise_hall_std", float),
    }
    updates = {}
    for src_key, (dst_key, cast) in mapping.items():
        raw = best.get(src_key, "")
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        updates[dst_key] = _coerce_with_type(s, cast)

    if apply_mode:
        mode = str(best.get("mode", "")).strip().lower()
        if mode == "ff_layerwise":
            updates["ff_layerwise"] = True
        elif mode in ("ff_e2e", "backprop"):
            updates["ff_layerwise"] = False

    return updates


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote best sweep row into a train config section.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument(
        "--csv",
        default="",
        help="Sweep CSV path (defaults to [sweep].out_csv or runs/experiments/manual/metrics/ff_sweep.csv)",
    )
    parser.add_argument(
        "--rank-by",
        choices=[
            "auto",
            "primary_eval_metric_robust",
            "rank_value",
            "score",
            "eval_sc_gap",
            "eval_sep",
            "eval_acc",
            "graphs_per_s",
        ],
        default="auto",
        help="Column to maximize when selecting best row",
    )
    parser.add_argument("--mode", default="", help="Optional mode filter (e.g., ff_e2e)")
    parser.add_argument("--top-k", type=int, default=5, help="Print top-k rows")
    parser.add_argument("--apply", action="store_true", help="Apply updates to config")
    parser.add_argument("--apply-to", default="", help="Config path to update (default: --config)")
    parser.add_argument("--apply-section", default="train", help="Config section to update")
    parser.add_argument("--apply-mode", dest="apply_mode", action="store_true", default=True)
    parser.add_argument("--no-apply-mode", dest="apply_mode", action="store_false")
    parser.add_argument("--backup", action="store_true", help="Backup config before writing")
    parser.add_argument("--backup-suffix", default=".bak", help="Backup suffix")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    sweep_cfg = cfg.get("sweep", {})
    csv_path = Path(
        args.csv or sweep_cfg.get("out_csv", "runs/experiments/manual/metrics/ff_sweep.csv")
    )
    rows = _read_rows(csv_path)

    mode_filter = args.mode.strip()
    if mode_filter:
        rows = [r for r in rows if str(r.get("mode", "")).strip() == mode_filter]
    if not rows:
        raise ValueError(f"No sweep rows remain after mode filter: {mode_filter!r}")

    rank_col = _pick_rank_column(rows, args.rank_by)
    ranked = sorted(rows, key=lambda r: _to_float(r.get(rank_col)) or float("-inf"), reverse=True)
    best = ranked[0]
    best_rank = _to_float(best.get(rank_col))
    if best_rank is None:
        raise ValueError(f"Best row has non-numeric rank column {rank_col!r}")

    print(f"Sweep CSV: {csv_path}")
    print(f"Rank column: {rank_col}")
    print(f"Best row: mode={best.get('mode', '')} {rank_col}={best_rank:.6f}")
    print("Top rows:")
    for row in ranked[: max(1, args.top_k)]:
        print(
            {
                "mode": row.get("mode", ""),
                rank_col: _to_float(row.get(rank_col)),
                "rank_metric": row.get("rank_metric"),
                "rank_value": _to_float(row.get("rank_value")),
                "primary_eval_metric_robust": _to_float(row.get("primary_eval_metric_robust")),
                "eval_sc_gap": _to_float(row.get("eval_sc_gap")),
                "eval_sep": _to_float(row.get("eval_sep")),
                "eval_acc": _to_float(row.get("eval_acc")),
                "graphs_per_s": _to_float(row.get("graphs_per_s")),
                "score": _to_float(row.get("score")),
            }
        )

    updates = _build_updates(best, apply_mode=args.apply_mode)
    print("Proposed updates:")
    for k in sorted(updates):
        print(f"  {k} = {updates[k]}")

    if not args.apply:
        print("Dry run only (use --apply to write changes).")
        return 0

    apply_to = Path(args.apply_to) if args.apply_to else cfg_path
    if args.backup:
        _backup_config(apply_to, args.backup_suffix)
    _apply_to_config(apply_to, args.apply_section, updates)
    print(f"Applied {len(updates)} updates to {apply_to} [{args.apply_section}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
