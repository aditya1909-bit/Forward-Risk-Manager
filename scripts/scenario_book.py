#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import replace
import random
import sys
import tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import (
    CompositeEnergyCritic,
    EnergyCritic,
    EnergyCriticEnsemble,
    GCNEncoder,
    SequenceEnergyCritic,
)
from frisk.hallucinate import HallucinationConfig, hallucinate_negative
from frisk.graph_artifact import load_graph_artifact


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def _load_state_dict_compat(path: str):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if isinstance(state.get("model"), dict):
            state = state["model"]
    if isinstance(state, dict):
        # torch.compile() and DataParallel can prefix keys with wrappers.
        out = {}
        for k, v in state.items():
            kk = str(k)
            for prefix in ("_orig_mod.", "module."):
                if kk.startswith(prefix):
                    kk = kk[len(prefix) :]
            out[kk] = v
        return out
    return state


def _build_critic(config: dict, hidden_dim: int, device: torch.device):
    critic_hidden_dim = max(1, int(config.get("critic_hidden_dim", hidden_dim)))
    critic_num_layers = max(1, int(config.get("critic_num_layers", 2)))
    critic_dropout = max(0.0, float(config.get("critic_dropout", config.get("dropout", 0.1))))
    critic_positive = str(config.get("critic_positive_activation", "softplus")).strip().lower()
    if critic_positive not in {"softplus", "square"}:
        critic_positive = "softplus"

    ensemble_size = max(1, int(config.get("critic_ensemble_size", 1)))
    seed_base = int(config.get("seed", 7))
    seed_stride = max(1, int(config.get("critic_ensemble_seed_stride", 1009)))
    critics = []
    for i in range(ensemble_size):
        if ensemble_size > 1:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed_base + i * seed_stride)
                member = EnergyCritic(
                    in_dim=hidden_dim,
                    hidden_dim=critic_hidden_dim,
                    num_layers=critic_num_layers,
                    dropout=critic_dropout,
                    positive_activation=critic_positive,
                )
        else:
            member = EnergyCritic(
                in_dim=hidden_dim,
                hidden_dim=critic_hidden_dim,
                num_layers=critic_num_layers,
                dropout=critic_dropout,
                positive_activation=critic_positive,
            )
        critics.append(member.to(device))

    if len(critics) == 1:
        base_critic = critics[0]
    else:
        base_critic = EnergyCriticEnsemble(critics=critics).to(device)

    seq_enabled = bool(config.get("sequence_critic_enabled", False))
    if not seq_enabled:
        return base_critic

    seq_hidden = max(1, int(config.get("sequence_critic_hidden_dim", hidden_dim)))
    seq_layers = max(1, int(config.get("sequence_critic_num_layers", 1)))
    seq_dropout = max(0.0, float(config.get("sequence_critic_dropout", 0.0)))
    seq_positive = str(config.get("sequence_critic_positive_activation", "softplus")).strip().lower()
    if seq_positive not in {"softplus", "square"}:
        seq_positive = "softplus"
    seq_weight = float(config.get("sequence_critic_weight", 0.0))
    seq_critic = SequenceEnergyCritic(
        in_dim=hidden_dim,
        hidden_dim=seq_hidden,
        num_layers=seq_layers,
        dropout=seq_dropout,
        positive_activation=seq_positive,
    ).to(device)
    return CompositeEnergyCritic(
        base_critic=base_critic,
        sequence_critic=seq_critic,
        sequence_weight=seq_weight,
    ).to(device)


def _critic_structure_summary(config: dict, hidden_dim: int) -> str:
    critic_hidden_dim = max(1, int(config.get("critic_hidden_dim", hidden_dim)))
    critic_num_layers = max(1, int(config.get("critic_num_layers", 2)))
    critic_dropout = max(0.0, float(config.get("critic_dropout", config.get("dropout", 0.1))))
    critic_positive = str(config.get("critic_positive_activation", "softplus")).strip().lower()
    ensemble_size = max(1, int(config.get("critic_ensemble_size", 1)))
    seq_enabled = bool(config.get("sequence_critic_enabled", False))
    seq_hidden = max(1, int(config.get("sequence_critic_hidden_dim", hidden_dim)))
    seq_layers = max(1, int(config.get("sequence_critic_num_layers", 1)))
    seq_dropout = max(0.0, float(config.get("sequence_critic_dropout", 0.0)))
    seq_positive = str(config.get("sequence_critic_positive_activation", "softplus")).strip().lower()
    seq_weight = float(config.get("sequence_critic_weight", 0.0))
    return (
        f"hidden_dim={hidden_dim}, critic_hidden_dim={critic_hidden_dim}, "
        f"critic_num_layers={critic_num_layers}, critic_dropout={critic_dropout}, "
        f"critic_positive_activation={critic_positive}, critic_ensemble_size={ensemble_size}, "
        f"sequence_critic_enabled={int(seq_enabled)}, sequence_critic_hidden_dim={seq_hidden}, "
        f"sequence_critic_num_layers={seq_layers}, sequence_critic_dropout={seq_dropout}, "
        f"sequence_critic_positive_activation={seq_positive}, sequence_critic_weight={seq_weight}"
    )


def _load_critic_checkpoint_or_raise(
    critic: torch.nn.Module,
    checkpoint_path: str,
    config: dict,
    hidden_dim: int,
) -> None:
    state = _load_state_dict_compat(checkpoint_path)
    try:
        critic.load_state_dict(state)
    except RuntimeError as exc:
        expected = _critic_structure_summary(config, hidden_dim=hidden_dim)
        raise RuntimeError(
            "Critic checkpoint load mismatch in scenario_book. "
            f"checkpoint={checkpoint_path}. "
            f"expected={expected}. "
            "Check that train.critic_* and train.sequence_critic_* settings match the checkpoint. "
            f"original_error={exc}"
        ) from exc


def _select_indices(dates, num_scenarios, seed, indices, date_list):
    if indices:
        return indices
    if date_list:
        idxs = []
        for d in date_list:
            if d in dates:
                idxs.append(dates.index(d))
            else:
                raise ValueError(f"Date {d} not found in graphs metadata.")
        return idxs
    rng = random.Random(seed)
    idxs = list(range(len(dates)))
    rng.shuffle(idxs)
    return idxs[:num_scenarios]


def _sample_ticker_preview(tickers_list, max_items=20):
    if not tickers_list:
        return ""
    uniq = sorted({t for sub in tickers_list for t in sub})
    if not uniq:
        return ""
    preview = uniq[:max_items]
    suffix = "..." if len(uniq) > max_items else ""
    return ", ".join(preview) + suffix


def _resolve_auto_target_ticker(tickers_list):
    counts: dict[str, int] = {}
    for tickers in tickers_list or []:
        for t in tickers:
            tt = str(t).upper().strip()
            if not tt:
                continue
            counts[tt] = counts.get(tt, 0) + 1
    if not counts:
        return ""
    # Pick the most frequently available ticker across windows.
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _constraint_diff(
    cum_return: torch.Tensor,
    target_drop: float,
    mode: str,
    tolerance: float = 0.0,
) -> torch.Tensor:
    tol = max(0.0, float(tolerance))
    if mode == "exact":
        diff = cum_return - float(target_drop)
        if tol > 0:
            diff = torch.sign(diff) * torch.relu(torch.abs(diff) - tol)
        return diff
    if mode == "at_least":
        if target_drop < 0:
            # For downside targets, allow returns <= target+tol.
            return torch.relu(cum_return - (float(target_drop) + tol))
        # For upside targets, allow returns >= target-tol.
        return torch.relu((float(target_drop) - tol) - cum_return)
    raise ValueError(f"Unknown constraint mode: {mode}")


def _constraint_hit(
    hall_minus_target: float,
    target_drop: float,
    mode: str,
    tolerance: float,
) -> bool:
    tol = max(0.0, float(tolerance))
    d = float(hall_minus_target)
    if mode == "exact":
        return abs(d) <= tol
    if target_drop < 0:
        return d <= tol
    return d >= -tol


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a scenario book of hallucinated windows.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument(
        "--critic-model",
        default="",
        help="Path to critic checkpoint (defaults to train.save_critic / train.critic_checkpoint_in).",
    )
    parser.add_argument("--num-scenarios", type=int, default=10, help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--indices", default="", help="Comma-separated graph indices")
    parser.add_argument("--dates", default="", help="Comma-separated graph dates")
    parser.add_argument("--target-ticker", default="", help="Ticker to constrain (e.g., NVDA)")
    parser.add_argument(
        "--target-drop",
        type=float,
        default=0.0,
        help="Target cumulative return over window (e.g., -0.10 for -10%%)",
    )
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=10.0,
        help="Penalty weight for the constraint",
    )
    parser.add_argument(
        "--constraint-tolerance",
        type=float,
        default=0.01,
        help="Tolerance band for constraint objective (used by exact mode).",
    )
    parser.add_argument(
        "--hall-steps",
        type=int,
        default=None,
        help="Override hallucination steps",
    )
    parser.add_argument(
        "--hall-lr",
        type=float,
        default=None,
        help="Override hallucination learning rate",
    )
    parser.add_argument(
        "--hall-l2",
        type=float,
        default=None,
        help="Override hallucination L2 weight",
    )
    parser.add_argument(
        "--hall-corr",
        type=float,
        default=None,
        help="Override hallucination correlation weight",
    )
    parser.add_argument(
        "--hall-mean-weight",
        type=float,
        default=None,
        help="Override hallucination mean penalty weight",
    )
    parser.add_argument(
        "--hall-std-weight",
        type=float,
        default=None,
        help="Override hallucination std penalty weight",
    )
    parser.add_argument(
        "--hall-clamp-std",
        type=float,
        default=None,
        help="Override hallucination clamp std (set higher to allow larger moves)",
    )
    parser.add_argument(
        "--hall-node-fraction",
        type=float,
        default=None,
        help="Override hallucination node fraction",
    )
    parser.add_argument(
        "--hall-init-noise",
        type=float,
        default=None,
        help="Override hallucination init noise scale",
    )
    parser.add_argument(
        "--min-hall-delta",
        type=float,
        default=0.0,
        help="Minimum mean abs delta between real and hallucinated returns (0 disables)",
    )
    parser.add_argument(
        "--hall-retry-limit",
        type=int,
        default=2,
        help="Max retries to regenerate hallucination if delta too small",
    )
    parser.add_argument(
        "--hall-retry-lr-mult",
        type=float,
        default=1.5,
        help="LR multiplier per retry when delta too small",
    )
    parser.add_argument(
        "--hall-retry-steps-inc",
        type=int,
        default=2,
        help="Step increment per retry when delta too small",
    )
    parser.add_argument(
        "--hall-retry-clamp-inc",
        type=float,
        default=0.5,
        help="Clamp std increment per retry when delta too small",
    )
    parser.add_argument(
        "--hall-retry-node-inc",
        type=float,
        default=0.1,
        help="Node fraction increment per retry when delta too small",
    )
    parser.add_argument(
        "--hall-adaptive-lr",
        action="store_true",
        help="Enable adaptive hallucination learning-rate decay on plateaus.",
    )
    parser.add_argument(
        "--hall-adaptive-lr-patience",
        type=int,
        default=2,
        help="Plateau patience (steps) before reducing hallucination LR.",
    )
    parser.add_argument(
        "--hall-adaptive-lr-decay",
        type=float,
        default=0.5,
        help="LR decay factor applied when adaptive LR triggers.",
    )
    parser.add_argument(
        "--hall-adaptive-lr-min",
        type=float,
        default=1e-4,
        help="Minimum LR when adaptive LR is enabled.",
    )
    parser.add_argument(
        "--early-stop-on-target-hit",
        action="store_true",
        help="Stop hallucination early after consecutive target hits.",
    )
    parser.add_argument(
        "--target-hit-patience",
        type=int,
        default=1,
        help="Consecutive hit steps required for early-stop.",
    )
    parser.add_argument(
        "--hall-moment-mean",
        type=float,
        default=0.0,
        help="Moment-matching mean penalty weight for hallucination.",
    )
    parser.add_argument(
        "--hall-moment-var",
        type=float,
        default=0.0,
        help="Moment-matching variance penalty weight for hallucination.",
    )
    parser.add_argument(
        "--hall-moment-skew",
        type=float,
        default=0.0,
        help="Moment-matching skewness penalty weight for hallucination.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Adapt hallucination hyperparameters until constraint hit rate is met",
    )
    parser.add_argument(
        "--target-hit-rate",
        type=float,
        default=0.6,
        help="Target hit rate for adaptive constraint tuning",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=0.01,
        help="Tolerance for constraint hit (absolute)",
    )
    parser.add_argument(
        "--nontarget-drift-weight",
        type=float,
        default=0.0,
        help="Penalty weight to keep non-target tickers near original returns.",
    )
    parser.add_argument(
        "--nontarget-drift-tolerance",
        type=float,
        default=0.0,
        help="Free band for non-target drift before penalty is applied.",
    )
    parser.add_argument(
        "--max-nontarget-drift",
        type=float,
        default=0.03,
        help="Adaptive target for mean non-target absolute return drift.",
    )
    parser.add_argument(
        "--max-adapt-steps",
        type=int,
        default=6,
        help="Maximum adaptive iterations",
    )
    parser.add_argument(
        "--adapt-constraint-mult",
        type=float,
        default=1.5,
        help="Multiplier for constraint weight when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-step-inc",
        type=int,
        default=2,
        help="Step increment when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-lr-mult",
        type=float,
        default=1.2,
        help="LR multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-l2-mult",
        type=float,
        default=0.8,
        help="L2 multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-mean-mult",
        type=float,
        default=0.8,
        help="Mean penalty multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-std-mult",
        type=float,
        default=0.8,
        help="Std penalty multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-corr-mult",
        type=float,
        default=0.8,
        help="Corr penalty multiplier when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-node-inc",
        type=float,
        default=0.1,
        help="Node fraction increment when under-hitting",
    )
    parser.add_argument(
        "--adapt-hall-clamp-inc",
        type=float,
        default=0.5,
        help="Clamp std increment when under-hitting",
    )
    parser.add_argument(
        "--adapt-max-constraint",
        type=float,
        default=200.0,
        help="Max constraint weight in adaptive mode",
    )
    parser.add_argument(
        "--adapt-max-steps",
        type=int,
        default=20,
        help="Max hallucination steps in adaptive mode",
    )
    parser.add_argument(
        "--adapt-max-lr",
        type=float,
        default=0.2,
        help="Max hallucination LR in adaptive mode",
    )
    parser.add_argument(
        "--adapt-max-clamp-std",
        type=float,
        default=8.0,
        help="Max clamp std in adaptive mode",
    )
    parser.add_argument(
        "--adapt-min-l2",
        type=float,
        default=0.005,
        help="Min hallucination L2 in adaptive mode",
    )
    parser.add_argument(
        "--adapt-min-mean",
        type=float,
        default=0.001,
        help="Min hallucination mean penalty in adaptive mode",
    )
    parser.add_argument(
        "--adapt-min-std",
        type=float,
        default=0.001,
        help="Min hallucination std penalty in adaptive mode",
    )
    parser.add_argument(
        "--adapt-min-corr",
        type=float,
        default=0.01,
        help="Min hallucination corr penalty in adaptive mode",
    )
    parser.add_argument(
        "--adapt-nontarget-mult",
        type=float,
        default=1.4,
        help="Multiplier for non-target drift weight when drift is too high.",
    )
    parser.add_argument(
        "--adapt-max-nontarget-weight",
        type=float,
        default=300.0,
        help="Cap for non-target drift penalty weight in adaptive mode.",
    )
    parser.add_argument(
        "--adapt-nontarget-reg-mult",
        type=float,
        default=1.15,
        help="Regularization multiplier when non-target drift is too high.",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=["exact", "at_least"],
        default="exact",
        help="Exact match or at-least constraint",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Limit tickers per scenario (0 = all)",
    )
    parser.add_argument(
        "--diag-out",
        default="",
        help="Optional CSV to record constraint diagnostics",
    )
    parser.add_argument(
        "--out",
        default="runs/experiments/manual/metrics/scenario_book.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    if args.target_ticker:
        args.target_ticker = args.target_ticker.strip().upper()
    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})
    build_cfg = cfg.get("build_graphs", {})
    scenario_cfg = cfg.get("scenario_book", {})

    def _maybe_override(key: str, flags, cast=None):
        flag_list = flags if isinstance(flags, (list, tuple)) else [flags]
        if any(flag in sys.argv for flag in flag_list):
            return
        if key not in scenario_cfg:
            return
        val = scenario_cfg[key]
        if isinstance(val, list):
            val = ",".join(str(x) for x in val)
        if cast:
            if cast is bool:
                if isinstance(val, str):
                    val = val.strip().lower() in ("1", "true", "yes", "y", "on")
                else:
                    val = bool(val)
            else:
                val = cast(val)
        setattr(args, key, val)

    _maybe_override("num_scenarios", "--num-scenarios", int)
    _maybe_override("seed", "--seed", int)
    _maybe_override("indices", "--indices")
    _maybe_override("dates", "--dates")
    _maybe_override("target_ticker", "--target-ticker")
    _maybe_override("target_drop", "--target-drop", float)
    _maybe_override("constraint_weight", "--constraint-weight", float)
    _maybe_override("constraint_tolerance", "--constraint-tolerance", float)
    _maybe_override("constraint_mode", "--constraint-mode")
    _maybe_override("max_tickers", "--max-tickers", int)
    _maybe_override("diag_out", "--diag-out")
    _maybe_override("out", "--out")
    _maybe_override("critic_model", "--critic-model")
    _maybe_override("adaptive", "--adaptive", bool)
    _maybe_override("target_hit_rate", "--target-hit-rate", float)
    _maybe_override("target_tolerance", "--target-tolerance", float)
    _maybe_override("nontarget_drift_weight", "--nontarget-drift-weight", float)
    _maybe_override("nontarget_drift_tolerance", "--nontarget-drift-tolerance", float)
    _maybe_override("max_nontarget_drift", "--max-nontarget-drift", float)
    _maybe_override("max_adapt_steps", "--max-adapt-steps", int)
    _maybe_override("adapt_constraint_mult", "--adapt-constraint-mult", float)
    _maybe_override("adapt_hall_step_inc", "--adapt-hall-step-inc", int)
    _maybe_override("adapt_hall_lr_mult", "--adapt-hall-lr-mult", float)
    _maybe_override("adapt_hall_l2_mult", "--adapt-hall-l2-mult", float)
    _maybe_override("adapt_hall_mean_mult", "--adapt-hall-mean-mult", float)
    _maybe_override("adapt_hall_std_mult", "--adapt-hall-std-mult", float)
    _maybe_override("adapt_hall_corr_mult", "--adapt-hall-corr-mult", float)
    _maybe_override("adapt_hall_node_inc", "--adapt-hall-node-inc", float)
    _maybe_override("adapt_hall_clamp_inc", "--adapt-hall-clamp-inc", float)
    _maybe_override("adapt_max_constraint", "--adapt-max-constraint", float)
    _maybe_override("adapt_max_steps", "--adapt-max-steps", int)
    _maybe_override("adapt_max_lr", "--adapt-max-lr", float)
    _maybe_override("adapt_max_clamp_std", "--adapt-max-clamp-std", float)
    _maybe_override("adapt_min_l2", "--adapt-min-l2", float)
    _maybe_override("adapt_min_mean", "--adapt-min-mean", float)
    _maybe_override("adapt_min_std", "--adapt-min-std", float)
    _maybe_override("adapt_min_corr", "--adapt-min-corr", float)
    _maybe_override("adapt_nontarget_mult", "--adapt-nontarget-mult", float)
    _maybe_override("adapt_max_nontarget_weight", "--adapt-max-nontarget-weight", float)
    _maybe_override("adapt_nontarget_reg_mult", "--adapt-nontarget-reg-mult", float)

    _maybe_override("hall_steps", "--hall-steps", int)
    _maybe_override("hall_lr", "--hall-lr", float)
    _maybe_override("hall_l2", "--hall-l2", float)
    _maybe_override("hall_corr", "--hall-corr", float)
    _maybe_override("hall_mean_weight", "--hall-mean-weight", float)
    _maybe_override("hall_std_weight", "--hall-std-weight", float)
    _maybe_override("hall_clamp_std", "--hall-clamp-std", float)
    _maybe_override("hall_node_fraction", "--hall-node-fraction", float)
    _maybe_override("hall_init_noise", "--hall-init-noise", float)
    _maybe_override("min_hall_delta", "--min-hall-delta", float)
    _maybe_override("hall_retry_limit", "--hall-retry-limit", int)
    _maybe_override("hall_retry_lr_mult", "--hall-retry-lr-mult", float)
    _maybe_override("hall_retry_steps_inc", "--hall-retry-steps-inc", int)
    _maybe_override("hall_retry_clamp_inc", "--hall-retry-clamp-inc", float)
    _maybe_override("hall_retry_node_inc", "--hall-retry-node-inc", float)
    _maybe_override("hall_adaptive_lr", "--hall-adaptive-lr", bool)
    _maybe_override("hall_adaptive_lr_patience", "--hall-adaptive-lr-patience", int)
    _maybe_override("hall_adaptive_lr_decay", "--hall-adaptive-lr-decay", float)
    _maybe_override("hall_adaptive_lr_min", "--hall-adaptive-lr-min", float)
    _maybe_override("early_stop_on_target_hit", "--early-stop-on-target-hit", bool)
    _maybe_override("target_hit_patience", "--target-hit-patience", int)
    _maybe_override("hall_moment_mean", "--hall-moment-mean", float)
    _maybe_override("hall_moment_var", "--hall-moment-var", float)
    _maybe_override("hall_moment_skew", "--hall-moment-skew", float)

    if args.target_ticker:
        args.target_ticker = args.target_ticker.strip().upper()
    if args.adaptive and not args.target_ticker:
        raise ValueError("--adaptive requires --target-ticker")

    graphs_path = Path(train_cfg.get("graphs", "data/processed/graphs.pt"))
    artifact = load_graph_artifact(graphs_path, include_tickers=True, prefer_lazy=True, prefer_sharded=True)
    graphs = artifact.graphs
    tickers_list = artifact.tickers or []
    dates = artifact.dates
    if not graphs:
        raise ValueError("No graphs found.")
    print(f"graph artifact: {artifact.path} (format={artifact.format})")

    if args.target_ticker in {"AUTO", "AUTO_DETECT", "AUTO-DETECT"}:
        resolved = _resolve_auto_target_ticker(tickers_list)
        if not resolved:
            raise ValueError("target_ticker=AUTO but no ticker metadata found in graphs.")
        args.target_ticker = resolved
        print(f"scenario target_ticker auto-resolved to {args.target_ticker}")

    indices = [int(x) for x in args.indices.split(",") if x.strip()] if args.indices else []
    date_list = [d.strip() for d in args.dates.split(",") if d.strip()] if args.dates else []

    target_candidates = None
    if args.target_ticker:
        if not tickers_list:
            raise ValueError("Graph payload missing tickers metadata; rebuild graphs with build_graphs.py.")
        target_candidates = [
            i for i, tickers in enumerate(tickers_list) if args.target_ticker in tickers
        ]
        if not target_candidates:
            preview = _sample_ticker_preview(tickers_list)
            hint = f" Sample tickers: {preview}" if preview else ""
            raise ValueError(
                f"Target ticker {args.target_ticker} not found in any graph.{hint} "
                "If you expected it, rebuild graphs with its price data and include it at build time."
            )

    if indices or date_list:
        idxs = _select_indices(dates, args.num_scenarios, args.seed, indices, date_list)
        if target_candidates is not None:
            missing = [i for i in idxs if i not in target_candidates]
            if missing:
                missing_dates = [dates[i] if i < len(dates) else "n/a" for i in missing]
                raise ValueError(
                    "Target ticker missing for some selected graphs. "
                    f"Missing indices: {missing} | dates: {missing_dates}"
                )
    else:
        if target_candidates is not None:
            rng = random.Random(args.seed)
            idxs = list(target_candidates)
            rng.shuffle(idxs)
            if args.num_scenarios <= 0:
                raise ValueError("--num-scenarios must be >= 1")
            if len(idxs) < args.num_scenarios:
                print(
                    f"Requested {args.num_scenarios} scenarios but only {len(idxs)} "
                    f"graphs include {args.target_ticker}. Using {len(idxs)}."
                )
            else:
                idxs = idxs[: args.num_scenarios]
        else:
            idxs = _select_indices(dates, args.num_scenarios, args.seed, indices, date_list)

    window = int(build_cfg.get("window", 20))
    feature_mode = build_cfg.get("feature_mode", "window")
    returns_len = window if feature_mode in ("window", "window_plus_summary", "window_plus_summary_fund") else 1

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=int(train_cfg.get("hidden_dim", 64)),
        num_layers=int(train_cfg.get("num_layers", 2)),
        dropout=float(train_cfg.get("dropout", 0.1)),
        conv_type=str(train_cfg.get("encoder_conv_type", "gcn")).strip().lower(),
        gat_heads=int(train_cfg.get("encoder_gat_heads", 2)),
        rgcn_num_relations=max(2, int(train_cfg.get("encoder_rgcn_num_relations", 8))),
        residual_edge_enabled=bool(train_cfg.get("residual_edge_weight_enabled", False)),
        residual_edge_hidden_dim=int(train_cfg.get("residual_edge_hidden_dim", 32)),
        residual_edge_max_delta=float(train_cfg.get("residual_edge_max_delta", 0.25)),
        residual_edge_detach_features=bool(train_cfg.get("residual_edge_detach_features", True)),
    )
    model_path = (
        str(train_cfg.get("save_encoder", "")).strip()
        or str(train_cfg.get("save_model", "")).strip()
    )
    if not model_path:
        raise ValueError("No encoder checkpoint path provided in train.save_encoder/train.save_model.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {model_path}")
    model.load_state_dict(_load_state_dict_compat(model_path))
    model.eval()

    critic = None
    strict_split = bool(train_cfg.get("strict_component_split", False))
    critic_path = (
        str(args.critic_model).strip()
        or str(train_cfg.get("save_critic", "")).strip()
        or str(train_cfg.get("critic_checkpoint_in", "")).strip()
    )
    if critic_path:
        critic_hidden_dim = int(train_cfg.get("hidden_dim", 64))
        critic = _build_critic(
            train_cfg,
            hidden_dim=critic_hidden_dim,
            device=torch.device("cpu"),
        )
        cpath = Path(critic_path)
        if not cpath.exists():
            raise FileNotFoundError(f"Critic checkpoint not found: {cpath}")
        _load_critic_checkpoint_or_raise(
            critic=critic,
            checkpoint_path=str(cpath),
            config=train_cfg,
            hidden_dim=critic_hidden_dim,
        )
        critic.eval()
        critic_path = str(cpath)
    elif strict_split:
        raise ValueError(
            "strict_component_split is enabled, but no critic checkpoint was provided. "
            "Set --critic-model or train.save_critic."
        )
    else:
        critic_path = ""

    encoder_checkpoint = str(model_path).strip()
    critic_checkpoint = str(critic_path).strip()
    objective_track = "critic" if critic is not None else "encoder_proxy"
    energy_component = "critic_energy" if critic is not None else "encoder_squared"
    split_mode = "strict" if strict_split else "legacy"
    train_neg_mode = str(train_cfg.get("neg_mode", "")).strip().lower()

    hall_cfg = HallucinationConfig(
        steps=int(args.hall_steps)
        if args.hall_steps is not None
        else int(train_cfg.get("hallucinate_steps", 4)),
        lr=float(args.hall_lr)
        if args.hall_lr is not None
        else float(train_cfg.get("hallucinate_lr", 0.05)),
        l2_weight=float(args.hall_l2)
        if args.hall_l2 is not None
        else float(train_cfg.get("hallucinate_l2", 0.02)),
        mean_weight=float(args.hall_mean_weight)
        if args.hall_mean_weight is not None
        else float(train_cfg.get("hallucinate_mean", 0.01)),
        std_weight=float(args.hall_std_weight)
        if args.hall_std_weight is not None
        else float(train_cfg.get("hallucinate_std", 0.01)),
        corr_weight=float(args.hall_corr)
        if args.hall_corr is not None
        else float(train_cfg.get("hallucinate_corr", 0.2)),
        clamp_std=float(args.hall_clamp_std)
        if args.hall_clamp_std is not None
        else float(train_cfg.get("hallucinate_clamp_std", 3.0)),
        goodness_temp=float(train_cfg.get("goodness_temp", 1.0)),
        node_fraction=float(args.hall_node_fraction)
        if args.hall_node_fraction is not None
        else float(train_cfg.get("hallucinate_node_fraction", 1.0)),
        node_min=int(train_cfg.get("hallucinate_node_min", 1)),
        init_noise=float(args.hall_init_noise)
        if args.hall_init_noise is not None
        else float(train_cfg.get("hallucinate_init_noise", 0.0)),
        return_slice_len=returns_len,
        penalty_scope=str(train_cfg.get("hallucinate_penalty_scope", "returns")),
        corr_scope=str(train_cfg.get("hallucinate_corr_scope", "returns")),
        freeze_non_return_features=bool(
            train_cfg.get("hallucinate_freeze_non_return_features", True)
        ),
        corr_every_n_steps=int(train_cfg.get("hallucinate_corr_every_n_steps", 1)),
        corr_edge_fraction=float(train_cfg.get("hallucinate_corr_edge_fraction", 1.0)),
        corr_edge_min=int(train_cfg.get("hallucinate_corr_edge_min", 1)),
        adaptive_lr=bool(args.hall_adaptive_lr),
        adaptive_lr_patience=int(args.hall_adaptive_lr_patience),
        adaptive_lr_decay=float(args.hall_adaptive_lr_decay),
        adaptive_lr_min=float(args.hall_adaptive_lr_min),
        early_stop_on_target_hit=bool(args.early_stop_on_target_hit),
        target_hit_patience=int(args.target_hit_patience),
        moment_mean_weight=float(args.hall_moment_mean),
        moment_var_weight=float(args.hall_moment_var),
        moment_skew_weight=float(args.hall_moment_skew),
        moment_scope=str(train_cfg.get("hallucinate_moment_scope", "returns")),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    def _run_once(
        hcfg: HallucinationConfig,
        constraint_weight: float,
        nontarget_drift_weight: float,
    ):
        scenario_rows = []
        diag_rows = []

        for scenario_id, idx in enumerate(idxs):
            data = graphs[idx]
            tickers = (
                tickers_list[idx]
                if idx < len(tickers_list)
                else [f"n{i}" for i in range(data.num_nodes)]
            )
            date = dates[idx] if idx < len(dates) else ""

            ret_mean = getattr(data, "ret_mean", None)
            ret_std = getattr(data, "ret_std", None)

            x_pos = data.x.clone()
            if ret_mean is not None and ret_std is not None:
                ret_mean = ret_mean.view(-1, 1)
                ret_std = ret_std.view(-1, 1)
                pos_returns = x_pos[:, :returns_len] * ret_std + ret_mean
            else:
                pos_returns = x_pos[:, :returns_len]

            constraint_fn = None
            constraint_monitor_fn = None
            force_indices = None
            target_idx = None
            nontarget_mean_abs_delta = None
            if args.target_ticker:
                if args.target_ticker not in tickers:
                    raise ValueError(f"Target ticker {args.target_ticker} not in graph.")
                target_idx = tickers.index(args.target_ticker)
                force_indices = [target_idx]
                if ret_mean is not None and ret_std is not None:
                    ret_mean_t = ret_mean.view(-1, 1)
                    ret_std_t = ret_std.view(-1, 1)
                else:
                    ret_mean_t = None
                    ret_std_t = None

                pos_returns_ref = pos_returns.detach()

                def _constraint(x_var, idx=target_idx):
                    if ret_mean_t is not None and ret_std_t is not None:
                        rets = x_var[idx, :returns_len] * ret_std_t[idx] + ret_mean_t[idx]
                    else:
                        rets = x_var[idx, :returns_len]
                    cum = torch.exp(rets.sum()) - 1.0
                    diff = _constraint_diff(
                        cum,
                        args.target_drop,
                        args.constraint_mode,
                        args.constraint_tolerance,
                    )
                    loss = constraint_weight * diff.pow(2)

                    if nontarget_drift_weight > 0 and x_var.size(0) > 1:
                        if ret_mean_t is not None and ret_std_t is not None:
                            all_rets = x_var[:, :returns_len] * ret_std_t + ret_mean_t
                        else:
                            all_rets = x_var[:, :returns_len]
                        nt_mask = torch.ones(all_rets.size(0), dtype=torch.bool, device=all_rets.device)
                        nt_mask[idx] = False
                        if nt_mask.any():
                            nt_ref = pos_returns_ref.to(all_rets.device)
                            drift = (all_rets[nt_mask] - nt_ref[nt_mask]).abs().mean()
                            tol = max(0.0, float(args.nontarget_drift_tolerance))
                            if tol > 0:
                                drift = torch.relu(drift - tol)
                            loss = loss + float(nontarget_drift_weight) * drift.pow(2)
                    return loss

                def _constraint_monitor(x_var, idx=target_idx):
                    if ret_mean_t is not None and ret_std_t is not None:
                        rets = x_var[idx, :returns_len] * ret_std_t[idx] + ret_mean_t[idx]
                    else:
                        rets = x_var[idx, :returns_len]
                    cum = torch.exp(rets.sum()) - 1.0
                    diff_t = _constraint_diff(
                        cum,
                        args.target_drop,
                        args.constraint_mode,
                        args.constraint_tolerance,
                    )
                    hall_minus_target = float(cum.detach().item() - args.target_drop)
                    hit = _constraint_hit(
                        hall_minus_target,
                        args.target_drop,
                        args.constraint_mode,
                        args.target_tolerance,
                    )
                    return {
                        "hit": bool(hit),
                        "diff": float(diff_t.detach().abs().item()),
                        "hall_minus_target": hall_minus_target,
                    }

                constraint_fn = _constraint
                constraint_monitor_fn = _constraint_monitor

            attempt = 0
            hcfg_local = hcfg
            while True:
                x_neg = hallucinate_negative(
                    model,
                    data.x,
                    data.edge_index,
                    getattr(data, "edge_attr", None),
                    torch.zeros(data.num_nodes, dtype=torch.long),
                    hcfg_local,
                    edge_weight=getattr(data, "edge_weight", None),
                    constraint_fn=constraint_fn,
                    constraint_monitor_fn=constraint_monitor_fn,
                    force_indices=force_indices,
                    critic=critic,
                )

                if ret_mean is not None and ret_std is not None:
                    neg_returns = x_neg[:, :returns_len] * ret_std + ret_mean
                else:
                    neg_returns = x_neg[:, :returns_len]

                if args.min_hall_delta <= 0:
                    break

                delta = float(torch.mean(torch.abs(neg_returns - pos_returns)).item())
                if delta >= args.min_hall_delta or attempt >= args.hall_retry_limit:
                    if delta < args.min_hall_delta and attempt >= args.hall_retry_limit:
                        print(
                            f"scenario {scenario_id} {date}: "
                            f"delta={delta:.4f} < {args.min_hall_delta} after {attempt} retries"
                        )
                    break

                attempt += 1
                new_steps = hcfg_local.steps + args.hall_retry_steps_inc
                new_lr = hcfg_local.lr * args.hall_retry_lr_mult
                new_clamp = (
                    None
                    if hcfg_local.clamp_std is None
                    else hcfg_local.clamp_std + args.hall_retry_clamp_inc
                )
                new_node_fraction = min(1.0, hcfg_local.node_fraction + args.hall_retry_node_inc)
                hcfg_local = replace(
                    hcfg_local,
                    steps=new_steps,
                    lr=new_lr,
                    clamp_std=new_clamp,
                    node_fraction=new_node_fraction,
                )

            if target_idx is not None and neg_returns.size(0) > 1:
                nt_mask = torch.ones(neg_returns.size(0), dtype=torch.bool, device=neg_returns.device)
                nt_mask[target_idx] = False
                if nt_mask.any():
                    nontarget_mean_abs_delta = float(
                        torch.mean(torch.abs(neg_returns[nt_mask] - pos_returns[nt_mask])).item()
                    )

            pos_returns_np = pos_returns.detach().cpu().numpy()
            neg_returns_np = neg_returns.detach().cpu().numpy()

            if target_idx is not None:
                real_cum = float(np.exp(np.sum(pos_returns_np[target_idx])) - 1.0)
                hall_cum = float(np.exp(np.sum(neg_returns_np[target_idx])) - 1.0)
                diff = hall_cum - args.target_drop
                print(
                    f"scenario {scenario_id} {date} {args.target_ticker}: "
                    f"target={args.target_drop:.2%} real={real_cum:.2%} "
                    f"hall={hall_cum:.2%} diff={diff:.2%}"
                )
                diag_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "graph_index": idx,
                        "date": date,
                        "ticker": args.target_ticker,
                        "constraint_mode": args.constraint_mode,
                        "constraint_tolerance": float(args.constraint_tolerance),
                        "target_drop": args.target_drop,
                        "real_cum_return": real_cum,
                        "hall_cum_return": hall_cum,
                        "hall_minus_target": diff,
                        "abs_error": abs(diff),
                        "nontarget_mean_abs_delta": nontarget_mean_abs_delta,
                        "objective_track": objective_track,
                        "energy_component": energy_component,
                        "component_split_mode": split_mode,
                        "encoder_checkpoint": encoder_checkpoint,
                        "critic_checkpoint": critic_checkpoint,
                        "train_neg_mode": train_neg_mode,
                        "hit": int(
                            _constraint_hit(
                                diff,
                                args.target_drop,
                                args.constraint_mode,
                                args.target_tolerance,
                            )
                        ),
                    }
                )

            selected = list(range(len(tickers)))
            if args.max_tickers and args.max_tickers > 0:
                pos_cum = np.exp(np.cumsum(pos_returns_np, axis=1))
                neg_cum = np.exp(np.cumsum(neg_returns_np, axis=1))
                diff = neg_cum[:, -1] - pos_cum[:, -1]
                ranked = np.argsort(diff)
                selected = ranked[: args.max_tickers].tolist()
                if args.target_ticker and args.target_ticker in tickers:
                    t_idx = tickers.index(args.target_ticker)
                    if t_idx not in selected:
                        selected.insert(0, t_idx)

            for i in selected:
                scenario_rows.append(
                    [
                        scenario_id,
                        idx,
                        date,
                        (args.target_ticker or ""),
                        tickers[i],
                        "real",
                        objective_track,
                        energy_component,
                        split_mode,
                        encoder_checkpoint,
                        critic_checkpoint,
                        train_neg_mode,
                    ]
                    + list(pos_returns_np[i])
                )
                scenario_rows.append(
                    [
                        scenario_id,
                        idx,
                        date,
                        (args.target_ticker or ""),
                        tickers[i],
                        "halluc",
                        objective_track,
                        energy_component,
                        split_mode,
                        encoder_checkpoint,
                        critic_checkpoint,
                        train_neg_mode,
                    ]
                    + list(neg_returns_np[i])
                )

        return scenario_rows, diag_rows

    constraint_weight = float(args.constraint_weight)
    nontarget_drift_weight = float(args.nontarget_drift_weight)
    max_steps = max(1, int(args.max_adapt_steps)) if args.adaptive else 1
    attempt = 0
    final_rows = []
    final_diag = []

    while attempt < max_steps:
        prev_state = (
            constraint_weight,
            hall_cfg.steps,
            hall_cfg.lr,
            hall_cfg.l2_weight,
            hall_cfg.mean_weight,
            hall_cfg.std_weight,
            hall_cfg.corr_weight,
            hall_cfg.node_fraction,
            hall_cfg.clamp_std,
            nontarget_drift_weight,
        )
        attempt += 1
        clamp_display = (
            f"{hall_cfg.clamp_std:.2f}" if hall_cfg.clamp_std is not None else "none"
        )
        print(
            "adaptive attempt "
            f"{attempt}/{max_steps} | "
            f"constraint_weight={constraint_weight:.3f} | "
            f"hall_steps={hall_cfg.steps} | "
            f"hall_lr={hall_cfg.lr:.4f} | "
            f"hall_l2={hall_cfg.l2_weight:.4f} | "
            f"hall_node_fraction={hall_cfg.node_fraction:.2f} | "
            f"hall_corr={hall_cfg.corr_weight:.3f} | "
            f"hall_mean={hall_cfg.mean_weight:.4f} | "
            f"hall_std={hall_cfg.std_weight:.4f} | "
            f"hall_clamp={clamp_display} | "
            f"nontarget_drift_weight={nontarget_drift_weight:.3f}"
        )
        scenario_rows, diag_rows = _run_once(hall_cfg, constraint_weight, nontarget_drift_weight)
        final_rows = scenario_rows
        final_diag = diag_rows

        if not args.adaptive or not diag_rows:
            break

        diffs = [row["hall_minus_target"] for row in diag_rows]
        abs_diffs = [abs(d) for d in diffs]
        hits = sum(
            1
            for d in diffs
            if _constraint_hit(
                d,
                args.target_drop,
                args.constraint_mode,
                args.target_tolerance,
            )
        )
        hit_rate = hits / len(diffs)
        mean_diff = sum(diffs) / len(diffs)
        med_diff = sorted(diffs)[len(diffs) // 2]
        mean_abs = sum(abs_diffs) / len(abs_diffs)
        med_abs = sorted(abs_diffs)[len(abs_diffs) // 2]
        p90_abs = float(np.quantile(np.array(abs_diffs), 0.9))
        nontarget_diffs = [
            float(row["nontarget_mean_abs_delta"])
            for row in diag_rows
            if row.get("nontarget_mean_abs_delta") is not None
        ]
        mean_nontarget = (
            sum(nontarget_diffs) / len(nontarget_diffs) if nontarget_diffs else float("nan")
        )
        print(
            "constraint summary: "
            f"hit_rate={hits}/{len(diffs)} ({hit_rate:.1%}) | "
            f"mean_diff={mean_diff:.4f} | median_diff={med_diff:.4f} | "
            f"mean_abs={mean_abs:.4f} | median_abs={med_abs:.4f} | p90_abs={p90_abs:.4f}"
            + (
                f" | mean_non_target_abs={mean_nontarget:.4f}"
                if np.isfinite(mean_nontarget)
                else ""
            )
        )
        drift_ok = (not nontarget_diffs) or (mean_nontarget <= float(args.max_nontarget_drift))
        if hit_rate >= args.target_hit_rate and drift_ok:
            print("Target hit rate and non-target drift thresholds reached.")
            break
        if hit_rate >= args.target_hit_rate and not drift_ok:
            print(
                "Target hit rate reached but non-target drift is above threshold; tightening drift control."
            )

        # Adaptive adjustments
        if args.constraint_mode == "exact":
            needs_harder = mean_diff > args.target_tolerance
            too_hard = mean_diff < -args.target_tolerance
        elif args.target_drop < 0:
            needs_harder = mean_diff > args.target_tolerance
            too_hard = mean_diff < -args.target_tolerance
        else:
            needs_harder = mean_diff < -args.target_tolerance
            too_hard = mean_diff > args.target_tolerance

        needs_drift_tightening = (not drift_ok) and np.isfinite(mean_nontarget)

        if needs_drift_tightening:
            base_weight = (
                nontarget_drift_weight
                if nontarget_drift_weight > 0
                else max(1.0, 0.25 * constraint_weight)
            )
            nontarget_drift_weight = min(
                base_weight * args.adapt_nontarget_mult,
                args.adapt_max_nontarget_weight,
            )
            hall_cfg.l2_weight = min(
                hall_cfg.l2_weight * args.adapt_nontarget_reg_mult,
                2.0,
            )
            hall_cfg.mean_weight = min(
                hall_cfg.mean_weight * args.adapt_nontarget_reg_mult,
                2.0,
            )
            hall_cfg.std_weight = min(
                hall_cfg.std_weight * args.adapt_nontarget_reg_mult,
                2.0,
            )
            hall_cfg.corr_weight = min(
                hall_cfg.corr_weight * args.adapt_nontarget_reg_mult,
                5.0,
            )
            hall_cfg.lr = max(
                hall_cfg.lr / max(args.adapt_hall_lr_mult, 1e-6),
                0.001,
            )
        elif needs_harder:
            constraint_weight = min(
                constraint_weight * args.adapt_constraint_mult,
                args.adapt_max_constraint,
            )
            hall_cfg.steps = min(hall_cfg.steps + args.adapt_hall_step_inc, args.adapt_max_steps)
            hall_cfg.lr = min(hall_cfg.lr * args.adapt_hall_lr_mult, args.adapt_max_lr)
            hall_cfg.l2_weight = max(hall_cfg.l2_weight * args.adapt_hall_l2_mult, args.adapt_min_l2)
            hall_cfg.mean_weight = max(hall_cfg.mean_weight * args.adapt_hall_mean_mult, args.adapt_min_mean)
            hall_cfg.std_weight = max(hall_cfg.std_weight * args.adapt_hall_std_mult, args.adapt_min_std)
            hall_cfg.corr_weight = max(hall_cfg.corr_weight * args.adapt_hall_corr_mult, args.adapt_min_corr)
            hall_cfg.node_fraction = min(
                1.0, hall_cfg.node_fraction + args.adapt_hall_node_inc
            )
            if hall_cfg.clamp_std is not None:
                hall_cfg.clamp_std = min(
                    hall_cfg.clamp_std + args.adapt_hall_clamp_inc,
                    args.adapt_max_clamp_std,
                )
        elif too_hard:
            constraint_weight = max(
                constraint_weight / args.adapt_constraint_mult, 1.0
            )
            hall_cfg.steps = max(1, hall_cfg.steps - args.adapt_hall_step_inc)
            hall_cfg.lr = max(hall_cfg.lr / args.adapt_hall_lr_mult, 0.001)
            hall_cfg.l2_weight = min(hall_cfg.l2_weight / args.adapt_hall_l2_mult, 0.2)
            hall_cfg.mean_weight = min(hall_cfg.mean_weight / args.adapt_hall_mean_mult, 0.2)
            hall_cfg.std_weight = min(hall_cfg.std_weight / args.adapt_hall_std_mult, 0.2)
            hall_cfg.corr_weight = min(hall_cfg.corr_weight / args.adapt_hall_corr_mult, 1.0)
            hall_cfg.node_fraction = max(0.1, hall_cfg.node_fraction - args.adapt_hall_node_inc)
            if hall_cfg.clamp_std is not None:
                hall_cfg.clamp_std = max(
                    1.0,
                    hall_cfg.clamp_std - args.adapt_hall_clamp_inc,
                )
        elif mean_abs > args.target_tolerance:
            # Centered but noisy around target: increase precision pressure.
            constraint_weight = min(
                constraint_weight * (1.0 + 0.5 * (args.adapt_constraint_mult - 1.0)),
                args.adapt_max_constraint,
            )
            hall_cfg.steps = min(hall_cfg.steps + 1, args.adapt_max_steps)
        else:
            # mean diff close to target but hit rate low -> increase diversity slightly
            hall_cfg.steps = min(hall_cfg.steps + 1, args.adapt_max_steps)
            hall_cfg.lr = min(hall_cfg.lr * 1.05, args.adapt_max_lr)

        new_state = (
            constraint_weight,
            hall_cfg.steps,
            hall_cfg.lr,
            hall_cfg.l2_weight,
            hall_cfg.mean_weight,
            hall_cfg.std_weight,
            hall_cfg.corr_weight,
            hall_cfg.node_fraction,
            hall_cfg.clamp_std,
            nontarget_drift_weight,
        )
        if new_state == prev_state and attempt < max_steps:
            print("adaptive tuning saturated at current caps; stopping early.")
            break

    # Write final scenario book
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "scenario_id",
            "graph_index",
            "date",
            "target_ticker",
            "ticker",
            "series",
            "objective_track",
            "energy_component",
            "component_split_mode",
            "encoder_checkpoint",
            "critic_checkpoint",
            "train_neg_mode",
        ] + [
            f"r{i}" for i in range(returns_len)
        ]
        w.writerow(header)
        for row in final_rows:
            w.writerow(row)

    if args.diag_out and final_diag:
        diag_path = Path(args.diag_out)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv

        with diag_path.open("w", newline="") as f:
            w = _csv.DictWriter(
                f,
                fieldnames=[
                    "scenario_id",
                    "graph_index",
                    "date",
                    "ticker",
                    "constraint_mode",
                    "constraint_tolerance",
                    "target_drop",
                    "real_cum_return",
                    "hall_cum_return",
                    "hall_minus_target",
                    "abs_error",
                    "nontarget_mean_abs_delta",
                    "objective_track",
                    "energy_component",
                    "component_split_mode",
                    "encoder_checkpoint",
                    "critic_checkpoint",
                    "train_neg_mode",
                    "hit",
                ],
            )
            w.writeheader()
            for row in final_diag:
                w.writerow(row)
        print(f"Wrote {diag_path}")
        diffs = [row["hall_minus_target"] for row in final_diag]
        abs_diffs = [abs(d) for d in diffs]
        if diffs:
            hits = sum(
                1
                for d in diffs
                if _constraint_hit(
                    d,
                    args.target_drop,
                    args.constraint_mode,
                    args.target_tolerance,
                )
            )
            hit_rate = hits / len(diffs)
            mean_diff = sum(diffs) / len(diffs)
            med_diff = sorted(diffs)[len(diffs) // 2]
            mean_abs = sum(abs_diffs) / len(abs_diffs)
            med_abs = sorted(abs_diffs)[len(abs_diffs) // 2]
            p90_abs = float(np.quantile(np.array(abs_diffs), 0.9))
            nontarget_diffs = [
                float(row["nontarget_mean_abs_delta"])
                for row in final_diag
                if row.get("nontarget_mean_abs_delta") is not None
            ]
            mean_nontarget = (
                sum(nontarget_diffs) / len(nontarget_diffs) if nontarget_diffs else float("nan")
            )
            print(
                "constraint summary: "
                f"hit_rate={hits}/{len(diffs)} ({hit_rate:.1%}) | "
                f"mean_diff={mean_diff:.4f} | median_diff={med_diff:.4f} | "
                f"mean_abs={mean_abs:.4f} | median_abs={med_abs:.4f} | p90_abs={p90_abs:.4f}"
                + (
                    f" | mean_non_target_abs={mean_nontarget:.4f}"
                    if np.isfinite(mean_nontarget)
                    else ""
                )
            )

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
