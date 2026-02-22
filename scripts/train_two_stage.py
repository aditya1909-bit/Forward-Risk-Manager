#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib


def _load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")
    subprocess.run(cmd, check=True, env=env)


def _run_with_crash_fallback(cmd: list[str], stage_name: str) -> None:
    try:
        _run(cmd)
        return
    except subprocess.CalledProcessError as exc:
        is_signal_crash = int(exc.returncode) < 0
        if not is_signal_crash:
            raise

        fallback_cmd = list(cmd)
        added = []
        if "--no-torch-compile" not in fallback_cmd:
            fallback_cmd.append("--no-torch-compile")
            added.append("--no-torch-compile")
        if "--no-auto-tune-batch" not in fallback_cmd:
            fallback_cmd.append("--no-auto-tune-batch")
            added.append("--no-auto-tune-batch")
        if not added:
            raise

        sig = -int(exc.returncode)
        print(
            f"{stage_name}: subprocess crashed with signal {sig}; "
            f"retrying once with {' '.join(added)}."
        )
        _run(fallback_cmd)


def _artifact_ready(path: str) -> bool:
    p = Path(path)
    return p.is_file() and p.stat().st_size > 0


def _append_flag(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def _apply_section_overrides(cmd: list[str], section: dict, keys: list[str]) -> None:
    for key in keys:
        if key not in section:
            continue
        flag = "--" + key.replace("_", "-")
        _append_flag(cmd, flag, section.get(key))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run strict two-stage training: encoder (SC) then critic (FF)."
    )
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument(
        "--encoder-out",
        default="",
        help="Override encoder checkpoint output path (defaults to train.save_encoder or train.save_model).",
    )
    parser.add_argument(
        "--critic-out",
        default="",
        help="Override critic checkpoint output path (defaults to train.save_critic).",
    )
    parser.add_argument(
        "--critic-neg-mode",
        default="time_flip+noise",
        choices=[
            "shuffle",
            "noise",
            "shuffle+noise",
            "time_flip",
            "shuffle+time_flip",
            "time_flip+noise",
            "block_bootstrap",
            "cross_asset_mix",
            "phase_randomize",
            "hallucinate",
            "schedule",
            "mix",
        ],
        help="Negative mode for critic stage.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip a stage if its output checkpoint already exists (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Always rerun both stages, even if checkpoints exist.",
    )
    parser.add_argument(
        "--force-stage1",
        action="store_true",
        help="Force rerunning encoder stage even when encoder checkpoint exists.",
    )
    parser.add_argument(
        "--force-stage2",
        action="store_true",
        help="Force rerunning critic stage even when critic checkpoint exists.",
    )
    parser.add_argument(
        "--stage2-no-torch-compile",
        dest="stage2_no_torch_compile",
        action="store_true",
        default=False,
        help="Disable torch.compile for critic stage.",
    )
    parser.add_argument(
        "--stage2-torch-compile",
        dest="stage2_no_torch_compile",
        action="store_false",
        help="Allow torch.compile during critic stage.",
    )
    parser.add_argument(
        "--stage1-no-torch-compile",
        dest="stage1_no_torch_compile",
        action="store_true",
        default=False,
        help="Disable torch.compile for encoder stage.",
    )
    parser.add_argument(
        "--stage1-torch-compile",
        dest="stage1_no_torch_compile",
        action="store_false",
        help="Allow torch.compile during encoder stage.",
    )
    parser.add_argument(
        "--stage1-no-auto-tune-batch",
        dest="stage1_no_auto_tune_batch",
        action="store_true",
        default=False,
        help="Disable auto batch-size probing for encoder stage.",
    )
    parser.add_argument(
        "--stage1-auto-tune-batch",
        dest="stage1_no_auto_tune_batch",
        action="store_false",
        help="Allow auto batch-size probing during encoder stage.",
    )
    parser.add_argument(
        "--stage2-no-auto-tune-batch",
        dest="stage2_no_auto_tune_batch",
        action="store_true",
        default=False,
        help="Disable auto batch-size probing for critic stage.",
    )
    parser.add_argument(
        "--stage2-auto-tune-batch",
        dest="stage2_no_auto_tune_batch",
        action="store_false",
        help="Allow auto batch-size probing during critic stage.",
    )
    parser.add_argument(
        "--stage1-compile-mode",
        default="max-autotune-no-cudagraphs",
        help="Compile mode override for encoder stage (default: max-autotune-no-cudagraphs).",
    )
    parser.add_argument(
        "--stage2-compile-mode",
        default="max-autotune-no-cudagraphs",
        help="Compile mode override for critic stage (default: max-autotune-no-cudagraphs).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    train_cfg = cfg.get("train", {})
    encoder_cfg = cfg.get("encoder", {})
    critic_cfg = cfg.get("critic", {})

    encoder_out = (
        str(args.encoder_out).strip()
        or str(encoder_cfg.get("save_encoder", "")).strip()
        or str(train_cfg.get("save_encoder", "")).strip()
        or str(train_cfg.get("save_model", "")).strip()
    )
    critic_out = (
        str(args.critic_out).strip()
        or str(critic_cfg.get("save_critic", "")).strip()
        or str(train_cfg.get("save_critic", "")).strip()
    )
    if not encoder_out:
        raise ValueError("No encoder output path found. Set --encoder-out or train.save_encoder/save_model.")
    if not critic_out:
        raise ValueError("No critic output path found. Set --critic-out or train.save_critic.")

    py = sys.executable

    stage1 = [
        py,
        "scripts/train_ff_gnn.py",
        "--config",
        str(cfg_path),
        "--neg-mode",
        "self_contrastive",
        "--strict-component-split",
        "--save-encoder",
        encoder_out,
    ]
    if args.stage1_no_torch_compile:
        stage1.append("--no-torch-compile")
    else:
        mode = str(args.stage1_compile_mode).strip()
        if mode:
            stage1.extend(["--torch-compile-mode", mode])
    if args.stage1_no_auto_tune_batch:
        stage1.append("--no-auto-tune-batch")
    _apply_section_overrides(
        stage1,
        encoder_cfg,
        [
            "epochs",
            "batch_size",
            "lr",
            "hidden_dim",
            "num_layers",
            "dropout",
            "goodness_target",
            "noise_std",
            "device",
            "seed",
            "loader_workers",
            "self_contrastive_view_mode",
            "self_contrastive_view_noise_std",
            "self_contrastive_ff_weight",
            "self_contrastive_ff_neg_mode",
            "self_contrastive_ff_noise_std",
            "self_contrastive_ff_target",
            "freeze_encoder",
            "encoder_checkpoint_in",
            "critic_checkpoint_in",
            "critic_hidden_dim",
            "critic_num_layers",
            "critic_dropout",
            "critic_positive_activation",
        ],
    )

    stage2 = [
        py,
        "scripts/train_ff_gnn.py",
        "--config",
        str(cfg_path),
        "--neg-mode",
        args.critic_neg_mode,
        "--strict-component-split",
        "--encoder-checkpoint-in",
        encoder_out,
        "--freeze-encoder",
        "--save-critic",
        critic_out,
    ]
    if args.stage2_no_torch_compile:
        stage2.append("--no-torch-compile")
    else:
        mode = str(args.stage2_compile_mode).strip()
        if mode:
            stage2.extend(["--torch-compile-mode", mode])
    if args.stage2_no_auto_tune_batch:
        stage2.append("--no-auto-tune-batch")
    _apply_section_overrides(
        stage2,
        critic_cfg,
        [
            "epochs",
            "batch_size",
            "lr",
            "hidden_dim",
            "num_layers",
            "dropout",
            "goodness_target",
            "noise_std",
            "device",
            "seed",
            "loader_workers",
            "freeze_critic",
            "critic_checkpoint_in",
            "critic_hidden_dim",
            "critic_num_layers",
            "critic_dropout",
            "critic_positive_activation",
        ],
    )

    stage1_done = _artifact_ready(encoder_out)
    stage2_done = _artifact_ready(critic_out)

    if args.resume and stage1_done and not args.force_stage1:
        print(f"skip stage1: encoder checkpoint exists at {encoder_out}")
    else:
        _run_with_crash_fallback(stage1, stage_name="stage1")
        if not _artifact_ready(encoder_out):
            raise RuntimeError(f"stage1 finished but encoder checkpoint missing/empty: {encoder_out}")

    if not _artifact_ready(encoder_out):
        raise RuntimeError(
            f"encoder checkpoint missing/empty before stage2: {encoder_out}. "
            "Run stage1 first or disable resume/force stage1."
        )

    if args.resume and stage2_done and not args.force_stage2:
        print(f"skip stage2: critic checkpoint exists at {critic_out}")
    else:
        _run_with_crash_fallback(stage2, stage_name="stage2")
        if not _artifact_ready(critic_out):
            raise RuntimeError(f"stage2 finished but critic checkpoint missing/empty: {critic_out}")

    print(f"done: encoder={encoder_out} critic={critic_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
