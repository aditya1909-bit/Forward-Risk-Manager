#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tomllib


def _load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
            "strict_component_split",
            "freeze_encoder",
            "freeze_critic",
            "encoder_checkpoint_in",
            "critic_checkpoint_in",
            "save_encoder",
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
            "strict_component_split",
            "freeze_encoder",
            "freeze_critic",
            "encoder_checkpoint_in",
            "critic_checkpoint_in",
            "save_critic",
            "critic_hidden_dim",
            "critic_num_layers",
            "critic_dropout",
            "critic_positive_activation",
        ],
    )

    _run(stage1)
    _run(stage2)
    print(f"done: encoder={encoder_out} critic={critic_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
