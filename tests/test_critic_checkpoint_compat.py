from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _composite_critic_cfg() -> dict:
    return {
        "seed": 7,
        "hidden_dim": 16,
        "critic_hidden_dim": 16,
        "critic_num_layers": 2,
        "critic_dropout": 0.1,
        "critic_positive_activation": "softplus",
        "critic_ensemble_size": 3,
        "critic_ensemble_seed_stride": 97,
        "sequence_critic_enabled": True,
        "sequence_critic_weight": 0.2,
        "sequence_critic_hidden_dim": 16,
        "sequence_critic_num_layers": 1,
        "sequence_critic_dropout": 0.0,
        "sequence_critic_positive_activation": "softplus",
    }


def test_scenario_book_composite_critic_checkpoint_loads(tmp_path: Path):
    mod = _load_script("scenario_book.py")
    cfg = _composite_critic_cfg()
    hidden_dim = int(cfg["hidden_dim"])
    ckpt_path = tmp_path / "scenario_critic.pt"

    critic_src = mod._build_critic(cfg, hidden_dim=hidden_dim, device=torch.device("cpu"))
    torch.save({"state_dict": critic_src.state_dict()}, ckpt_path)

    critic_dst = mod._build_critic(cfg, hidden_dim=hidden_dim, device=torch.device("cpu"))
    mod._load_critic_checkpoint_or_raise(
        critic=critic_dst,
        checkpoint_path=str(ckpt_path),
        config=cfg,
        hidden_dim=hidden_dim,
    )


def test_feature_attribution_composite_critic_checkpoint_loads(tmp_path: Path):
    mod = _load_script("feature_attribution.py")
    cfg = _composite_critic_cfg()
    hidden_dim = int(cfg["hidden_dim"])
    ckpt_path = tmp_path / "feature_attr_critic.pt"

    critic_src = mod._build_critic(cfg, hidden_dim=hidden_dim, device=torch.device("cpu"))
    torch.save({"state_dict": critic_src.state_dict()}, ckpt_path)

    critic_dst = mod._build_critic(cfg, hidden_dim=hidden_dim, device=torch.device("cpu"))
    mod._load_critic_checkpoint_or_raise(
        critic=critic_dst,
        checkpoint_path=str(ckpt_path),
        config=cfg,
        hidden_dim=hidden_dim,
    )


def test_scenario_book_mismatch_error_includes_structure_summary(tmp_path: Path):
    mod = _load_script("scenario_book.py")
    base_cfg = _composite_critic_cfg()
    hidden_dim = int(base_cfg["hidden_dim"])
    ckpt_path = tmp_path / "mismatch_critic.pt"

    plain_cfg = dict(base_cfg)
    plain_cfg["critic_ensemble_size"] = 1
    plain_cfg["sequence_critic_enabled"] = False
    critic_src = mod._build_critic(plain_cfg, hidden_dim=hidden_dim, device=torch.device("cpu"))
    torch.save({"state_dict": critic_src.state_dict()}, ckpt_path)

    critic_dst = mod._build_critic(base_cfg, hidden_dim=hidden_dim, device=torch.device("cpu"))
    with pytest.raises(RuntimeError) as exc:
        mod._load_critic_checkpoint_or_raise(
            critic=critic_dst,
            checkpoint_path=str(ckpt_path),
            config=base_cfg,
            hidden_dim=hidden_dim,
        )

    msg = str(exc.value)
    assert "Critic checkpoint load mismatch in scenario_book" in msg
    assert str(ckpt_path) in msg
    assert "critic_ensemble_size=" in msg
    assert "sequence_critic_enabled=" in msg
