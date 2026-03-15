#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
import tomllib

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.ff import goodness
from frisk.models import (
    CompositeEnergyCritic,
    EnergyCritic,
    EnergyCriticEnsemble,
    GCNEncoder,
    SequenceEnergyCritic,
)
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
            "Critic checkpoint load mismatch in feature_attribution. "
            f"checkpoint={checkpoint_path}. "
            f"expected={expected}. "
            "Check that train.critic_* and train.sequence_critic_* settings match the checkpoint. "
            f"original_error={exc}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute gradient x input attributions for FF graph features."
    )
    parser.add_argument("--config", required=True, help="Path to config TOML")
    parser.add_argument("--graphs", default="", help="Override graphs .pt path")
    parser.add_argument("--model", default="", help="Override encoder checkpoint path")
    parser.add_argument("--critic-model", default="", help="Optional critic checkpoint path")
    parser.add_argument("--num-graphs", type=int, default=128, help="Number of graphs to sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        default="runs/experiments/default/diagnostics/feature_attribution.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})

    graphs_path = Path(args.graphs or train_cfg.get("graphs", "data/processed/graphs.pt"))
    artifact = load_graph_artifact(graphs_path, include_tickers=False, prefer_lazy=True, prefer_sharded=True)
    graphs = artifact.graphs
    dates = artifact.dates
    if not graphs:
        raise ValueError("No graphs found.")

    model_path = str(args.model or train_cfg.get("save_encoder", train_cfg.get("save_model", ""))).strip()
    if not model_path:
        raise ValueError("No encoder checkpoint path provided.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {model_path}")

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
    model.load_state_dict(_load_state_dict_compat(model_path))
    model.eval()

    critic = None
    critic_path = str(args.critic_model or train_cfg.get("save_critic", "")).strip()
    if critic_path and Path(critic_path).exists():
        critic_hidden_dim = int(train_cfg.get("hidden_dim", 64))
        critic = _build_critic(
            train_cfg,
            hidden_dim=critic_hidden_dim,
            device=torch.device("cpu"),
        )
        _load_critic_checkpoint_or_raise(
            critic=critic,
            checkpoint_path=critic_path,
            config=train_cfg,
            hidden_dim=critic_hidden_dim,
        )
        critic.eval()

    rng = random.Random(args.seed)
    idxs = list(range(len(graphs)))
    rng.shuffle(idxs)
    idxs = idxs[: max(1, min(len(idxs), int(args.num_graphs)))]

    total_attr = torch.zeros(graphs[0].x.shape[1], dtype=torch.float32)
    energy_rows: list[dict] = []
    for idx in idxs:
        g = graphs[idx]
        x = g.x.detach().clone().requires_grad_(True)
        batch = torch.zeros(g.num_nodes, dtype=torch.long)
        edge_weight = getattr(g, "edge_weight", None)
        edge_type = getattr(g, "edge_type", None)
        h = model(x, g.edge_index, edge_weight=edge_weight, edge_type=edge_type)
        energy = goodness(
            h,
            batch,
            temperature=float(train_cfg.get("goodness_temp", 1.0)),
            critic=critic,
        ).mean()
        energy.backward()
        attr = (x.grad * x).abs().mean(dim=0).detach().cpu()
        total_attr += attr
        energy_rows.append(
            {
                "graph_index": int(idx),
                "date": dates[idx] if idx < len(dates) else "",
                "energy": float(energy.detach().item()),
            }
        )

    avg_attr = total_attr / max(1, len(idxs))
    denom = float(avg_attr.sum().item()) + 1e-12
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["feature_idx", "importance", "importance_norm"],
        )
        w.writeheader()
        for i in range(avg_attr.numel()):
            imp = float(avg_attr[i].item())
            w.writerow(
                {
                    "feature_idx": i,
                    "importance": imp,
                    "importance_norm": imp / denom,
                }
            )

    energy_out = out_path.with_name(f"{out_path.stem}_graph_energy.csv")
    with energy_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["graph_index", "date", "energy"])
        w.writeheader()
        for row in energy_rows:
            w.writerow(row)

    top = torch.argsort(avg_attr, descending=True)[: min(10, avg_attr.numel())].tolist()
    print(f"Wrote {out_path}")
    print(f"Wrote {energy_out}")
    print("Top features:", ", ".join(str(int(i)) for i in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
