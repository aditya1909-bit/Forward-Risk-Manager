#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import tomllib

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import GCNEncoder


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


def _project_embeddings(z: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, str]:
    m = str(method).strip().lower()
    if m == "umap":
        try:
            import umap  # type: ignore

            proj = umap.UMAP(n_components=2, random_state=seed)
            return proj.fit_transform(z), "umap"
        except Exception:
            m = "tsne"
    if m == "tsne":
        try:
            from sklearn.manifold import TSNE  # type: ignore

            proj = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
            return proj.fit_transform(z), "tsne"
        except Exception:
            m = "pca"
    if m == "pca":
        try:
            from sklearn.decomposition import PCA  # type: ignore

            proj = PCA(n_components=2, random_state=seed)
            return proj.fit_transform(z), "pca"
        except Exception:
            pass

    # SVD fallback
    zc = z - z.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(zc, full_matrices=False)
    out = u[:, :2] * s[:2]
    return out, "svd"


def main() -> int:
    parser = argparse.ArgumentParser(description="Project graph embeddings for diagnostics.")
    parser.add_argument("--config", required=True, help="Path to config TOML")
    parser.add_argument("--graphs", default="", help="Override graphs .pt path")
    parser.add_argument("--model", default="", help="Override encoder checkpoint path")
    parser.add_argument(
        "--method",
        choices=["umap", "tsne", "pca", "svd"],
        default="umap",
        help="Projection method (falls back automatically if unavailable).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        default="runs/experiments/default/diagnostics/embedding_projection.csv",
        help="Output projection CSV",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})

    graphs_path = Path(args.graphs or train_cfg.get("graphs", "data/processed/graphs.pt"))
    try:
        payload = torch.load(graphs_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(graphs_path, map_location="cpu")
    graphs = payload["graphs"] if isinstance(payload, dict) else payload
    dates = payload.get("dates", []) if isinstance(payload, dict) else []
    tickers = payload.get("tickers", []) if isinstance(payload, dict) else []
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
        residual_edge_enabled=bool(train_cfg.get("residual_edge_weight_enabled", False)),
        residual_edge_hidden_dim=int(train_cfg.get("residual_edge_hidden_dim", 32)),
        residual_edge_max_delta=float(train_cfg.get("residual_edge_max_delta", 0.25)),
        residual_edge_detach_features=bool(train_cfg.get("residual_edge_detach_features", True)),
    )
    model.load_state_dict(_load_state_dict_compat(model_path))
    model.eval()

    loader = DataLoader(
        graphs,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        drop_last=False,
    )
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            edge_weight = getattr(batch, "edge_weight", None)
            h = model(batch.x, batch.edge_index, edge_weight=edge_weight)
            z = global_mean_pool(h, batch.batch)
            embeddings.append(z.detach().cpu())
    z_all = torch.cat(embeddings, dim=0).numpy()
    proj, method_used = _project_embeddings(z_all, args.method, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "graph_index",
                "date",
                "num_nodes",
                "num_tickers",
                "proj_x",
                "proj_y",
                "projection_method",
            ],
        )
        w.writeheader()
        for i in range(proj.shape[0]):
            num_tickers = len(tickers[i]) if i < len(tickers) else 0
            w.writerow(
                {
                    "graph_index": i,
                    "date": dates[i] if i < len(dates) else "",
                    "num_nodes": int(graphs[i].num_nodes),
                    "num_tickers": int(num_tickers),
                    "proj_x": float(proj[i, 0]),
                    "proj_y": float(proj[i, 1]),
                    "projection_method": method_used,
                }
            )
    print(f"Wrote {out_path} (method={method_used})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
