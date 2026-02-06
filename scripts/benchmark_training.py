#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
import sys
import tomllib

import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import GCNEncoder
from frisk.ff import goodness, make_negative, ff_loss
from frisk.hallucinate import HallucinationConfig, hallucinate_negative


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _choose_device(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    return torch.device("cpu")


def _split_graphs(graphs, eval_frac: float = 0.2, seed: int = 7):
    rng = random.Random(seed)
    idx = list(range(len(graphs)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - eval_frac))
    train = [graphs[i] for i in idx[:cut]]
    evals = [graphs[i] for i in idx[cut:]]
    return train, evals


def _get_use_mode(epoch: int, neg_mode: str, warmup: int, mix_start: float, mix_end: float, ramp: int):
    if neg_mode == "schedule":
        return "shuffle" if epoch <= warmup else "hallucinate"
    if neg_mode == "mix":
        if epoch <= warmup:
            return "shuffle"
        ramp = max(1, ramp)
        progress = min(1.0, (epoch - warmup) / ramp)
        p_hall = mix_start + progress * (mix_end - mix_start)
        return "hallucinate" if random.random() < p_hall else "shuffle"
    return neg_mode


def _make_negatives(
    model,
    x,
    batch,
    edge_index,
    edge_attr,
    edge_weight,
    use_mode,
    noise_std,
    hall_cfg: HallucinationConfig,
):
    if use_mode == "hallucinate":
        return hallucinate_negative(
            model,
            x,
            edge_index,
            edge_attr,
            batch,
            hall_cfg,
            edge_weight=edge_weight,
        )
    return make_negative(x, batch, mode=use_mode, noise_std=noise_std)


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _eval_ff_metrics(model, loader, goodness_temp, goodness_target, neg_mode, noise_std, hall_cfg):
    model.eval()
    gpos = []
    gneg = []
    acc_num = 0
    acc_den = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            x = batch.x
            edge_weight = getattr(batch, "edge_weight", None)
            h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
            g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)
            x_neg = _make_negatives(
                model,
                x,
                batch.batch,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                edge_weight,
                neg_mode,
                noise_std,
                hall_cfg,
            )
            h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
            g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp)
            pred_pos = (g_pos > goodness_target)
            pred_neg = (g_neg <= goodness_target)
            acc_num += (pred_pos.sum() + pred_neg.sum()).item()
            acc_den += 2 * g_pos.numel()
            gpos.append(g_pos.mean().item())
            gneg.append(g_neg.mean().item())
    acc = acc_num / acc_den if acc_den else 0.0
    return float(np.mean(gpos)), float(np.mean(gneg)), float(acc)


def _benchmark_ff(
    graphs,
    device,
    config,
    layerwise: bool,
):
    train_graphs, eval_graphs = _split_graphs(graphs, eval_frac=config["eval_frac"], seed=config["seed"])
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "drop_last": False,
        "num_workers": config["loader_workers"],
        "pin_memory": bool(config.get("pin_memory", False)) if device.type == "cuda" else False,
    }
    if config["loader_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
        mp_ctx = config.get("multiprocessing_context", "")
        if mp_ctx:
            loader_kwargs["multiprocessing_context"] = mp_ctx
    loader = DataLoader(train_graphs, **loader_kwargs)
    eval_loader = DataLoader(eval_graphs, batch_size=config["batch_size"], shuffle=False)

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    optim = Adam(model.parameters(), lr=config["lr"])

    hall_cfg = HallucinationConfig(
        steps=config["hall_steps"],
        lr=config["hall_lr"],
        l2_weight=config["hall_l2"],
        mean_weight=config["hall_mean"],
        std_weight=config["hall_std"],
        corr_weight=config["hall_corr"],
        clamp_std=config["hall_clamp"],
        goodness_temp=config["goodness_temp"],
        node_fraction=config["hall_node_fraction"],
        node_min=config["hall_node_min"],
    )

    epoch_times = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        graphs_seen = 0
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            x = batch.x
            edge_weight = getattr(batch, "edge_weight", None)

            use_mode = _get_use_mode(
                epoch,
                config["neg_mode"],
                config["neg_warmup_epochs"],
                config["neg_mix_start"],
                config["neg_mix_end"],
                config["neg_mix_ramp_epochs"],
            )

            if layerwise:
                x_in = x
                for li in range(len(model.layers)):
                    h_pos = model.forward_layer(x_in, batch.edge_index, edge_weight, li)
                    g_pos = goodness(h_pos, batch.batch, temperature=config["goodness_temp"])
                    x_neg = _make_negatives(
                        model,
                        x_in,
                        batch.batch,
                        batch.edge_index,
                        getattr(batch, "edge_attr", None),
                        edge_weight,
                        "hallucinate" if (use_mode == "hallucinate" and li == 0) else "shuffle",
                        config["noise_std"],
                        hall_cfg,
                    )
                    h_neg = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                    g_neg = goodness(h_neg, batch.batch, temperature=config["goodness_temp"])
                    loss = ff_loss(g_pos, g_neg, target=config["goodness_target"])
                    optim.zero_grad()
                    loss.backward()
                    if config["grad_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    optim.step()
                    total_loss += loss.item()
                    x_in = h_pos.detach()
            else:
                h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                g_pos = goodness(h_pos, batch.batch, temperature=config["goodness_temp"])
                x_neg = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    use_mode,
                    config["noise_std"],
                    hall_cfg,
                )
                h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                g_neg = goodness(h_neg, batch.batch, temperature=config["goodness_temp"])
                loss = ff_loss(g_pos, g_neg, target=config["goodness_target"])
                optim.zero_grad()
                loss.backward()
                if config["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optim.step()
                total_loss += loss.item()

            graphs_seen += batch.num_graphs

        _sync(device)
        dt = time.perf_counter() - t0
        epoch_times.append((dt, graphs_seen))

    gpos, gneg, acc = _eval_ff_metrics(
        model,
        eval_loader,
        config["goodness_temp"],
        config["goodness_target"],
        config["eval_neg_mode"],
        config["noise_std"],
        hall_cfg,
    )
    warm = int(config.get("timing_warmup_epochs", 0))
    usable = epoch_times[warm:] if warm < len(epoch_times) else epoch_times
    avg_time = float(np.mean([t for t, _ in usable]))
    avg_gps = float(np.mean([g / t for t, g in usable]))
    return {
        "avg_epoch_s": avg_time,
        "graphs_per_s": avg_gps,
        "eval_g_pos": gpos,
        "eval_g_neg": gneg,
        "eval_sep": gpos - gneg,
        "eval_acc": acc,
    }


def _benchmark_backprop(graphs, device, config):
    train_graphs, eval_graphs = _split_graphs(graphs, eval_frac=config["eval_frac"], seed=config["seed"])
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "drop_last": False,
        "num_workers": config["loader_workers"],
        "pin_memory": bool(config.get("pin_memory", False)) if device.type == "cuda" else False,
    }
    if config["loader_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
        mp_ctx = config.get("multiprocessing_context", "")
        if mp_ctx:
            loader_kwargs["multiprocessing_context"] = mp_ctx
    loader = DataLoader(train_graphs, **loader_kwargs)
    eval_loader = DataLoader(eval_graphs, batch_size=config["batch_size"], shuffle=False)

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    head = torch.nn.Linear(config["hidden_dim"], 1).to(device)
    optim = Adam(list(model.parameters()) + list(head.parameters()), lr=config["lr"])
    bce = torch.nn.BCEWithLogitsLoss()

    hall_cfg = HallucinationConfig(
        steps=config["hall_steps"],
        lr=config["hall_lr"],
        l2_weight=config["hall_l2"],
        mean_weight=config["hall_mean"],
        std_weight=config["hall_std"],
        corr_weight=config["hall_corr"],
        clamp_std=config["hall_clamp"],
        goodness_temp=config["goodness_temp"],
        node_fraction=config["hall_node_fraction"],
        node_min=config["hall_node_min"],
    )

    epoch_times = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        graphs_seen = 0
        for batch in loader:
            batch = batch.to(device)
            edge_weight = getattr(batch, "edge_weight", None)
            x = batch.x
            h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
            z_pos = global_mean_pool(h_pos, batch.batch)
            y_pos = torch.ones(z_pos.size(0), device=device)

            use_mode = _get_use_mode(
                epoch,
                config["neg_mode"],
                config["neg_warmup_epochs"],
                config["neg_mix_start"],
                config["neg_mix_end"],
                config["neg_mix_ramp_epochs"],
            )
            x_neg = _make_negatives(
                model,
                x,
                batch.batch,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                edge_weight,
                use_mode,
                config["noise_std"],
                hall_cfg,
            )
            h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
            z_neg = global_mean_pool(h_neg, batch.batch)
            y_neg = torch.zeros(z_neg.size(0), device=device)

            z = torch.cat([z_pos, z_neg], dim=0)
            y = torch.cat([y_pos, y_neg], dim=0)
            logits = head(z).squeeze(1)
            loss = bce(logits, y)

            optim.zero_grad()
            loss.backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), config["grad_clip"])
            optim.step()
            graphs_seen += batch.num_graphs
        _sync(device)
        dt = time.perf_counter() - t0
        epoch_times.append((dt, graphs_seen))

    # eval accuracy
    model.eval()
    head.eval()
    correct = 0
    total = 0
    gpos = []
    gneg = []
    eval_losses = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            edge_weight = getattr(batch, "edge_weight", None)
            x = batch.x
            h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
            z_pos = global_mean_pool(h_pos, batch.batch)
            y_pos = torch.ones(z_pos.size(0), device=device)

            x_neg = _make_negatives(
                model,
                x,
                batch.batch,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                edge_weight,
                config["eval_neg_mode"],
                config["noise_std"],
                hall_cfg,
            )
            h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
            z_neg = global_mean_pool(h_neg, batch.batch)
            y_neg = torch.zeros(z_neg.size(0), device=device)

            z = torch.cat([z_pos, z_neg], dim=0)
            y = torch.cat([y_pos, y_neg], dim=0)
            logits = head(z).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            eval_losses.append(bce(logits, y).item())
            g_pos = goodness(h_pos, batch.batch, temperature=config["goodness_temp"])
            g_neg = goodness(h_neg, batch.batch, temperature=config["goodness_temp"])
            gpos.append(g_pos.mean().item())
            gneg.append(g_neg.mean().item())

    warm = int(config.get("timing_warmup_epochs", 0))
    usable = epoch_times[warm:] if warm < len(epoch_times) else epoch_times
    avg_time = float(np.mean([t for t, _ in usable]))
    avg_gps = float(np.mean([g / t for t, g in usable]))
    return {
        "avg_epoch_s": avg_time,
        "graphs_per_s": avg_gps,
        "eval_acc": correct / total if total else 0.0,
        "eval_bce": float(np.mean(eval_losses)) if eval_losses else 0.0,
        "eval_g_pos": float(np.mean(gpos)) if gpos else 0.0,
        "eval_g_neg": float(np.mean(gneg)) if gneg else 0.0,
        "eval_sep": float(np.mean(gpos)) - float(np.mean(gneg)) if gpos and gneg else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FF vs backprop training.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument(
        "--modes",
        default="ff_layerwise,ff_e2e,backprop",
        help="Comma-separated modes: ff_layerwise,ff_e2e,backprop",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})
    bench_cfg = cfg.get("benchmark", {})

    graphs_path = Path(train_cfg.get("graphs", "data/processed/graphs.pt"))
    payload = torch.load(graphs_path, map_location="cpu", weights_only=False)
    graphs = payload["graphs"]

    device = _choose_device(train_cfg.get("device", "cpu"))
    _set_seed(int(train_cfg.get("seed", 7)))
    if train_cfg.get("torch_num_threads"):
        torch.set_num_threads(int(train_cfg["torch_num_threads"]))
    if train_cfg.get("torch_num_interop_threads"):
        torch.set_num_interop_threads(int(train_cfg["torch_num_interop_threads"]))

    config = {
        "epochs": int(bench_cfg.get("epochs", 5)),
        "batch_size": int(bench_cfg.get("batch_size", train_cfg.get("batch_size", 16))),
        "hidden_dim": int(train_cfg.get("hidden_dim", 64)),
        "num_layers": int(train_cfg.get("num_layers", 2)),
        "dropout": float(train_cfg.get("dropout", 0.1)),
        "lr": float(train_cfg.get("lr", 1e-3)),
        "neg_mode": str(bench_cfg.get("neg_mode", train_cfg.get("neg_mode", "shuffle"))),
        "eval_neg_mode": str(bench_cfg.get("eval_neg_mode", "shuffle")),
        "noise_std": float(train_cfg.get("noise_std", 0.05)),
        "neg_warmup_epochs": int(train_cfg.get("neg_warmup_epochs", 0)),
        "neg_mix_start": float(train_cfg.get("neg_mix_start", 0.0)),
        "neg_mix_end": float(train_cfg.get("neg_mix_end", 0.3)),
        "neg_mix_ramp_epochs": int(train_cfg.get("neg_mix_ramp_epochs", 10)),
        "goodness_target": float(train_cfg.get("goodness_target", 1.0)),
        "goodness_temp": float(train_cfg.get("goodness_temp", 1.0)),
        "grad_clip": float(train_cfg.get("grad_clip", 1.0)),
        "loader_workers": int(train_cfg.get("loader_workers", 0)),
        "persistent_workers": bool(train_cfg.get("dataloader_persistent_workers", True)),
        "prefetch_factor": int(train_cfg.get("dataloader_prefetch_factor", 2)),
        "pin_memory": bool(train_cfg.get("dataloader_pin_memory", False)),
        "multiprocessing_context": str(train_cfg.get("dataloader_mp_context", "")),
        "eval_frac": float(bench_cfg.get("eval_frac", 0.2)),
        "seed": int(train_cfg.get("seed", 7)),
        "hall_steps": int(train_cfg.get("hallucinate_steps", 3)),
        "hall_lr": float(train_cfg.get("hallucinate_lr", 0.03)),
        "hall_l2": float(train_cfg.get("hallucinate_l2", 0.05)),
        "hall_mean": float(train_cfg.get("hallucinate_mean", 0.01)),
        "hall_std": float(train_cfg.get("hallucinate_std", 0.01)),
        "hall_corr": float(train_cfg.get("hallucinate_corr", 0.3)),
        "hall_clamp": float(train_cfg.get("hallucinate_clamp_std", 3.0)),
        "hall_node_fraction": float(train_cfg.get("hallucinate_node_fraction", 0.5)),
        "hall_node_min": int(train_cfg.get("hallucinate_node_min", 20)),
        "timing_warmup_epochs": int(bench_cfg.get("timing_warmup_epochs", 1)),
    }

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    for mode in modes:
        if mode == "ff_layerwise":
            res = _benchmark_ff(graphs, device, config, layerwise=True)
        elif mode == "ff_e2e":
            res = _benchmark_ff(graphs, device, config, layerwise=False)
        elif mode == "backprop":
            res = _benchmark_backprop(graphs, device, config)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        res["mode"] = mode
        results.append(res)

    out_path = Path(bench_cfg.get("out_csv", "reports/benchmark.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    keys = sorted({k for r in results for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"Wrote {out_path}")
    for r in results:
        print(r)

    plot_path = bench_cfg.get("plot_path", "reports/benchmark_speed_sep.png")
    try:
        import matplotlib.pyplot as plt

        xs = [r["graphs_per_s"] for r in results]
        ys = [r.get("eval_sep", 0.0) for r in results]
        labels = [r["mode"] for r in results]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys, color="#4C78A8")
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("graphs/sec")
        ax.set_ylabel("eval_sep (g_pos - g_neg)")
        ax.set_title("Speed vs Separation")
        fig.tight_layout()
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
