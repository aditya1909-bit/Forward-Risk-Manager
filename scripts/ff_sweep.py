#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import random
import time
from pathlib import Path
import sys
import tomllib

import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

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


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _split_graphs(graphs, eval_frac: float, seed: int):
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
    forward_fn=None,
    window_len: int | None = None,
    summary_dim: int = 0,
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
            forward_fn=forward_fn,
        )
    return make_negative(
        x,
        batch,
        mode=use_mode,
        noise_std=noise_std,
        window_len=window_len,
        summary_dim=summary_dim,
    )


def _eval_ff_metrics(
    model,
    loader,
    goodness_temp,
    goodness_target,
    neg_mode,
    noise_std,
    hall_cfg,
    window_len: int | None = None,
    summary_dim: int = 0,
):
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
                window_len=window_len,
                summary_dim=summary_dim,
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


def _run_ff_trial(graphs, device, cfg, layerwise: bool):
    train_graphs, eval_graphs = _split_graphs(graphs, cfg["eval_frac"], cfg["seed"])
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "shuffle": True,
        "drop_last": False,
        "num_workers": cfg["loader_workers"],
        "pin_memory": bool(cfg.get("pin_memory", False)) if device.type == "cuda" else False,
    }
    if cfg["loader_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        mp_ctx = cfg.get("multiprocessing_context", "")
        if mp_ctx:
            loader_kwargs["multiprocessing_context"] = mp_ctx
    loader = DataLoader(train_graphs, **loader_kwargs)
    eval_loader = DataLoader(eval_graphs, batch_size=cfg["batch_size"], shuffle=False)

    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    optim = Adam(model.parameters(), lr=cfg["lr"])

    hall_cfg = HallucinationConfig(
        steps=cfg["hall_steps"],
        lr=cfg["hall_lr"],
        l2_weight=cfg["hall_l2"],
        mean_weight=cfg["hall_mean"],
        std_weight=cfg["hall_std"],
        corr_weight=cfg["hall_corr"],
        clamp_std=cfg["hall_clamp"],
        goodness_temp=cfg["goodness_temp"],
        node_fraction=cfg["hall_node_fraction"],
        node_min=cfg["hall_node_min"],
    )
    hall_cfg_layer = HallucinationConfig(
        steps=cfg["hall_steps"],
        lr=cfg["hall_lr"],
        l2_weight=cfg["hall_l2"],
        mean_weight=cfg["layerwise_hall_mean"],
        std_weight=cfg["layerwise_hall_std"],
        corr_weight=cfg["layerwise_hall_corr"],
        clamp_std=cfg["hall_clamp"],
        goodness_temp=cfg["goodness_temp"],
        node_fraction=cfg["hall_node_fraction"],
        node_min=cfg["hall_node_min"],
    )

    epoch_times = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        graphs_seen = 0
        for batch in loader:
            batch = batch.to(device)
            x = batch.x
            edge_weight = getattr(batch, "edge_weight", None)

            use_mode = _get_use_mode(
                epoch,
                cfg["neg_mode"],
                cfg["neg_warmup_epochs"],
                cfg["neg_mix_start"],
                cfg["neg_mix_end"],
                cfg["neg_mix_ramp_epochs"],
            )

            if layerwise:
                x_in = x
                for li in range(len(model.layers)):
                    layer_mode = use_mode
                    if use_mode == "hallucinate" and li > 0:
                        layer_mode = "shuffle"
                    h_pos = model.forward_layer(x_in, batch.edge_index, edge_weight, li)
                    g_pos = goodness(h_pos, batch.batch, temperature=cfg["goodness_temp"])

                    if layer_mode == "hallucinate":
                        forward_fn = lambda x_var, li=li: model.forward_layer(
                            x_var, batch.edge_index, edge_weight, li
                        )
                    x_neg = _make_negatives(
                        model,
                        x_in,
                        batch.batch,
                        batch.edge_index,
                        getattr(batch, "edge_attr", None),
                        edge_weight,
                        layer_mode,
                        cfg["noise_std"],
                        hall_cfg_layer,
                        forward_fn=forward_fn,
                        window_len=cfg.get("window_len"),
                        summary_dim=cfg.get("summary_dim", 0),
                    )
                    else:
                    x_neg = _make_negatives(
                        model,
                        x_in,
                        batch.batch,
                        batch.edge_index,
                        getattr(batch, "edge_attr", None),
                        edge_weight,
                        cfg["layerwise_neg_mode"],
                        cfg["layerwise_noise_std"],
                        hall_cfg,
                        window_len=cfg.get("window_len"),
                        summary_dim=cfg.get("summary_dim", 0),
                    )

                    if layer_mode == "hallucinate":
                        h_neg_probe = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                        g_neg_probe = goodness(
                            h_neg_probe, batch.batch, temperature=cfg["goodness_temp"]
                        ).mean().item()
                        g_pos_probe = g_pos.mean().item()
                        if g_neg_probe > g_pos_probe + cfg["neg_gate_margin"]:
                            x_neg = make_negative(x_in, batch.batch, mode="shuffle", noise_std=cfg["noise_std"])

                    h_neg = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                    g_neg = goodness(h_neg, batch.batch, temperature=cfg["goodness_temp"])
                    loss = ff_loss(g_pos, g_neg, target=cfg["goodness_target"])

                    optim.zero_grad()
                    loss.backward()
                    if cfg["grad_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optim.step()

                    x_in = h_pos.detach()
            else:
                h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                g_pos = goodness(h_pos, batch.batch, temperature=cfg["goodness_temp"])
                x_neg = _make_negatives(
                    model,
                    x,
                    batch.batch,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    edge_weight,
                    use_mode,
                    cfg["noise_std"],
                    hall_cfg,
                    window_len=cfg.get("window_len"),
                    summary_dim=cfg.get("summary_dim", 0),
                )

                if use_mode == "hallucinate":
                    h_neg_probe = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                    g_neg_probe = goodness(
                        h_neg_probe, batch.batch, temperature=cfg["goodness_temp"]
                    ).mean().item()
                    g_pos_probe = g_pos.mean().item()
                    if g_neg_probe > g_pos_probe + cfg["neg_gate_margin"]:
                        x_neg = make_negative(x, batch.batch, mode="shuffle", noise_std=cfg["noise_std"])

                h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                g_neg = goodness(h_neg, batch.batch, temperature=cfg["goodness_temp"])
                loss = ff_loss(g_pos, g_neg, target=cfg["goodness_target"])

                optim.zero_grad()
                loss.backward()
                if cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optim.step()

            graphs_seen += batch.num_graphs

        _sync(device)
        dt = time.perf_counter() - t0
        epoch_times.append((dt, graphs_seen))

    gpos, gneg, acc = _eval_ff_metrics(
        model,
        eval_loader,
        cfg["goodness_temp"],
        cfg["goodness_target"],
        cfg["eval_neg_mode"],
        cfg["noise_std"],
        hall_cfg,
        window_len=cfg.get("window_len"),
        summary_dim=cfg.get("summary_dim", 0),
    )
    warm = int(cfg.get("timing_warmup_epochs", 0))
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


def _run_trial_worker(args):
    (
        graphs_path,
        cfg,
        combo,
        layerwise,
        device_str,
        seed,
        worker_threads,
        worker_interop_threads,
    ) = args
    if worker_threads:
        torch.set_num_threads(int(worker_threads))
    if worker_interop_threads:
        torch.set_num_interop_threads(int(worker_interop_threads))
    _set_seed(seed)
    device = _choose_device(device_str)
    try:
        payload = torch.load(Path(graphs_path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(graphs_path), map_location="cpu")
    graphs = payload["graphs"]
    return _run_ff_trial(graphs, device, cfg, layerwise=layerwise)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep FF hyperparams and rank by eval_sep.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument(
        "--section",
        default="sweep",
        help="Config section to use (default: sweep, e.g., sweep_layerwise)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})
    sweep_cfg = cfg.get(args.section, {})
    build_cfg = cfg.get("build_graphs", {})

    graphs_path = Path(train_cfg.get("graphs", "data/processed/graphs.pt"))
    device_str = str(train_cfg.get("device", "cpu"))

    neg_mode_val = sweep_cfg.get("neg_mode", train_cfg.get("neg_mode", "shuffle"))
    if isinstance(neg_mode_val, list):
        neg_mode_val = neg_mode_val[0] if neg_mode_val else "shuffle"

    feature_mode = build_cfg.get("feature_mode", "window")
    window_len = int(build_cfg.get("window", 20))
    returns_len = window_len if feature_mode in ("window", "window_plus_summary") else 1
    summary_dim = 5 if feature_mode == "window_plus_summary" else 0

    base = {
        "epochs": int(sweep_cfg.get("epochs", 3)),
        "batch_size": int(sweep_cfg.get("batch_size", train_cfg.get("batch_size", 16))),
        "hidden_dim": int(train_cfg.get("hidden_dim", 64)),
        "num_layers": int(train_cfg.get("num_layers", 2)),
        "dropout": float(train_cfg.get("dropout", 0.1)),
        "lr": float(train_cfg.get("lr", 1e-3)),
        "neg_mode": str(neg_mode_val),
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
        "eval_frac": float(sweep_cfg.get("eval_frac", 0.2)),
        "seed": int(sweep_cfg.get("seed", train_cfg.get("seed", 7))),
        "hall_steps": int(train_cfg.get("hallucinate_steps", 3)),
        "hall_lr": float(train_cfg.get("hallucinate_lr", 0.03)),
        "hall_l2": float(train_cfg.get("hallucinate_l2", 0.05)),
        "hall_mean": float(train_cfg.get("hallucinate_mean", 0.01)),
        "hall_std": float(train_cfg.get("hallucinate_std", 0.01)),
        "hall_corr": float(train_cfg.get("hallucinate_corr", 0.3)),
        "hall_clamp": float(train_cfg.get("hallucinate_clamp_std", 3.0)),
        "hall_node_fraction": float(train_cfg.get("hallucinate_node_fraction", 0.5)),
        "hall_node_min": int(train_cfg.get("hallucinate_node_min", 20)),
        "neg_gate_margin": float(train_cfg.get("neg_gate_margin", 1.0)),
        "eval_neg_mode": str(sweep_cfg.get("eval_neg_mode", "shuffle")),
        "timing_warmup_epochs": int(sweep_cfg.get("timing_warmup_epochs", 1)),
        "layerwise_neg_mode": str(train_cfg.get("layerwise_neg_mode", "shuffle")),
        "layerwise_noise_std": float(train_cfg.get("layerwise_noise_std", train_cfg.get("noise_std", 0.05))),
        "layerwise_hall_corr": float(train_cfg.get("layerwise_hall_corr", 0.0)),
        "layerwise_hall_mean": float(train_cfg.get("layerwise_hall_mean", train_cfg.get("hallucinate_mean", 0.01))),
        "layerwise_hall_std": float(train_cfg.get("layerwise_hall_std", train_cfg.get("hallucinate_std", 0.01))),
        "window_len": int(returns_len),
        "summary_dim": int(summary_dim),
    }

    modes = sweep_cfg.get("modes", ["ff_layerwise", "ff_e2e"])
    if isinstance(modes, str):
        modes = [m.strip() for m in modes.split(",") if m.strip()]

    meta_keys = {
        "epochs",
        "batch_size",
        "eval_frac",
        "out_csv",
        "modes",
        "seed",
        "max_runs",
        "timing_warmup_epochs",
        "eval_neg_mode",
    }

    grid_keys = []
    grid_vals = []
    for k, v in sweep_cfg.items():
        if k in meta_keys:
            continue
        if isinstance(v, list):
            grid_keys.append(k)
            grid_vals.append(v)
        else:
            base[k] = v

    combos = [dict(zip(grid_keys, vals)) for vals in itertools.product(*grid_vals)] if grid_keys else [{}]
    max_runs = sweep_cfg.get("max_runs", None)
    if max_runs is not None:
        combos = combos[: int(max_runs)]

    parallel_workers = int(sweep_cfg.get("parallel_workers", 1))
    parallel_backend = str(sweep_cfg.get("parallel_backend", "process")).lower()
    parallel_mp_context = str(sweep_cfg.get("parallel_mp_context", "spawn"))
    parallel_force_cpu = bool(sweep_cfg.get("parallel_force_cpu", True))
    worker_threads = int(sweep_cfg.get("worker_torch_threads", 1 if parallel_workers > 1 else 0))
    worker_interop = int(sweep_cfg.get("worker_torch_interop_threads", 1 if parallel_workers > 1 else 0))
    worker_loader_workers = int(sweep_cfg.get("worker_loader_workers", base["loader_workers"]))

    if parallel_workers > 1 and device_str != "cpu":
        if parallel_force_cpu:
            print(f"Parallel sweep forcing device=cpu (was {device_str})")
            device_str = "cpu"
        else:
            print(f"Parallel sweep disabled on device={device_str}; using serial execution.")
            parallel_workers = 1

    if parallel_workers > 1:
        print(
            f"Running sweep in parallel: workers={parallel_workers}, backend={parallel_backend}, "
            f"mp_context={parallel_mp_context}"
        )
        base["loader_workers"] = worker_loader_workers

    if parallel_workers <= 1:
        # serial execution; load graphs once
        try:
            payload = torch.load(graphs_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(graphs_path, map_location="cpu")
        graphs = payload["graphs"]
        if not graphs:
            raise ValueError("No graphs found in the provided file.")
        device = _choose_device(device_str)
        if train_cfg.get("torch_num_threads"):
            torch.set_num_threads(int(train_cfg["torch_num_threads"]))
        if train_cfg.get("torch_num_interop_threads"):
            torch.set_num_interop_threads(int(train_cfg["torch_num_interop_threads"]))

    results = []
    run_idx = 0
    tasks = []
    total_trials = len(combos) * len(modes)
    pbar = tqdm(
        total=total_trials,
        desc="Sweep",
        unit="trial",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for combo in combos:
        cfg_run = base.copy()
        cfg_run.update(combo)
        for mode in modes:
            run_idx += 1
            layerwise = mode == "ff_layerwise"
            seed = int(cfg_run.get("seed", 7)) + run_idx
            if parallel_workers > 1:
                tasks.append(
                    (
                        str(graphs_path),
                        cfg_run,
                        combo,
                        layerwise,
                        device_str,
                        seed,
                        worker_threads,
                        worker_interop,
                    )
                )
            else:
                _set_seed(seed)
                res = _run_ff_trial(graphs, device, cfg_run, layerwise=layerwise)
                res["mode"] = mode
                res.update(combo)
                for k in (
                    "neg_mode",
                    "goodness_temp",
                    "goodness_target",
                    "neg_mix_end",
                    "hall_steps",
                    "hall_lr",
                    "hall_node_fraction",
                    "layerwise_neg_mode",
                    "layerwise_noise_std",
                    "layerwise_hall_corr",
                    "layerwise_hall_mean",
                    "layerwise_hall_std",
                ):
                    if k in cfg_run:
                        res[k] = cfg_run[k]
                results.append(res)
                pbar.update(1)

    if parallel_workers > 1 and tasks:
        if parallel_backend not in ("process", "thread", "threads"):
            raise ValueError(f"Unknown parallel_backend: {parallel_backend}")
        if parallel_backend in ("thread", "threads"):
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
                for res, task in zip(ex.map(_run_trial_worker, tasks), tasks):
                    _, cfg_run, combo, layerwise, *_ = task
                    res["mode"] = "ff_layerwise" if layerwise else "ff_e2e"
                    res.update(combo)
                    for k in (
                        "neg_mode",
                        "goodness_temp",
                        "goodness_target",
                        "neg_mix_end",
                        "hall_steps",
                        "hall_lr",
                        "hall_node_fraction",
                        "layerwise_neg_mode",
                        "layerwise_noise_std",
                        "layerwise_hall_corr",
                        "layerwise_hall_mean",
                        "layerwise_hall_std",
                    ):
                        if k in cfg_run:
                            res[k] = cfg_run[k]
                    results.append(res)
                    pbar.update(1)
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp

            ctx = mp.get_context(parallel_mp_context)
            with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=ctx) as ex:
                for res, task in zip(ex.map(_run_trial_worker, tasks), tasks):
                    _, cfg_run, combo, layerwise, *_ = task
                    res["mode"] = "ff_layerwise" if layerwise else "ff_e2e"
                    res.update(combo)
                    for k in (
                        "neg_mode",
                        "goodness_temp",
                        "goodness_target",
                        "neg_mix_end",
                        "hall_steps",
                        "hall_lr",
                        "hall_node_fraction",
                        "layerwise_neg_mode",
                        "layerwise_noise_std",
                        "layerwise_hall_corr",
                        "layerwise_hall_mean",
                        "layerwise_hall_std",
                    ):
                        if k in cfg_run:
                            res[k] = cfg_run[k]
                    results.append(res)
                    pbar.update(1)
    pbar.close()

    out_path = Path(sweep_cfg.get("out_csv", "reports/ff_sweep.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    keys = sorted({k for r in results for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    if results:
        best = max(results, key=lambda r: r.get("eval_sep", float("-inf")))
        top_k = int(sweep_cfg.get("top_k", 10))
        ranked = sorted(results, key=lambda r: r.get("eval_sep", float("-inf")), reverse=True)
        print(f"Wrote {out_path}")
        print(f"Best by eval_sep: {best}")
        print(f"Top {top_k} by eval_sep:")
        for r in ranked[:top_k]:
            print(r)
    else:
        print("No sweep results produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
