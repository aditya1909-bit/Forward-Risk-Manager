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

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import GCNEncoder
from frisk.ff import goodness, make_negative, ff_loss

_THREAD_STATE = {"threads": None, "interop": None}
_THREAD_WARNED = False


def _maybe_set_torch_threads(threads: int | None, interop: int | None) -> None:
    global _THREAD_STATE, _THREAD_WARNED
    if interop:
        if _THREAD_STATE["interop"] is None:
            try:
                current = torch.get_num_interop_threads()
                if current == int(interop):
                    _THREAD_STATE["interop"] = int(interop)
                else:
                    torch.set_num_interop_threads(int(interop))
                    _THREAD_STATE["interop"] = int(interop)
            except RuntimeError as exc:
                if not _THREAD_WARNED:
                    print(f"Warning: cannot set torch_num_interop_threads ({exc})")
                    _THREAD_WARNED = True
        elif _THREAD_STATE["interop"] != int(interop) and not _THREAD_WARNED:
            print(
                "Warning: torch_num_interop_threads cannot be changed after set; "
                f"keeping {_THREAD_STATE['interop']}, ignoring {interop}"
            )
            _THREAD_WARNED = True
    if threads:
        if _THREAD_STATE["threads"] is None:
            try:
                current = torch.get_num_threads()
                if current == int(threads):
                    _THREAD_STATE["threads"] = int(threads)
                else:
                    torch.set_num_threads(int(threads))
                    _THREAD_STATE["threads"] = int(threads)
            except RuntimeError as exc:
                if not _THREAD_WARNED:
                    print(f"Warning: cannot set torch_num_threads ({exc})")
                    _THREAD_WARNED = True
            _THREAD_STATE["threads"] = int(threads)
        elif _THREAD_STATE["threads"] != int(threads) and not _THREAD_WARNED:
            print(
                "Warning: torch_num_threads cannot be changed after set; "
                f"keeping {_THREAD_STATE['threads']}, ignoring {threads}"
            )
            _THREAD_WARNED = True


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


def _trial_worker(args):
    (
        graphs_path,
        cfg,
        seed,
        sample_graphs,
        max_batches,
        worker_threads,
        worker_interop_threads,
        loader_workers,
        device_str,
    ) = args
    _maybe_set_torch_threads(worker_threads, worker_interop_threads)
    _set_seed(seed)
    device = _choose_device(device_str)
    try:
        payload = torch.load(Path(graphs_path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(graphs_path), map_location="cpu")
    graphs = payload["graphs"]
    if sample_graphs and sample_graphs < len(graphs):
        graphs = graphs[:sample_graphs]

    loader = DataLoader(
        graphs,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=loader_workers,
    )
    model = GCNEncoder(
        in_dim=graphs[0].x.shape[1],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    optim = Adam(model.parameters(), lr=cfg["lr"])

    graphs_seen = 0
    t0 = time.perf_counter()
    for _ in range(cfg["epochs"]):
        model.train()
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            batch = batch.to(device)
            x = batch.x
            edge_weight = getattr(batch, "edge_weight", None)
            h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
            g_pos = goodness(h_pos, batch.batch, temperature=cfg["goodness_temp"])
            x_neg = make_negative(x, batch.batch, mode=cfg["neg_mode"], noise_std=cfg["noise_std"])
            h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
            g_neg = goodness(h_neg, batch.batch, temperature=cfg["goodness_temp"])
            loss = ff_loss(g_pos, g_neg, target=cfg["goodness_target"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            graphs_seen += batch.num_graphs
    _sync(device)
    dt = time.perf_counter() - t0
    return graphs_seen, dt


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-tune sweep parallelism settings.")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--apply", action="store_true", help="Write best settings back to config")
    parser.add_argument("--apply-to", default="", help="Config path to update (defaults to --config)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {})
    tune_cfg = cfg.get("sweep_tune", {})

    graphs_path = Path(train_cfg.get("graphs", "data/processed/graphs.pt"))
    device_str = str(tune_cfg.get("device", "cpu"))

    base = {
        "epochs": int(tune_cfg.get("epochs", 1)),
        "batch_size": int(tune_cfg.get("batch_size", train_cfg.get("batch_size", 16))),
        "hidden_dim": int(train_cfg.get("hidden_dim", 64)),
        "num_layers": int(train_cfg.get("num_layers", 2)),
        "dropout": float(train_cfg.get("dropout", 0.1)),
        "lr": float(train_cfg.get("lr", 1e-3)),
        "goodness_temp": float(train_cfg.get("goodness_temp", 1.0)),
        "goodness_target": float(train_cfg.get("goodness_target", 1.0)),
        "neg_mode": str(tune_cfg.get("neg_mode", "shuffle")),
        "noise_std": float(train_cfg.get("noise_std", 0.05)),
    }

    sample_graphs = int(tune_cfg.get("sample_graphs", 64))
    max_batches = int(tune_cfg.get("max_batches", 2))
    parallel_backend = str(tune_cfg.get("parallel_backend", "process"))
    mp_context = str(tune_cfg.get("parallel_mp_context", "spawn"))
    isolate_thread_settings = bool(tune_cfg.get("isolate_thread_settings", True))

    workers_list = tune_cfg.get("parallel_workers", [1, 2, 3, 4])
    torch_threads_list = tune_cfg.get("worker_torch_threads", [1])
    torch_interop_list = tune_cfg.get("worker_torch_interop_threads", [1])
    loader_workers_list = tune_cfg.get("worker_loader_workers", [0])

    combos = list(itertools.product(workers_list, torch_threads_list, torch_interop_list, loader_workers_list))
    results = []
    unique_interop = {int(ti) for _, _, ti, _ in combos}
    unique_threads = {int(tt) for _, tt, _, _ in combos}
    if len(unique_interop) > 1 or len(unique_threads) > 1:
        if int(tune_cfg.get("parallel_workers", [1])[0]) <= 1 and parallel_backend in ("thread", "threads"):
            print(
                "Warning: varying thread/interop settings in a single process is not supported; "
                "results may be inaccurate. Use process backend or set a single value."
            )

    for pw, tt, ti, lw in combos:
        if int(pw) <= 1:
            # serial (optionally isolate per combo in a subprocess to allow different thread settings)
            task = (
                str(graphs_path),
                base,
                7,
                sample_graphs,
                max_batches,
                int(tt),
                int(ti),
                int(lw),
                device_str,
            )
            if isolate_thread_settings and (len(unique_threads) > 1 or len(unique_interop) > 1):
                from concurrent.futures import ProcessPoolExecutor
                import multiprocessing as mp

                ctx = mp.get_context(mp_context)
                with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
                    graphs_seen, dt = list(ex.map(_trial_worker, [task]))[0]
            else:
                _set_seed(7)
                graphs_seen, dt = _trial_worker(task)
            results.append(
                {
                    "parallel_workers": int(pw),
                    "worker_torch_threads": int(tt),
                    "worker_torch_interop_threads": int(ti),
                    "worker_loader_workers": int(lw),
                    "wall_time_s": dt,
                    "graphs_seen": graphs_seen,
                    "graphs_per_s": graphs_seen / dt if dt > 0 else 0.0,
                }
            )
            continue

        tasks = []
        for wi in range(int(pw)):
            tasks.append(
                (
                    str(graphs_path),
                    base,
                    7 + wi,
                    sample_graphs,
                    max_batches,
                    int(tt),
                    int(ti),
                    int(lw),
                    device_str,
                )
            )

        t0 = time.perf_counter()
        if parallel_backend in ("thread", "threads"):
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=int(pw)) as ex:
                results_worker = list(ex.map(_trial_worker, tasks))
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp

            ctx = mp.get_context(mp_context)
            # group by thread/interop so each pool uses a fixed setting per worker
            groups = {}
            for task in tasks:
                key = (task[5], task[6])  # worker_threads, worker_interop_threads
                groups.setdefault(key, []).append(task)
            results_worker = []
            for key, group_tasks in groups.items():
                with ProcessPoolExecutor(max_workers=int(pw), mp_context=ctx) as ex:
                    results_worker.extend(list(ex.map(_trial_worker, group_tasks)))
        wall = time.perf_counter() - t0
        graphs_seen = int(sum(g for g, _ in results_worker))
        results.append(
            {
                "parallel_workers": int(pw),
                "worker_torch_threads": int(tt),
                "worker_torch_interop_threads": int(ti),
                "worker_loader_workers": int(lw),
                "wall_time_s": wall,
                "graphs_seen": graphs_seen,
                "graphs_per_s": graphs_seen / wall if wall > 0 else 0.0,
            }
        )

    out_path = Path(tune_cfg.get("out_csv", "reports/sweep_parallel_tune.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    keys = sorted({k for r in results for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    best = max(results, key=lambda r: r.get("graphs_per_s", 0.0))
    print(f"Wrote {out_path}")
    print(f"Best by graphs_per_s: {best}")

    apply_flag = bool(tune_cfg.get("apply", False) or args.apply)
    apply_to = args.apply_to or str(tune_cfg.get("apply_to", "")) or args.config
    apply_section = str(tune_cfg.get("apply_section", "sweep"))
    min_improve = float(tune_cfg.get("apply_min_improvement", 0.0))
    apply_backup = bool(tune_cfg.get("apply_backup", False))
    backup_suffix = str(tune_cfg.get("apply_backup_suffix", ".bak"))
    if apply_flag:
        baseline = None
        for r in results:
            if int(r.get("parallel_workers", 0)) == 1:
                baseline = max(baseline or 0.0, r.get("graphs_per_s", 0.0))
        if baseline is None:
            baseline = best.get("graphs_per_s", 0.0)
        improve = (
            (best.get("graphs_per_s", 0.0) - baseline) / baseline
            if baseline and baseline > 0
            else 0.0
        )
        if improve < min_improve:
            print(
                f"Skip apply: improvement {improve:.2%} < min {min_improve:.2%} "
                f"(baseline={baseline:.4f}, best={best.get('graphs_per_s', 0.0):.4f})"
            )
            return 0
        if apply_backup:
            _backup_config(Path(apply_to), backup_suffix)
        _apply_to_config(
            Path(apply_to),
            apply_section,
            {
                "parallel_workers": best["parallel_workers"],
                "worker_torch_threads": best["worker_torch_threads"],
                "worker_torch_interop_threads": best["worker_torch_interop_threads"],
                "worker_loader_workers": best["worker_loader_workers"],
            },
        )
        print(f"Applied best settings to {apply_to} [{apply_section}]")
    return 0


def _backup_config(path: Path, suffix: str) -> None:
    import time

    ts = time.strftime("%Y%m%d%H%M%S")
    backup = Path(f"{path}{suffix}.{ts}")
    backup.write_text(path.read_text())
    print(f"Wrote backup {backup}")


def _apply_to_config(path: Path, section: str, updates: dict) -> None:
    def _format_value(val):
        if isinstance(val, str):
            return f"\"{val}\""
        if isinstance(val, bool):
            return "true" if val else "false"
        return str(val)

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


if __name__ == "__main__":
    raise SystemExit(main())
