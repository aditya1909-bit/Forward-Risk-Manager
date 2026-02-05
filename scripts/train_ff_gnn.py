#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import tomllib
import os

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import GraphSAGEEncoder
from frisk.ff import goodness, make_negative, ff_loss
from frisk.hallucinate import HallucinationConfig, hallucinate_negative


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _get_setting(args: argparse.Namespace, section: dict, key: str, default):
    if hasattr(args, key):
        return getattr(args, key)
    if key in section:
        return section[key]
    return default


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg


def _try_batch_size(
    graphs,
    model,
    device,
    batch_size,
    loader_workers,
    neg_mode,
    noise_std,
    goodness_target,
    hall_cfg: HallucinationConfig,
):
    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=loader_workers,
    )
    batch = next(iter(loader))
    batch = batch.to(device)
    x = batch.x
    h_pos = model(x, batch.edge_index)
    g_pos = goodness(h_pos, batch.batch)
    if neg_mode == "hallucinate":
        x_neg = hallucinate_negative(
            model,
            x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            batch.batch,
            hall_cfg,
        )
    else:
        x_neg = make_negative(x, batch.batch, mode=neg_mode, noise_std=noise_std)
    h_neg = model(x_neg, batch.edge_index)
    g_neg = goodness(h_neg, batch.batch)
    loss = ff_loss(g_pos, g_neg, target=goodness_target)
    loss.backward()
    model.zero_grad(set_to_none=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Forward-Forward GNN on rolling correlation graphs.")
    parser.add_argument("--config", help="Path to TOML config")
    parser.add_argument("--graphs", help="Path to graphs.pt from build_graphs.py", default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--hidden-dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-layers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--goodness-target", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--neg-mode",
        choices=["shuffle", "noise", "shuffle+noise", "hallucinate", "schedule", "mix"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--noise-std", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--loader-workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--auto-tune-batch", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--auto-tune-max-batch", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--auto-tune-factor", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--auto-tune-min-batch", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--neg-warmup-epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--neg-mix-start", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--neg-mix-end", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--neg-mix-ramp-epochs", type=int, default=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    section = cfg.get("train", {})

    graphs_path = _get_setting(args, section, "graphs", None)
    if not graphs_path:
        raise ValueError("Provide --graphs (or set it in config).")

    epochs = _get_setting(args, section, "epochs", 10)
    batch_size = _get_setting(args, section, "batch_size", 8)
    lr = _get_setting(args, section, "lr", 1e-3)
    hidden_dim = _get_setting(args, section, "hidden_dim", 64)
    num_layers = _get_setting(args, section, "num_layers", 2)
    dropout = _get_setting(args, section, "dropout", 0.1)
    goodness_target = _get_setting(args, section, "goodness_target", 1.0)
    neg_mode = _get_setting(args, section, "neg_mode", "shuffle")
    noise_std = _get_setting(args, section, "noise_std", 0.05)
    device_choice = _get_setting(args, section, "device", "cpu")
    seed = _get_setting(args, section, "seed", 7)
    loader_workers = _get_setting(args, section, "loader_workers", 0)
    log_csv = _get_setting(args, section, "log_csv", "")
    plot_path = _get_setting(args, section, "plot_path", "")
    save_model = _get_setting(args, section, "save_model", "")
    auto_tune = _get_setting(args, section, "auto_tune_batch", False)
    auto_tune_max = _get_setting(args, section, "auto_tune_max_batch", 64)
    auto_tune_factor = _get_setting(args, section, "auto_tune_factor", 2)
    auto_tune_min = _get_setting(args, section, "auto_tune_min_batch", 1)
    neg_warmup_epochs = _get_setting(args, section, "neg_warmup_epochs", 0)
    neg_mix_start = _get_setting(args, section, "neg_mix_start", 0.0)
    neg_mix_end = _get_setting(args, section, "neg_mix_end", 0.7)
    neg_mix_ramp_epochs = _get_setting(args, section, "neg_mix_ramp_epochs", 10)

    hall_steps = _get_setting(args, section, "hallucinate_steps", 10)
    hall_lr = _get_setting(args, section, "hallucinate_lr", 0.1)
    hall_l2 = _get_setting(args, section, "hallucinate_l2", 0.1)
    hall_mean = _get_setting(args, section, "hallucinate_mean", 0.05)
    hall_std = _get_setting(args, section, "hallucinate_std", 0.05)
    hall_corr = _get_setting(args, section, "hallucinate_corr", 1.0)
    hall_clamp = _get_setting(args, section, "hallucinate_clamp_std", 3.0)

    set_seed(seed)
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    elif device_choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        payload = torch.load(Path(graphs_path), map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch versions don't support weights_only
        payload = torch.load(Path(graphs_path), map_location="cpu")
    graphs = payload["graphs"]
    if not graphs:
        raise ValueError("No graphs found in the provided file.")

    print(f"device: {device}")
    print(f"mps built: {torch.backends.mps.is_built()}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    fallback = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "1") != "0"
    print(f"mps fallback enabled: {fallback}")
    print(
        f"neg_mode: {neg_mode} | batch_size: {batch_size} | loader_workers: {loader_workers}"
    )
    if neg_mode in ("schedule", "mix"):
        print(f"neg_warmup_epochs: {neg_warmup_epochs}")
    if neg_mode == "mix":
        print(
            f"neg_mix_start: {neg_mix_start} | neg_mix_end: {neg_mix_end} | "
            f"neg_mix_ramp_epochs: {neg_mix_ramp_epochs}"
        )
    print(
        "auto_tune_batch: "
        f"{auto_tune} (max={auto_tune_max}, factor={auto_tune_factor}, min={auto_tune_min})"
    )

    input_dim = graphs[0].x.shape[1]
    model = GraphSAGEEncoder(
        in_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    hall_cfg = HallucinationConfig(
        steps=hall_steps,
        lr=hall_lr,
        l2_weight=hall_l2,
        mean_weight=hall_mean,
        std_weight=hall_std,
        corr_weight=hall_corr,
        clamp_std=hall_clamp,
    )

    if auto_tune and device.type == "mps":
        print("Auto-tuning batch size for MPS...")
        test_bs = batch_size
        best_bs = None
        while test_bs <= auto_tune_max:
            try:
                model.train()
                _try_batch_size(
                    graphs,
                    model,
                    device,
                    test_bs,
                    loader_workers,
                    neg_mode,
                    noise_std,
                    goodness_target,
                    hall_cfg,
                )
                best_bs = test_bs
                test_bs = int(test_bs * auto_tune_factor)
                if test_bs == best_bs:
                    break
            except RuntimeError as exc:
                if _is_oom(exc):
                    break
                raise
            finally:
                if device.type == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass

        if best_bs is None:
            test_bs = max(auto_tune_min, int(batch_size / auto_tune_factor))
            while test_bs >= auto_tune_min:
                try:
                    model.train()
                    _try_batch_size(
                        graphs,
                        model,
                        device,
                        test_bs,
                        loader_workers,
                        neg_mode,
                        noise_std,
                        goodness_target,
                        hall_cfg,
                    )
                    best_bs = test_bs
                    break
                except RuntimeError as exc:
                    if _is_oom(exc):
                        test_bs = int(test_bs / auto_tune_factor)
                        continue
                    raise
                finally:
                    if device.type == "mps":
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass

        if best_bs is not None and best_bs != batch_size:
            print(f"Auto-tune selected batch_size={best_bs}")
            batch_size = best_bs

    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=loader_workers,
    )
    optim = Adam(model.parameters(), lr=lr)

    if log_csv:
        log_path = Path(log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            f.write("epoch,loss,g_pos,g_neg\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        batches = 0

        for batch in loader:
            try:
                batch = batch.to(device)
            except Exception as exc:
                if device.type == "mps":
                    raise RuntimeError(
                        "MPS device placement failed for PyG tensors. "
                        "If you hit unsupported ops, rerun with --device cpu."
                    ) from exc
                raise
            x = batch.x
            h_pos = model(x, batch.edge_index)
            g_pos = goodness(h_pos, batch.batch)

            if neg_mode == "schedule":
                use_mode = "shuffle" if epoch <= neg_warmup_epochs else "hallucinate"
            elif neg_mode == "mix":
                if epoch <= neg_warmup_epochs:
                    use_mode = "shuffle"
                else:
                    ramp = max(1, neg_mix_ramp_epochs)
                    progress = min(1.0, (epoch - neg_warmup_epochs) / ramp)
                    p_hall = neg_mix_start + progress * (neg_mix_end - neg_mix_start)
                    use_mode = "hallucinate" if random.random() < p_hall else "shuffle"
            else:
                use_mode = neg_mode

            if use_mode == "hallucinate":
                x_neg = hallucinate_negative(
                    model,
                    x,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    batch.batch,
                    hall_cfg,
                )
            else:
                x_neg = make_negative(x, batch.batch, mode=use_mode, noise_std=noise_std)
            h_neg = model(x_neg, batch.edge_index)
            g_neg = goodness(h_neg, batch.batch)

            loss = ff_loss(g_pos, g_neg, target=goodness_target)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_pos += g_pos.mean().item()
            total_neg += g_neg.mean().item()
            batches += 1

        print(
            f"Epoch {epoch:02d} | loss={total_loss / batches:.4f} "
            f"g_pos={total_pos / batches:.4f} g_neg={total_neg / batches:.4f}"
        )

        if log_csv:
            with Path(log_csv).open("a") as f:
                f.write(
                    f"{epoch},{total_loss / batches:.6f},"
                    f"{total_pos / batches:.6f},{total_neg / batches:.6f}\n"
                )

    if save_model:
        save_path = Path(save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    if log_csv and plot_path:
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.read_csv(log_csv)
            plt.figure(figsize=(8, 5))
            plt.plot(df["epoch"], df["loss"], label="loss")
            plt.plot(df["epoch"], df["g_pos"], label="g_pos")
            plt.plot(df["epoch"], df["g_neg"], label="g_neg")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plot_path = Path(plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
        except Exception as exc:
            print(f"Plotting failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
