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

from frisk.models import GCNEncoder
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
    goodness_temp,
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
    edge_weight = getattr(batch, "edge_weight", None)
    h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
    g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)
    if neg_mode == "hallucinate":
        x_neg = hallucinate_negative(
            model,
            x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            batch.batch,
            hall_cfg,
            edge_weight=edge_weight,
        )
    else:
        x_neg = make_negative(x, batch.batch, mode=neg_mode, noise_std=noise_std)
    h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
    g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp)
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
    parser.add_argument("--neg-gate-margin", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--grad-clip", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--temp-sweep", default=argparse.SUPPRESS)
    parser.add_argument("--ff-layerwise", action="store_true", default=argparse.SUPPRESS)
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
    goodness_temp = _get_setting(args, section, "goodness_temp", 1.0)
    temp_sweep = _get_setting(args, section, "temp_sweep", "")
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
    neg_gate_margin = _get_setting(args, section, "neg_gate_margin", 0.1)
    grad_clip = _get_setting(args, section, "grad_clip", 1.0)
    ff_layerwise = _get_setting(args, section, "ff_layerwise", False) or getattr(
        args, "ff_layerwise", False
    )

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
    print(f"ff_layerwise: {ff_layerwise}")
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
    model = GCNEncoder(
        in_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    if temp_sweep:
        temps = [float(t.strip()) for t in str(temp_sweep).split(",") if t.strip()]
        if not temps:
            raise ValueError("temp_sweep provided but no valid values found.")
        print(f"Temp sweep: {temps}")
        loader = DataLoader(
            graphs,
            batch_size=min(batch_size, 32),
            shuffle=True,
            drop_last=False,
            num_workers=loader_workers,
        )
        batch = next(iter(loader)).to(device)
        x = batch.x
        edge_weight = getattr(batch, "edge_weight", None)
        h = model(x, batch.edge_index, edge_weight=edge_weight)
        for t in temps:
            g = goodness(h, batch.batch, temperature=t).mean().item()
            print(f"goodness_temp={t} -> mean_goodness={g:.4f}")
        return 0

    hall_cfg = HallucinationConfig(
        steps=hall_steps,
        lr=hall_lr,
        l2_weight=hall_l2,
        mean_weight=hall_mean,
        std_weight=hall_std,
        corr_weight=hall_corr,
        clamp_std=hall_clamp,
        goodness_temp=goodness_temp,
        node_fraction=_get_setting(args, section, "hallucinate_node_fraction", 1.0),
        node_min=_get_setting(args, section, "hallucinate_node_min", 1),
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
                    goodness_temp,
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
                        goodness_temp,
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
            f.write("epoch,loss,g_pos,g_neg,hallucinate_ratio,gate_ratio\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        batches = 0

        hall_used = 0
        total_used = 0
        hall_gated = 0

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
            edge_weight = getattr(batch, "edge_weight", None)

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

            if ff_layerwise:
                x_in = x
                layer_losses = 0.0
                layer_gpos = 0.0
                layer_gneg = 0.0
                for li in range(len(model.layers)):
                    layer_mode = use_mode
                    if use_mode == "hallucinate" and li > 0:
                        # Hallucination penalties are defined in input space; keep deeper layers shuffle-based.
                        layer_mode = "shuffle"
                    h_pos = model.forward_layer(x_in, batch.edge_index, edge_weight, li)
                    g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)

                    if layer_mode == "hallucinate":
                        forward_fn = lambda x_var, li=li: model.forward_layer(
                            x_var, batch.edge_index, edge_weight, li
                        )
                        x_neg = hallucinate_negative(
                            model,
                            x_in,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            batch.batch,
                            hall_cfg,
                            edge_weight=edge_weight,
                            forward_fn=forward_fn,
                        )
                        hall_used += 1
                    else:
                        x_neg = make_negative(x_in, batch.batch, mode=layer_mode, noise_std=noise_std)
                    total_used += 1

                    if layer_mode == "hallucinate":
                        h_neg_probe = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                        g_neg_probe = goodness(
                            h_neg_probe, batch.batch, temperature=goodness_temp
                        ).mean().item()
                        g_pos_probe = g_pos.mean().item()
                        if g_neg_probe > g_pos_probe + neg_gate_margin:
                            x_neg = make_negative(x_in, batch.batch, mode="shuffle", noise_std=noise_std)
                            hall_used -= 1
                            hall_gated += 1

                    h_neg = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                    g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp)

                    loss = ff_loss(g_pos, g_neg, target=goodness_target)
                    optim.zero_grad()
                    loss.backward()
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optim.step()

                    layer_losses += loss.item()
                    layer_gpos += g_pos.mean().item()
                    layer_gneg += g_neg.mean().item()
                    x_in = h_pos.detach()

                total_loss += layer_losses / len(model.layers)
                total_pos += layer_gpos / len(model.layers)
                total_neg += layer_gneg / len(model.layers)
            else:
                h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)

                if use_mode == "hallucinate":
                    x_neg = hallucinate_negative(
                        model,
                        x,
                        batch.edge_index,
                        getattr(batch, "edge_attr", None),
                        batch.batch,
                        hall_cfg,
                        edge_weight=edge_weight,
                    )
                    hall_used += 1
                else:
                    x_neg = make_negative(x, batch.batch, mode=use_mode, noise_std=noise_std)
                total_used += 1

                if use_mode == "hallucinate":
                    h_neg_probe = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                    g_neg_probe = goodness(
                        h_neg_probe, batch.batch, temperature=goodness_temp
                    ).mean().item()
                    g_pos_probe = g_pos.mean().item()
                    if g_neg_probe > g_pos_probe + neg_gate_margin:
                        x_neg = make_negative(x, batch.batch, mode="shuffle", noise_std=noise_std)
                        hall_used -= 1
                        hall_gated += 1
                h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp)

                loss = ff_loss(g_pos, g_neg, target=goodness_target)
                optim.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()

                total_loss += loss.item()
                total_pos += g_pos.mean().item()
                total_neg += g_neg.mean().item()
            batches += 1

        hall_ratio = hall_used / total_used if total_used else 0.0
        gate_ratio = hall_gated / total_used if total_used else 0.0
        print(
            f"Epoch {epoch:02d} | loss={total_loss / batches:.4f} "
            f"g_pos={total_pos / batches:.4f} g_neg={total_neg / batches:.4f} "
            f"hall_ratio={hall_ratio:.2f} gate_ratio={gate_ratio:.2f}"
        )

        if log_csv:
            with Path(log_csv).open("a") as f:
                f.write(
                    f"{epoch},{total_loss / batches:.6f},"
                    f"{total_pos / batches:.6f},{total_neg / batches:.6f},"
                    f"{hall_ratio:.4f},{gate_ratio:.4f}\n"
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
            if "hallucinate_ratio" in df.columns:
                plt.plot(df["epoch"], df["hallucinate_ratio"], label="hall_ratio")
            if "gate_ratio" in df.columns:
                plt.plot(df["epoch"], df["gate_ratio"], label="gate_ratio")
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
