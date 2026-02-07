#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import tomllib
import os
import csv
import math

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

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


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _compute_risk_targets(
    prices_path: str,
    ticker: str,
    dates: list[str],
    horizon: int,
    standardize: bool,
) -> tuple[list[float | None], float, float]:
    prices: list[tuple[str, float]] = []
    with Path(prices_path).open() as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("prices.csv missing header")
        price_col = "adj_close" if "adj_close" in r.fieldnames else "close"
        for row in r:
            if row.get("ticker") != ticker:
                continue
            date = row.get("date")
            if not date:
                continue
            val = row.get(price_col, "")
            if not val:
                continue
            try:
                price = float(val)
            except ValueError:
                continue
            prices.append((date, price))
    if not prices:
        raise ValueError(f"No prices found for ticker {ticker} in {prices_path}")

    prices.sort(key=lambda x: x[0])
    date_list = [d for d, _ in prices]
    price_list = [p for _, p in prices]
    returns = [
        math.log(price_list[i + 1] / price_list[i])
        for i in range(len(price_list) - 1)
        if price_list[i] > 0 and price_list[i + 1] > 0
    ]
    idx_map = {d: i for i, d in enumerate(date_list)}

    targets: list[float | None] = []
    for d in dates:
        idx = idx_map.get(d)
        if idx is None:
            targets.append(None)
            continue
        if idx + horizon > len(returns):
            targets.append(None)
            continue
        window = returns[idx : idx + horizon]
        if not window:
            targets.append(None)
            continue
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / len(window)
        vol = math.sqrt(var)
        targets.append(vol)

    finite = [t for t in targets if t is not None]
    if not finite:
        return targets, 0.0, 1.0
    mean = sum(finite) / len(finite)
    var = sum((x - mean) ** 2 for x in finite) / len(finite)
    std = math.sqrt(var) if var > 0 else 1.0

    if standardize:
        targets = [((t - mean) / (std + 1e-6)) if t is not None else None for t in targets]
    return targets, mean, std


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
    window_len: int | None,
    summary_dim: int,
    multiscale: bool,
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
    if multiscale:
        layers_pos = model(x, batch.edge_index, edge_weight=edge_weight, return_all=True)
        if neg_mode == "hallucinate":
            x_neg_hall = hallucinate_negative(
                model,
                x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                batch.batch,
                hall_cfg,
                edge_weight=edge_weight,
            )
        else:
            x_neg_hall = make_negative(
                x,
                batch.batch,
                mode=neg_mode,
                noise_std=noise_std,
                window_len=window_len,
                summary_dim=summary_dim,
            )
        x_neg_time = make_negative(
            x,
            batch.batch,
            mode="time_flip",
            noise_std=noise_std,
            window_len=window_len,
            summary_dim=summary_dim,
        )
        layers_neg_h = model(x_neg_hall, batch.edge_index, edge_weight=edge_weight, return_all=True)
        layers_neg_t = model(x_neg_time, batch.edge_index, edge_weight=edge_weight, return_all=True)
        loss = 0.0
        for h_pos, h_neg_h, h_neg_t in zip(layers_pos, layers_neg_h, layers_neg_t):
            g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)
            g_neg_h = goodness(h_neg_h, batch.batch, temperature=goodness_temp)
            g_neg_t = goodness(h_neg_t, batch.batch, temperature=goodness_temp)
            loss = loss + ff_loss(g_pos, g_neg_h, target=goodness_target)
            loss = loss + ff_loss(g_pos, g_neg_t, target=goodness_target)
        loss = loss / max(1, len(layers_pos))
    else:
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
            x_neg = make_negative(
                x,
                batch.batch,
                mode=neg_mode,
                noise_std=noise_std,
                window_len=window_len,
                summary_dim=summary_dim,
            )
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
        choices=[
            "shuffle",
            "noise",
            "shuffle+noise",
            "time_flip",
            "shuffle+time_flip",
            "time_flip+noise",
            "hallucinate",
            "schedule",
            "mix",
        ],
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
    parser.add_argument("--ff-multiscale", action="store_true", default=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    section = cfg.get("train", {})
    build_cfg = cfg.get("build_graphs", {})

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
    dataloader_persistent = _get_setting(args, section, "dataloader_persistent_workers", True)
    dataloader_prefetch = _get_setting(args, section, "dataloader_prefetch_factor", 2)
    dataloader_pin_memory = _get_setting(args, section, "dataloader_pin_memory", False)
    dataloader_mp_context = _get_setting(args, section, "dataloader_mp_context", "")
    torch_num_threads = _get_setting(args, section, "torch_num_threads", None)
    torch_num_interop_threads = _get_setting(args, section, "torch_num_interop_threads", None)
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
    ff_multiscale = _get_setting(args, section, "ff_multiscale", False) or getattr(
        args, "ff_multiscale", False
    )
    if ff_multiscale and ff_layerwise:
        print("ff_multiscale enabled; disabling ff_layerwise.")
        ff_layerwise = False

    hall_steps = _get_setting(args, section, "hallucinate_steps", 10)
    hall_lr = _get_setting(args, section, "hallucinate_lr", 0.1)
    hall_l2 = _get_setting(args, section, "hallucinate_l2", 0.1)
    hall_mean = _get_setting(args, section, "hallucinate_mean", 0.05)
    hall_std = _get_setting(args, section, "hallucinate_std", 0.05)
    hall_corr = _get_setting(args, section, "hallucinate_corr", 1.0)
    hall_clamp = _get_setting(args, section, "hallucinate_clamp_std", 3.0)
    hall_node_fraction = _get_setting(args, section, "hallucinate_node_fraction", 1.0)
    hall_node_min = _get_setting(args, section, "hallucinate_node_min", 1)
    hall_curriculum = section.get("hallucinate_curriculum", {})
    hall_curr_enabled = bool(hall_curriculum.get("enabled", False))
    hall_curr_start = int(hall_curriculum.get("start_epoch", 1))
    hall_curr_ramp = int(hall_curriculum.get("ramp_epochs", 1))
    layerwise_neg_mode = _get_setting(args, section, "layerwise_neg_mode", "shuffle")
    layerwise_noise_std = _get_setting(args, section, "layerwise_noise_std", noise_std)
    layerwise_hall_corr = _get_setting(args, section, "layerwise_hall_corr", 0.0)
    layerwise_hall_mean = _get_setting(args, section, "layerwise_hall_mean", hall_mean)
    layerwise_hall_std = _get_setting(args, section, "layerwise_hall_std", hall_std)
    feature_mode = build_cfg.get("feature_mode", "window")
    window_len = int(build_cfg.get("window", 20))
    returns_len = window_len if feature_mode in ("window", "window_plus_summary") else 1
    summary_dim = 5 if feature_mode == "window_plus_summary" else 0

    energy_penalty_weight = float(_get_setting(args, section, "energy_penalty_weight", 0.0))
    energy_penalty_mode = _get_setting(args, section, "energy_penalty_mode", "last")

    risk_head_enabled = bool(_get_setting(args, section, "risk_head_enabled", False))
    risk_ticker = _get_setting(args, section, "risk_ticker", "MDY")
    risk_horizon = int(_get_setting(args, section, "risk_horizon", 21))
    risk_loss_weight = float(_get_setting(args, section, "risk_loss_weight", 0.1))
    risk_loss_type = _get_setting(args, section, "risk_loss_type", "huber")
    risk_standardize = bool(_get_setting(args, section, "risk_standardize", True))

    def _hall_cfg_for_epoch(epoch: int, corr_override: float | None = None, mean_override: float | None = None, std_override: float | None = None) -> HallucinationConfig:
        if not hall_curr_enabled:
            return HallucinationConfig(
                steps=hall_steps,
                lr=hall_lr,
                l2_weight=hall_l2,
                mean_weight=hall_mean if mean_override is None else mean_override,
                std_weight=hall_std if std_override is None else std_override,
                corr_weight=hall_corr if corr_override is None else corr_override,
                clamp_std=hall_clamp,
                goodness_temp=goodness_temp,
                node_fraction=hall_node_fraction,
                node_min=hall_node_min,
            )

        if epoch < hall_curr_start:
            t = 0.0
        else:
            ramp = max(1, hall_curr_ramp)
            t = min(1.0, (epoch - hall_curr_start) / ramp)

        def _curr_val(name: str, base: float, cast=None):
            start = hall_curriculum.get(f"{name}_start", base)
            end = hall_curriculum.get(f"{name}_end", start)
            val = _lerp(float(start), float(end), t)
            if cast is not None:
                return cast(val)
            return val

        steps = max(1, _curr_val("steps", hall_steps, lambda v: int(round(v))))
        lr = _curr_val("lr", hall_lr)
        l2 = _curr_val("l2", hall_l2)
        mean_w = _curr_val("mean", hall_mean) if mean_override is None else mean_override
        std_w = _curr_val("std", hall_std) if std_override is None else std_override
        corr_w = _curr_val("corr", hall_corr) if corr_override is None else corr_override
        clamp_std = _curr_val("clamp_std", hall_clamp)
        node_fraction = _curr_val("node_fraction", hall_node_fraction)
        node_fraction = min(1.0, max(0.0, float(node_fraction)))
        node_min = max(1, _curr_val("node_min", hall_node_min, lambda v: int(round(v))))

        return HallucinationConfig(
            steps=steps,
            lr=lr,
            l2_weight=l2,
            mean_weight=mean_w,
            std_weight=std_w,
            corr_weight=corr_w,
            clamp_std=clamp_std,
            goodness_temp=goodness_temp,
            node_fraction=node_fraction,
            node_min=node_min,
        )

    set_seed(seed)
    if torch_num_threads:
        torch.set_num_threads(int(torch_num_threads))
    if torch_num_interop_threads:
        torch.set_num_interop_threads(int(torch_num_interop_threads))
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
    dates = payload.get("dates", [])
    if not graphs:
        raise ValueError("No graphs found in the provided file.")

    for i, g in enumerate(graphs):
        setattr(g, "graph_idx", i)

    print(f"device: {device}")
    print(f"mps built: {torch.backends.mps.is_built()}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    fallback = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "1") != "0"
    print(f"mps fallback enabled: {fallback}")
    print(
        f"neg_mode: {neg_mode} | batch_size: {batch_size} | loader_workers: {loader_workers}"
    )
    print(f"ff_layerwise: {ff_layerwise} | ff_multiscale: {ff_multiscale}")
    if energy_penalty_weight > 0:
        print(
            f"energy_penalty: {energy_penalty_weight} (mode={energy_penalty_mode})"
        )
    if risk_head_enabled:
        print(
            f"risk_head: ticker={risk_ticker} horizon={risk_horizon} "
            f"weight={risk_loss_weight} type={risk_loss_type} std={risk_standardize}"
        )
    if torch_num_threads or torch_num_interop_threads:
        print(
            f"torch threads: {torch.get_num_threads()} | interop: {torch.get_num_interop_threads()}"
        )
    if hall_curr_enabled:
        steps_start = hall_curriculum.get("steps_start", hall_steps)
        steps_end = hall_curriculum.get("steps_end", hall_steps)
        lr_start = hall_curriculum.get("lr_start", hall_lr)
        lr_end = hall_curriculum.get("lr_end", hall_lr)
        frac_start = hall_curriculum.get("node_fraction_start", hall_node_fraction)
        frac_end = hall_curriculum.get("node_fraction_end", hall_node_fraction)
        print(
            "hallucination curriculum: "
            f"start={hall_curr_start}, ramp={hall_curr_ramp}, "
            f"steps {steps_start}->{steps_end}, "
            f"lr {lr_start}->{lr_end}, "
            f"node_fraction {frac_start}->{frac_end}"
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
    model = GCNEncoder(
        in_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    risk_head = None
    risk_targets = None
    risk_target_mean = 0.0
    risk_target_std = 1.0
    if risk_head_enabled:
        if ff_layerwise:
            print("risk_head disabled when ff_layerwise is enabled.")
            risk_head_enabled = False
        elif not dates:
            print("risk_head disabled: graphs payload missing dates.")
            risk_head_enabled = False
        else:
            prices_path = build_cfg.get("prices", "data/processed/prices.csv")
            try:
                risk_targets, risk_target_mean, risk_target_std = _compute_risk_targets(
                    prices_path=prices_path,
                    ticker=str(risk_ticker),
                    dates=dates,
                    horizon=risk_horizon,
                    standardize=risk_standardize,
                )
            except Exception as exc:
                print(f"risk_head disabled: {exc}")
                risk_head_enabled = False

    if risk_head_enabled:
        risk_head = torch.nn.Linear(hidden_dim, 1).to(device)

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

    hall_cfg = _hall_cfg_for_epoch(hall_curr_start if hall_curr_enabled else 1)

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
                    returns_len,
                    summary_dim,
                    ff_multiscale,
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
                    returns_len,
                    summary_dim,
                    ff_multiscale,
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

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
        "num_workers": loader_workers,
        "pin_memory": bool(dataloader_pin_memory) if device.type == "cuda" else False,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = bool(dataloader_persistent)
        loader_kwargs["prefetch_factor"] = int(dataloader_prefetch)
        if dataloader_mp_context:
            loader_kwargs["multiprocessing_context"] = dataloader_mp_context
    loader = DataLoader(graphs, **loader_kwargs)
    if risk_head is not None:
        optim = Adam(list(model.parameters()) + list(risk_head.parameters()), lr=lr)
    else:
        optim = Adam(model.parameters(), lr=lr)

    if log_csv:
        log_path = Path(log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            f.write(
                "epoch,loss,g_pos,g_neg,hallucinate_ratio,gate_ratio,hall_hardness,"
                "energy_penalty,risk_loss\n"
            )

    epoch_iter = tqdm(
        range(1, epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for epoch in epoch_iter:
        model.train()
        hall_cfg = _hall_cfg_for_epoch(epoch)
        hall_cfg_layer = _hall_cfg_for_epoch(
            epoch,
            corr_override=layerwise_hall_corr,
            mean_override=layerwise_hall_mean,
            std_override=layerwise_hall_std,
        )
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        batches = 0
        energy_penalty_sum = 0.0
        risk_loss_sum = 0.0
        risk_batches = 0

        hall_used = 0
        total_used = 0
        hall_gated = 0
        hall_hardness_sum = 0.0
        hall_hardness_count = 0

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

            if ff_multiscale:
                layers_pos = model(x, batch.edge_index, edge_weight=edge_weight, return_all=True)

                hall_active = use_mode == "hallucinate"
                if use_mode == "hallucinate":
                    x_neg_hall = hallucinate_negative(
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
                    x_neg_hall = make_negative(
                        x,
                        batch.batch,
                        mode=use_mode,
                        noise_std=noise_std,
                        window_len=returns_len,
                        summary_dim=summary_dim,
                    )
                total_used += 1

                x_neg_time = make_negative(
                    x,
                    batch.batch,
                    mode="time_flip",
                    noise_std=noise_std,
                    window_len=returns_len,
                    summary_dim=summary_dim,
                )

                layers_neg_h = model(
                    x_neg_hall, batch.edge_index, edge_weight=edge_weight, return_all=True
                )
                layers_neg_t = model(
                    x_neg_time, batch.edge_index, edge_weight=edge_weight, return_all=True
                )

                if use_mode == "hallucinate":
                    g_pos_probe = goodness(
                        layers_pos[-1], batch.batch, temperature=goodness_temp
                    ).mean().item()
                    g_neg_probe = goodness(
                        layers_neg_h[-1], batch.batch, temperature=goodness_temp
                    ).mean().item()
                    if g_neg_probe > g_pos_probe + neg_gate_margin:
                        x_neg_hall = make_negative(
                            x,
                            batch.batch,
                            mode="shuffle",
                            noise_std=noise_std,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        hall_used -= 1
                        hall_gated += 1
                        hall_active = False
                        layers_neg_h = model(
                            x_neg_hall,
                            batch.edge_index,
                            edge_weight=edge_weight,
                            return_all=True,
                        )

                batch_loss = 0.0
                for h_p, h_n_h, h_n_t in zip(layers_pos, layers_neg_h, layers_neg_t):
                    g_p = goodness(h_p, batch.batch, temperature=goodness_temp)
                    g_n_h = goodness(h_n_h, batch.batch, temperature=goodness_temp)
                    g_n_t = goodness(h_n_t, batch.batch, temperature=goodness_temp)
                    batch_loss += ff_loss(g_p, g_n_h, target=goodness_target)
                    batch_loss += ff_loss(g_p, g_n_t, target=goodness_target)
                batch_loss = batch_loss / max(1, len(layers_pos))

                energy_penalty_val = 0.0
                if energy_penalty_weight > 0:
                    if energy_penalty_mode == "all":
                        energy_penalty_val = sum(
                            h.pow(2).mean() for h in layers_pos
                        ) / max(1, len(layers_pos))
                    else:
                        energy_penalty_val = layers_pos[-1].pow(2).mean()
                    batch_loss = batch_loss + energy_penalty_weight * energy_penalty_val

                risk_loss_val = 0.0
                if risk_head is not None and risk_targets is not None:
                    graph_idx = batch.graph_idx
                    if not torch.is_tensor(graph_idx):
                        graph_idx = torch.tensor(graph_idx)
                    idx_list = graph_idx.tolist()
                    target_vals = [
                        risk_targets[i] if risk_targets[i] is not None else float("nan")
                        for i in idx_list
                    ]
                    target = torch.tensor(target_vals, dtype=torch.float32, device=device)
                    mask = torch.isfinite(target)
                    if mask.any():
                        embed = global_mean_pool(layers_pos[-1], batch.batch)
                        pred = risk_head(embed).squeeze(-1)
                        if risk_loss_type == "mse":
                            risk_loss_val = F.mse_loss(pred[mask], target[mask])
                        else:
                            risk_loss_val = F.smooth_l1_loss(pred[mask], target[mask])
                        batch_loss = batch_loss + risk_loss_weight * risk_loss_val

                optim.zero_grad()
                batch_loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()

                g_pos_last = goodness(
                    layers_pos[-1], batch.batch, temperature=goodness_temp
                ).mean().item()
                g_neg_h_last = goodness(
                    layers_neg_h[-1], batch.batch, temperature=goodness_temp
                ).mean().item()
                g_neg_t_last = goodness(
                    layers_neg_t[-1], batch.batch, temperature=goodness_temp
                ).mean().item()

                total_loss += batch_loss.item()
                total_pos += g_pos_last
                total_neg += (g_neg_h_last + g_neg_t_last) / 2
                if hall_active:
                    hall_hardness_sum += (g_neg_h_last - g_pos_last)
                    hall_hardness_count += 1
                energy_penalty_sum += float(energy_penalty_val) if energy_penalty_weight > 0 else 0.0
                if risk_loss_val is not None and risk_loss_val != 0.0:
                    risk_loss_sum += float(risk_loss_val)
                    risk_batches += 1
            elif ff_layerwise:
                x_in = x
                layer_losses = 0.0
                layer_gpos = 0.0
                layer_gneg = 0.0
                for li in range(len(model.layers)):
                    layer_mode = use_mode
                    if use_mode == "hallucinate" and li > 0:
                        layer_mode = "hallucinate"
                    h_pos = model.forward_layer(x_in, batch.edge_index, edge_weight, li)
                    g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)

                    hall_active = layer_mode == "hallucinate"
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
                            hall_cfg_layer,
                            edge_weight=edge_weight,
                            forward_fn=forward_fn,
                        )
                        hall_used += 1
                    else:
                        x_neg = make_negative(
                            x_in,
                            batch.batch,
                            mode=layerwise_neg_mode,
                            noise_std=layerwise_noise_std,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                    total_used += 1

                    if layer_mode == "hallucinate":
                        h_neg_probe = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                        g_neg_probe = goodness(
                            h_neg_probe, batch.batch, temperature=goodness_temp
                        ).mean().item()
                        g_pos_probe = g_pos.mean().item()
                        if g_neg_probe > g_pos_probe + neg_gate_margin:
                            x_neg = make_negative(
                                x_in,
                                batch.batch,
                                mode="shuffle",
                                noise_std=noise_std,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            hall_used -= 1
                            hall_gated += 1
                            hall_active = False

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
                    if hall_active:
                        hall_hardness_sum += (g_neg.mean().item() - g_pos.mean().item())
                        hall_hardness_count += 1
                    x_in = h_pos.detach()

                total_loss += layer_losses / len(model.layers)
                total_pos += layer_gpos / len(model.layers)
                total_neg += layer_gneg / len(model.layers)
            else:
                h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)

                hall_active = use_mode == "hallucinate"
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
                    x_neg = make_negative(
                        x,
                        batch.batch,
                        mode=use_mode,
                        noise_std=noise_std,
                        window_len=returns_len,
                        summary_dim=summary_dim,
                    )
                total_used += 1

                if use_mode == "hallucinate":
                    h_neg_probe = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                    g_neg_probe = goodness(
                        h_neg_probe, batch.batch, temperature=goodness_temp
                    ).mean().item()
                    g_pos_probe = g_pos.mean().item()
                    if g_neg_probe > g_pos_probe + neg_gate_margin:
                        x_neg = make_negative(
                            x,
                            batch.batch,
                            mode="shuffle",
                            noise_std=noise_std,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        hall_used -= 1
                        hall_gated += 1
                        hall_active = False
                h_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight)
                g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp)

                loss = ff_loss(g_pos, g_neg, target=goodness_target)
                energy_penalty_val = 0.0
                if energy_penalty_weight > 0:
                    energy_penalty_val = h_pos.pow(2).mean()
                    loss = loss + energy_penalty_weight * energy_penalty_val

                risk_loss_val = 0.0
                if risk_head is not None and risk_targets is not None:
                    graph_idx = batch.graph_idx
                    if not torch.is_tensor(graph_idx):
                        graph_idx = torch.tensor(graph_idx)
                    idx_list = graph_idx.tolist()
                    target_vals = [
                        risk_targets[i] if risk_targets[i] is not None else float("nan")
                        for i in idx_list
                    ]
                    target = torch.tensor(target_vals, dtype=torch.float32, device=device)
                    mask = torch.isfinite(target)
                    if mask.any():
                        embed = global_mean_pool(h_pos, batch.batch)
                        pred = risk_head(embed).squeeze(-1)
                        if risk_loss_type == "mse":
                            risk_loss_val = F.mse_loss(pred[mask], target[mask])
                        else:
                            risk_loss_val = F.smooth_l1_loss(pred[mask], target[mask])
                        loss = loss + risk_loss_weight * risk_loss_val
                optim.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()

                total_loss += loss.item()
                total_pos += g_pos.mean().item()
                total_neg += g_neg.mean().item()
                if hall_active:
                    hall_hardness_sum += (g_neg.mean().item() - g_pos.mean().item())
                    hall_hardness_count += 1
                energy_penalty_sum += float(energy_penalty_val) if energy_penalty_weight > 0 else 0.0
                if risk_loss_val is not None and risk_loss_val != 0.0:
                    risk_loss_sum += float(risk_loss_val)
                    risk_batches += 1
            batches += 1

        hall_ratio = hall_used / total_used if total_used else 0.0
        gate_ratio = hall_gated / total_used if total_used else 0.0
        hall_hardness = hall_hardness_sum / hall_hardness_count if hall_hardness_count else 0.0
        energy_penalty_epoch = energy_penalty_sum / batches if batches else 0.0
        risk_loss_epoch = risk_loss_sum / risk_batches if risk_batches else 0.0
        # Progress bar only; metrics are saved to CSV/plots.

        if log_csv:
            with Path(log_csv).open("a") as f:
                f.write(
                    f"{epoch},{total_loss / batches:.6f},"
                    f"{total_pos / batches:.6f},{total_neg / batches:.6f},"
                    f"{hall_ratio:.4f},{gate_ratio:.4f},{hall_hardness:.6f},"
                    f"{energy_penalty_epoch:.6f},{risk_loss_epoch:.6f}\n"
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
            if "hall_hardness" in df.columns:
                plt.plot(df["epoch"], df["hall_hardness"], label="hall_hardness")
            if "energy_penalty" in df.columns:
                plt.plot(df["epoch"], df["energy_penalty"], label="energy_penalty")
            if "risk_loss" in df.columns:
                plt.plot(df["epoch"], df["risk_loss"], label="risk_loss")
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
