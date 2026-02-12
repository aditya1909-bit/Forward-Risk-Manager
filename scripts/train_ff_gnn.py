#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import random
import sys
import tomllib
import csv
import math
import hashlib

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from frisk.models import GCNEncoder
from frisk.ff import (
    ff_loss,
    goodness,
    make_negative,
    pairwise_distance_forward_loss,
    permute_graph_embeddings,
    self_contrastive_loss,
)
from frisk.hallucinate import HallucinationConfig, hallucinate_negative
from frisk.device import collect_device_diagnostics, empty_device_cache, resolve_device

_RISK_TARGET_MEM_CACHE: dict[str, tuple[list[float | None], float, float]] = {}
_NEG_AUG_MODES = {
    "shuffle",
    "noise",
    "shuffle+noise",
    "time_flip",
    "shuffle+time_flip",
    "time_flip+noise",
}


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


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _block_endpoint_indices(num_layers: int, block_size: int) -> list[int]:
    if num_layers <= 0:
        return []
    step = max(1, int(block_size))
    endpoints = list(range(step - 1, num_layers, step))
    if endpoints[-1] != num_layers - 1:
        endpoints.append(num_layers - 1)
    return endpoints


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_amp_dtype(value) -> torch.dtype:
    name = str(value).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def _build_optimizer(params, lr: float, device: torch.device, use_fused: bool):
    params = tuple(params)
    if not params:
        raise ValueError("optimizer got an empty parameter list")
    kwargs = {}
    if device.type == "cuda":
        kwargs["foreach"] = True
        if use_fused:
            kwargs["fused"] = True
    try:
        return Adam(params, lr=lr, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
    try:
        return Adam(params, lr=lr, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("foreach", None)
        return Adam(params, lr=lr, **kwargs)


def _make_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_if_needed(enabled: bool, dtype: torch.dtype):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _optimizer_step(
    optim: torch.optim.Optimizer,
    loss: torch.Tensor,
    grad_clip: float,
    clip_params,
    scaler,
) -> None:
    optim.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
        scaler.step(optim)
        scaler.update()
        return
    loss.backward()
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
    optim.step()


def _parse_positive_int_list(value, fallback: int) -> list[int]:
    vals: list[int] = []
    raw_items = []
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    elif value is not None:
        raw_items = str(value).split(",")

    seen = set()
    for item in raw_items:
        s = str(item).strip()
        if not s:
            continue
        try:
            v = int(float(s))
        except ValueError:
            continue
        if v <= 0 or v in seen:
            continue
        seen.add(v)
        vals.append(v)

    if vals:
        return vals
    return [max(1, int(fallback))]


def _compute_risk_targets(
    prices_path: str,
    ticker: str,
    dates: list[str],
    horizon: int,
    standardize: bool,
    max_abs_logret: float,
    cache_dir: str | None = "runs/cache",
) -> tuple[list[float | None], float, float]:
    prices_file = Path(prices_path)
    try:
        st = prices_file.stat()
        file_sig = f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        file_sig = "missing"
    dates_hash = hashlib.sha1("\n".join(dates).encode("utf-8")).hexdigest()
    cache_key = hashlib.sha1(
        "|".join(
            [
                str(prices_file.resolve()),
                str(ticker).upper(),
                str(horizon),
                str(int(bool(standardize))),
                f"{float(max_abs_logret):.8f}",
                file_sig,
                dates_hash,
            ]
        ).encode("utf-8")
    ).hexdigest()

    cached = _RISK_TARGET_MEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_path: Path | None = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"risk_targets_{cache_key}.pt"
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                targets = payload["targets"]
                mean = float(payload["mean"])
                std = float(payload["std"])
                result = (targets, mean, std)
                _RISK_TARGET_MEM_CACHE[cache_key] = result
                return result
            except Exception:
                pass

    prices_by_date: dict[str, list[float]] = {}
    ticker_norm = str(ticker).upper()
    with Path(prices_path).open() as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("prices.csv missing header")
        price_col = "adj_close" if "adj_close" in r.fieldnames else "close"
        for row in r:
            if str(row.get("ticker", "")).upper() != ticker_norm:
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
            if not math.isfinite(price) or price <= 0:
                continue
            prices_by_date.setdefault(date, []).append(price)
    prices: list[tuple[str, float]] = []
    for date, vals in prices_by_date.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        mid = len(vals_sorted) // 2
        if len(vals_sorted) % 2 == 1:
            px = vals_sorted[mid]
        else:
            px = 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
        prices.append((date, float(px)))
    if not prices:
        raise ValueError(f"No prices found for ticker {ticker} in {prices_path}")

    prices.sort(key=lambda x: x[0])
    date_list = [d for d, _ in prices]
    price_list = [p for _, p in prices]
    returns = []
    clip = float(max_abs_logret)
    for i in range(len(price_list) - 1):
        if price_list[i] <= 0 or price_list[i + 1] <= 0:
            returns.append(0.0)
            continue
        ret = math.log(price_list[i + 1] / price_list[i])
        if clip > 0 and abs(ret) > clip:
            ret = math.copysign(clip, ret)
        returns.append(ret)
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
    result = (targets, mean, std)
    _RISK_TARGET_MEM_CACHE[cache_key] = result
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"targets": targets, "mean": mean, "std": std}, cache_path)
        except Exception:
            pass
    return result


def _compute_multi_horizon_risk_loss(
    risk_head: torch.nn.Module,
    embeddings: torch.Tensor,
    graph_idx,
    risk_targets_by_horizon: list[list[float | None]],
    device: torch.device,
    risk_loss_type: str,
) -> torch.Tensor | None:
    if not risk_targets_by_horizon:
        return None

    if torch.is_tensor(graph_idx):
        idx_list = graph_idx.detach().cpu().tolist()
    elif isinstance(graph_idx, (list, tuple)):
        idx_list = list(graph_idx)
    else:
        idx_list = [int(graph_idx)]

    target_rows = []
    for gi in idx_list:
        row = []
        for horizon_targets in risk_targets_by_horizon:
            if 0 <= gi < len(horizon_targets):
                t = horizon_targets[gi]
            else:
                t = None
            row.append(float(t) if t is not None else float("nan"))
        target_rows.append(row)

    target = torch.tensor(target_rows, dtype=torch.float32, device=device)
    mask = torch.isfinite(target)
    if not mask.any():
        return None

    pred = risk_head(embeddings)
    if pred.ndim == 1:
        pred = pred.unsqueeze(-1)
    if pred.shape != target.shape:
        raise RuntimeError(
            f"risk head output shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if risk_loss_type == "mse":
        return F.mse_loss(pred[mask], target[mask])
    return F.smooth_l1_loss(pred[mask], target[mask])


def _self_contrastive_batch_loss(
    h_pos: torch.Tensor,
    h_view: torch.Tensor,
    batch: torch.Tensor,
    temperature: float,
    max_graphs: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z_pos = global_mean_pool(h_pos, batch)
    z_view = global_mean_pool(h_view, batch)
    if max_graphs and z_pos.size(0) > max_graphs:
        idx = torch.randperm(z_pos.size(0), device=z_pos.device)[:max_graphs]
        z_pos = z_pos.index_select(0, idx)
        z_view = z_view.index_select(0, idx)
    loss, pos_score, neg_score = self_contrastive_loss(
        z_pos,
        z_view,
        temperature=temperature,
    )
    return loss, pos_score, neg_score, z_pos, z_view


def _make_self_contrastive_view(
    x: torch.Tensor,
    batch: torch.Tensor,
    view_mode: str,
    view_noise_std: float,
    window_len: int | None = None,
    summary_dim: int = 0,
) -> torch.Tensor:
    mode = str(view_mode).strip().lower()
    if mode in ("", "auto"):
        mode = "shuffle+noise"
    valid_modes = {
        "shuffle",
        "noise",
        "shuffle+noise",
        "time_flip",
        "shuffle+time_flip",
        "time_flip+noise",
    }
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported self_contrastive view_mode={view_mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )
    return make_negative(
        x,
        batch,
        mode=mode,
        noise_std=max(0.0, float(view_noise_std)),
        window_len=window_len,
        summary_dim=summary_dim,
    )


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
    self_contrastive_temp: float,
    self_contrastive_max_graphs: int,
    self_contrastive_view_mode: str,
    self_contrastive_view_noise_std: float,
    self_contrastive_ff_weight: float,
    self_contrastive_ff_neg_mode: str,
    self_contrastive_ff_noise_std: float,
    self_contrastive_ff_target: float,
    distance_forward_weight: float,
    distance_forward_margin: float,
    distance_forward_max_graphs: int,
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
        if neg_mode == "self_contrastive":
            x_view = _make_self_contrastive_view(
                x,
                batch.batch,
                view_mode=self_contrastive_view_mode,
                view_noise_std=self_contrastive_view_noise_std,
                window_len=window_len,
                summary_dim=summary_dim,
            )
            layers_view = model(
                x_view, batch.edge_index, edge_weight=edge_weight, return_all=True
            )
            loss = 0.0
            z_pos_last = None
            z_neg_last = None
            for h_pos, h_view in zip(layers_pos, layers_view):
                sc_loss, _, _, z_pos, z_view = _self_contrastive_batch_loss(
                    h_pos,
                    h_view,
                    batch.batch,
                    temperature=self_contrastive_temp,
                    max_graphs=self_contrastive_max_graphs,
                )
                loss = loss + sc_loss
                z_pos_last = z_pos
                z_neg_last = permute_graph_embeddings(z_view)
            loss = loss / max(1, len(layers_pos))
            if (
                distance_forward_weight > 0
                and z_pos_last is not None
                and z_neg_last is not None
            ):
                loss = loss + distance_forward_weight * pairwise_distance_forward_loss(
                    z_pos_last,
                    z_neg_last,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
            if self_contrastive_ff_weight > 0:
                x_neg_aux = make_negative(
                    x,
                    batch.batch,
                    mode=self_contrastive_ff_neg_mode,
                    noise_std=self_contrastive_ff_noise_std,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )
                layers_neg_aux = model(
                    x_neg_aux,
                    batch.edge_index,
                    edge_weight=edge_weight,
                    return_all=True,
                )
                g_pos_aux = goodness(layers_pos[-1], batch.batch, temperature=goodness_temp)
                g_neg_aux = goodness(layers_neg_aux[-1], batch.batch, temperature=goodness_temp)
                loss = loss + self_contrastive_ff_weight * ff_loss(
                    g_pos_aux,
                    g_neg_aux,
                    target=self_contrastive_ff_target,
                )
        else:
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
            layers_neg_h = model(
                x_neg_hall, batch.edge_index, edge_weight=edge_weight, return_all=True
            )
            layers_neg_t = model(
                x_neg_time, batch.edge_index, edge_weight=edge_weight, return_all=True
            )
            loss = 0.0
            for h_pos, h_neg_h, h_neg_t in zip(layers_pos, layers_neg_h, layers_neg_t):
                g_pos = goodness(h_pos, batch.batch, temperature=goodness_temp)
                g_neg_h = goodness(h_neg_h, batch.batch, temperature=goodness_temp)
                g_neg_t = goodness(h_neg_t, batch.batch, temperature=goodness_temp)
                loss = loss + ff_loss(g_pos, g_neg_h, target=goodness_target)
                loss = loss + ff_loss(g_pos, g_neg_t, target=goodness_target)
            loss = loss / max(1, len(layers_pos))
            if distance_forward_weight > 0:
                z_pos = global_mean_pool(layers_pos[-1], batch.batch)
                z_neg_h = global_mean_pool(layers_neg_h[-1], batch.batch)
                z_neg_t = global_mean_pool(layers_neg_t[-1], batch.batch)
                dist_loss_h = pairwise_distance_forward_loss(
                    z_pos,
                    z_neg_h,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
                dist_loss_t = pairwise_distance_forward_loss(
                    z_pos,
                    z_neg_t,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
                loss = loss + distance_forward_weight * 0.5 * (dist_loss_h + dist_loss_t)
    else:
        h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
        if neg_mode == "self_contrastive":
            x_view = _make_self_contrastive_view(
                x,
                batch.batch,
                view_mode=self_contrastive_view_mode,
                view_noise_std=self_contrastive_view_noise_std,
                window_len=window_len,
                summary_dim=summary_dim,
            )
            h_view = model(x_view, batch.edge_index, edge_weight=edge_weight)
            loss, _, _, z_pos, z_view = _self_contrastive_batch_loss(
                h_pos,
                h_view,
                batch.batch,
                temperature=self_contrastive_temp,
                max_graphs=self_contrastive_max_graphs,
            )
            if distance_forward_weight > 0:
                z_neg = permute_graph_embeddings(z_view)
                loss = loss + distance_forward_weight * pairwise_distance_forward_loss(
                    z_pos,
                    z_neg,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
            if self_contrastive_ff_weight > 0:
                x_neg_aux = make_negative(
                    x,
                    batch.batch,
                    mode=self_contrastive_ff_neg_mode,
                    noise_std=self_contrastive_ff_noise_std,
                    window_len=window_len,
                    summary_dim=summary_dim,
                )
                h_neg_aux = model(x_neg_aux, batch.edge_index, edge_weight=edge_weight)
                g_pos_aux = goodness(h_pos, batch.batch, temperature=goodness_temp)
                g_neg_aux = goodness(h_neg_aux, batch.batch, temperature=goodness_temp)
                loss = loss + self_contrastive_ff_weight * ff_loss(
                    g_pos_aux,
                    g_neg_aux,
                    target=self_contrastive_ff_target,
                )
        else:
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
            if distance_forward_weight > 0:
                z_pos = global_mean_pool(h_pos, batch.batch)
                z_neg = global_mean_pool(h_neg, batch.batch)
                loss = loss + distance_forward_weight * pairwise_distance_forward_loss(
                    z_pos,
                    z_neg,
                    margin=distance_forward_margin,
                    max_graphs=distance_forward_max_graphs,
                )
    loss.backward()
    model.zero_grad(set_to_none=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
            "self_contrastive",
        ],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--noise-std", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=argparse.SUPPRESS)
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
    parser.add_argument("--self-contrastive-view-mode", default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-view-noise-std", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-ff-weight", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-ff-neg-mode", default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-ff-noise-std", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--self-contrastive-ff-target", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--self-contrastive-energy-penalty-scale",
        type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--temp-sweep", default=argparse.SUPPRESS)
    parser.add_argument("--ff-layerwise", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--ff-blockwise", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--ff-block-size", type=int, default=argparse.SUPPRESS)
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
    device_choice = _get_setting(args, section, "device", "auto")
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
    self_contrastive_temp = float(_get_setting(args, section, "self_contrastive_temp", 0.2))
    self_contrastive_view_mode = str(
        _get_setting(args, section, "self_contrastive_view_mode", "shuffle+noise")
    )
    self_contrastive_view_noise_std = float(
        _get_setting(args, section, "self_contrastive_view_noise_std", noise_std)
    )
    self_contrastive_max_graphs = int(_get_setting(args, section, "self_contrastive_max_graphs", 0))
    self_contrastive_ff_weight = float(
        _get_setting(args, section, "self_contrastive_ff_weight", 0.0)
    )
    self_contrastive_ff_neg_mode = str(
        _get_setting(args, section, "self_contrastive_ff_neg_mode", "shuffle+noise")
    )
    self_contrastive_ff_noise_std = float(
        _get_setting(args, section, "self_contrastive_ff_noise_std", noise_std)
    )
    self_contrastive_ff_target = float(
        _get_setting(args, section, "self_contrastive_ff_target", goodness_target)
    )
    self_contrastive_energy_penalty_scale = float(
        _get_setting(args, section, "self_contrastive_energy_penalty_scale", 0.0)
    )
    distance_forward_weight = float(_get_setting(args, section, "distance_forward_weight", 0.0))
    distance_forward_margin = float(_get_setting(args, section, "distance_forward_margin", 0.15))
    distance_forward_max_graphs = int(_get_setting(args, section, "distance_forward_max_graphs", 0))
    distance_forward_interval = int(_get_setting(args, section, "distance_forward_interval", 1))
    amp_requested = _to_bool(_get_setting(args, section, "amp", True))
    amp_dtype_raw = _get_setting(args, section, "amp_dtype", "float16")
    fused_optimizer = _to_bool(_get_setting(args, section, "fused_optimizer", True))
    ff_layerwise = _get_setting(args, section, "ff_layerwise", False) or getattr(
        args, "ff_layerwise", False
    )
    ff_blockwise = _to_bool(_get_setting(args, section, "ff_blockwise", False)) or getattr(
        args, "ff_blockwise", False
    )
    ff_block_size = int(_get_setting(args, section, "ff_block_size", 2))
    ff_multiscale = _get_setting(args, section, "ff_multiscale", False) or getattr(
        args, "ff_multiscale", False
    )
    if neg_mode == "self_contrastive" and ff_layerwise:
        print("self_contrastive mode requires end-to-end FF; disabling ff_layerwise.")
        ff_layerwise = False
    if ff_blockwise and not ff_layerwise:
        print("ff_blockwise requires ff_layerwise; disabling ff_blockwise.")
        ff_blockwise = False
    if ff_block_size < 1:
        ff_block_size = 1
    if ff_blockwise and ff_block_size <= 1:
        print("ff_block_size <= 1; disabling ff_blockwise.")
        ff_blockwise = False
    if ff_multiscale and ff_layerwise:
        print("ff_multiscale enabled; disabling ff_layerwise.")
        ff_layerwise = False
    if ff_multiscale and ff_blockwise:
        ff_blockwise = False

    hall_steps = _get_setting(args, section, "hallucinate_steps", 10)
    hall_lr = _get_setting(args, section, "hallucinate_lr", 0.1)
    hall_l2 = _get_setting(args, section, "hallucinate_l2", 0.1)
    hall_mean = _get_setting(args, section, "hallucinate_mean", 0.05)
    hall_std = _get_setting(args, section, "hallucinate_std", 0.05)
    hall_corr = _get_setting(args, section, "hallucinate_corr", 1.0)
    hall_clamp = _get_setting(args, section, "hallucinate_clamp_std", 3.0)
    hall_node_fraction = _get_setting(args, section, "hallucinate_node_fraction", 1.0)
    hall_node_min = _get_setting(args, section, "hallucinate_node_min", 1)
    hall_init_noise = _get_setting(args, section, "hallucinate_init_noise", 0.0)
    hall_min_delta = _get_setting(args, section, "hallucinate_min_delta", 0.0)
    hall_fallback_noise = _get_setting(args, section, "hallucinate_fallback_noise", 1.0)
    hall_penalty_scope = str(
        _get_setting(args, section, "hallucinate_penalty_scope", "returns")
    ).strip().lower()
    hall_corr_scope = str(
        _get_setting(args, section, "hallucinate_corr_scope", "returns")
    ).strip().lower()
    hall_freeze_non_return = _to_bool(
        _get_setting(args, section, "hallucinate_freeze_non_return_features", True)
    )
    if hall_penalty_scope not in {"all", "returns"}:
        hall_penalty_scope = "returns"
    if hall_corr_scope not in {"all", "returns"}:
        hall_corr_scope = "returns"
    hall_curriculum = section.get("hallucinate_curriculum", {})
    hall_curr_enabled = bool(hall_curriculum.get("enabled", False))
    hall_curr_start = int(hall_curriculum.get("start_epoch", 1))
    hall_curr_ramp = int(hall_curriculum.get("ramp_epochs", 1))

    def _sync_hall_curriculum_end() -> None:
        if not hall_curr_enabled:
            return
        hall_curriculum["steps_end"] = int(hall_steps)
        hall_curriculum["lr_end"] = float(hall_lr)
        hall_curriculum["l2_end"] = float(hall_l2)
        hall_curriculum["mean_end"] = float(hall_mean)
        hall_curriculum["std_end"] = float(hall_std)
        hall_curriculum["corr_end"] = float(hall_corr)
        hall_curriculum["node_fraction_end"] = float(hall_node_fraction)
        hall_curriculum["node_min_end"] = int(hall_node_min)

    layerwise_neg_mode = _get_setting(args, section, "layerwise_neg_mode", "shuffle")
    layerwise_noise_std = _get_setting(args, section, "layerwise_noise_std", noise_std)
    layerwise_hall_corr = _get_setting(args, section, "layerwise_hall_corr", 0.0)
    layerwise_hall_mean = _get_setting(args, section, "layerwise_hall_mean", hall_mean)
    layerwise_hall_std = _get_setting(args, section, "layerwise_hall_std", hall_std)
    feature_mode = build_cfg.get("feature_mode", "window")
    window_len = int(build_cfg.get("window", 20))
    if feature_mode in ("window", "window_plus_summary", "window_plus_summary_fund"):
        returns_len = window_len
    else:
        returns_len = 1
    if feature_mode == "window_plus_summary":
        summary_dim = 5
    elif feature_mode == "window_plus_summary_fund":
        summary_dim = 10
    else:
        summary_dim = 0

    energy_penalty_weight = float(_get_setting(args, section, "energy_penalty_weight", 0.0))
    energy_penalty_mode = _get_setting(args, section, "energy_penalty_mode", "last")

    risk_head_enabled = bool(_get_setting(args, section, "risk_head_enabled", False))
    risk_ticker = _get_setting(args, section, "risk_ticker", "MDY")
    risk_horizon = int(_get_setting(args, section, "risk_horizon", 21))
    risk_horizons = _parse_positive_int_list(
        _get_setting(args, section, "risk_horizons", section.get("risk_horizon", risk_horizon)),
        fallback=risk_horizon,
    )
    risk_loss_weight = float(_get_setting(args, section, "risk_loss_weight", 0.1))
    risk_loss_type = _get_setting(args, section, "risk_loss_type", "huber")
    risk_standardize = bool(_get_setting(args, section, "risk_standardize", True))
    risk_cache_dir = _get_setting(args, section, "risk_cache_dir", "runs/cache")
    risk_max_abs_logret = float(_get_setting(args, section, "risk_max_abs_logret", 0.5))

    adaptive_hall_enabled = _to_bool(_get_setting(args, section, "adaptive_hallucination", True))
    adaptive_hall_close_high = float(_get_setting(args, section, "adaptive_hall_close_high", 0.75))
    adaptive_hall_hardness_low = float(_get_setting(args, section, "adaptive_hall_hardness_low", -0.2))
    adaptive_hall_hardness_high = float(_get_setting(args, section, "adaptive_hall_hardness_high", 0.4))
    adaptive_hall_lr_mult = float(_get_setting(args, section, "adaptive_hall_lr_mult", 1.15))
    adaptive_hall_lr_max = float(_get_setting(args, section, "adaptive_hall_lr_max", 0.2))
    adaptive_hall_steps_inc = int(_get_setting(args, section, "adaptive_hall_steps_inc", 1))
    adaptive_hall_steps_max = int(_get_setting(args, section, "adaptive_hall_steps_max", 16))
    adaptive_hall_node_inc = float(_get_setting(args, section, "adaptive_hall_node_inc", 0.05))
    adaptive_hall_reg_mult = float(_get_setting(args, section, "adaptive_hall_reg_mult", 0.9))
    adaptive_hall_reg_min = float(_get_setting(args, section, "adaptive_hall_reg_min", 0.001))
    adaptive_hall_min_delta_mult = float(_get_setting(args, section, "adaptive_hall_min_delta_mult", 0.9))
    adaptive_hall_min_delta_min = float(_get_setting(args, section, "adaptive_hall_min_delta_min", 0.005))
    adaptive_hall_ratio_high = float(_get_setting(args, section, "adaptive_hall_ratio_high", 0.8))
    adaptive_mix_step = float(_get_setting(args, section, "adaptive_mix_step", 0.05))
    adaptive_mix_end_max = float(_get_setting(args, section, "adaptive_mix_end_max", 0.85))
    adaptive_gate_margin_step = float(_get_setting(args, section, "adaptive_gate_margin_step", 0.05))
    adaptive_gate_margin_min = float(_get_setting(args, section, "adaptive_gate_margin_min", 0.1))
    adaptive_gate_margin_max = float(_get_setting(args, section, "adaptive_gate_margin_max", 2.5))

    adaptive_target_enabled = _to_bool(_get_setting(args, section, "adaptive_goodness_target", True))
    adaptive_target_warmup = int(_get_setting(args, section, "adaptive_goodness_target_warmup", 5))
    adaptive_target_alpha = float(_get_setting(args, section, "adaptive_goodness_target_alpha", 0.15))
    adaptive_target_margin = float(_get_setting(args, section, "adaptive_goodness_target_margin", 0.0))
    adaptive_target_min = float(_get_setting(args, section, "adaptive_goodness_target_min", 0.1))
    adaptive_target_max = float(_get_setting(args, section, "adaptive_goodness_target_max", 10.0))
    self_contrastive_temp = _clamp(self_contrastive_temp, 1e-4, 10.0)
    self_contrastive_view_mode = str(self_contrastive_view_mode).strip().lower()
    if self_contrastive_view_mode in ("", "auto"):
        self_contrastive_view_mode = "shuffle+noise"
    self_contrastive_view_noise_std = max(0.0, float(self_contrastive_view_noise_std))
    self_contrastive_max_graphs = max(0, int(self_contrastive_max_graphs))
    self_contrastive_ff_weight = max(0.0, float(self_contrastive_ff_weight))
    self_contrastive_ff_neg_mode = str(self_contrastive_ff_neg_mode).strip().lower()
    if self_contrastive_ff_neg_mode not in _NEG_AUG_MODES:
        self_contrastive_ff_neg_mode = "shuffle+noise"
    self_contrastive_ff_noise_std = max(0.0, float(self_contrastive_ff_noise_std))
    self_contrastive_ff_target = float(self_contrastive_ff_target)
    self_contrastive_energy_penalty_scale = max(0.0, float(self_contrastive_energy_penalty_scale))
    distance_forward_weight = max(0.0, float(distance_forward_weight))
    distance_forward_margin = max(0.0, float(distance_forward_margin))
    distance_forward_max_graphs = max(0, int(distance_forward_max_graphs))
    distance_forward_interval = max(1, int(distance_forward_interval))
    if neg_mode == "self_contrastive" and adaptive_target_enabled:
        print("adaptive_goodness_target disabled for self_contrastive mode.")
        adaptive_target_enabled = False
    adaptive_mix_end_max = _clamp(adaptive_mix_end_max, 0.0, 0.99)
    if neg_mode == "mix":
        neg_mix_end = _clamp(float(neg_mix_end), float(neg_mix_start), adaptive_mix_end_max)

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
                init_noise=hall_init_noise,
                return_slice_len=returns_len,
                penalty_scope=hall_penalty_scope,
                corr_scope=hall_corr_scope,
                freeze_non_return_features=hall_freeze_non_return,
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
            init_noise=hall_init_noise,
            return_slice_len=returns_len,
            penalty_scope=hall_penalty_scope,
            corr_scope=hall_corr_scope,
            freeze_non_return_features=hall_freeze_non_return,
        )

    set_seed(seed)
    if torch_num_threads:
        torch.set_num_threads(int(torch_num_threads))
    if torch_num_interop_threads:
        torch.set_num_interop_threads(int(torch_num_interop_threads))
    device = resolve_device(device_choice)
    amp_dtype = _parse_amp_dtype(amp_dtype_raw)
    amp_enabled = bool(amp_requested and device.type == "cuda")
    if amp_enabled and amp_dtype == torch.bfloat16:
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if not bf16_ok:
            amp_dtype = torch.float16
            print("amp_dtype=bfloat16 not supported on this CUDA device; falling back to float16.")
    scaler = _make_scaler(amp_enabled and amp_dtype == torch.float16)

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

    print(f"device request: {device_choice}")
    print(f"device: {device}")
    for key, value in collect_device_diagnostics().items():
        print(f"{key}: {value}")
    print(
        f"neg_mode: {neg_mode} | batch_size: {batch_size} | loader_workers: {loader_workers}"
    )
    print(
        "ff_mode: "
        f"layerwise={ff_layerwise}, blockwise={ff_blockwise}, "
        f"block_size={ff_block_size}, multiscale={ff_multiscale}"
    )
    if energy_penalty_weight > 0:
        print(
            f"energy_penalty: {energy_penalty_weight} (mode={energy_penalty_mode})"
        )
    if risk_head_enabled:
        print(
            f"risk_head: ticker={risk_ticker} horizons={risk_horizons} "
            f"weight={risk_loss_weight} type={risk_loss_type} std={risk_standardize} "
            f"max_abs_logret={risk_max_abs_logret}"
        )
    if neg_mode == "self_contrastive":
        print(f"self_contrastive_temp: {self_contrastive_temp}")
        print(
            "self_contrastive_view: "
            f"mode={self_contrastive_view_mode}, noise_std={self_contrastive_view_noise_std}"
        )
        if self_contrastive_max_graphs > 0:
            print(f"self_contrastive_max_graphs: {self_contrastive_max_graphs}")
        if self_contrastive_ff_weight > 0:
            print(
                "self_contrastive_ff_aux: "
                f"weight={self_contrastive_ff_weight}, "
                f"neg_mode={self_contrastive_ff_neg_mode}, "
                f"noise_std={self_contrastive_ff_noise_std}, "
                f"target={self_contrastive_ff_target}"
            )
        if energy_penalty_weight > 0 and self_contrastive_energy_penalty_scale != 1.0:
            print(
                "self_contrastive_energy_penalty_scale: "
                f"{self_contrastive_energy_penalty_scale}"
            )
    if distance_forward_weight > 0:
        print(
            "distance_forward: "
            f"weight={distance_forward_weight}, margin={distance_forward_margin}"
        )
        if distance_forward_interval > 1:
            print(f"distance_forward_interval: every {distance_forward_interval} batches")
        if distance_forward_max_graphs > 0:
            print(f"distance_forward_max_graphs: {distance_forward_max_graphs}")
    if amp_enabled:
        dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
        print(f"amp: enabled ({dtype_name})")
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
        _sync_hall_curriculum_end()
    print(
        "hallucination scope: "
        f"penalty={hall_penalty_scope}, corr={hall_corr_scope}, "
        f"freeze_non_return={hall_freeze_non_return}, return_slice_len={returns_len}"
    )
    if neg_mode in ("schedule", "mix"):
        print(f"neg_warmup_epochs: {neg_warmup_epochs}")
    if neg_mode == "mix":
        print(
            f"neg_mix_start: {neg_mix_start} | neg_mix_end: {neg_mix_end} | "
            f"neg_mix_ramp_epochs: {neg_mix_ramp_epochs}"
        )
    if adaptive_hall_enabled:
        print(
            "adaptive_hallucination: "
            f"close_high={adaptive_hall_close_high}, "
            f"hardness_low={adaptive_hall_hardness_low}, "
            f"hardness_high={adaptive_hall_hardness_high}, "
            f"ratio_high={adaptive_hall_ratio_high}, "
            f"mix_end_max={adaptive_mix_end_max}"
        )
    if adaptive_target_enabled:
        print(
            "adaptive_goodness_target: "
            f"warmup={adaptive_target_warmup}, alpha={adaptive_target_alpha}, "
            f"range=[{adaptive_target_min}, {adaptive_target_max}]"
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
    ff_block_endpoints = (
        _block_endpoint_indices(len(model.layers), ff_block_size) if ff_blockwise else []
    )
    if ff_blockwise:
        print(
            "ff_blockwise endpoints: "
            + ",".join(str(i + 1) for i in ff_block_endpoints)
            + f" (total_layers={len(model.layers)})"
        )

    risk_head = None
    risk_targets_by_horizon: list[list[float | None]] | None = None
    risk_horizons_effective: list[int] = []
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
                risk_targets_by_horizon = []
                risk_horizons_effective = []
                for horizon in risk_horizons:
                    targets_h, _, _ = _compute_risk_targets(
                        prices_path=prices_path,
                        ticker=str(risk_ticker),
                        dates=dates,
                        horizon=int(horizon),
                        standardize=risk_standardize,
                        max_abs_logret=risk_max_abs_logret,
                        cache_dir=str(risk_cache_dir) if risk_cache_dir else None,
                    )
                    risk_targets_by_horizon.append(targets_h)
                    risk_horizons_effective.append(int(horizon))
                if not risk_targets_by_horizon:
                    raise ValueError("no valid risk horizons configured")
            except Exception as exc:
                print(f"risk_head disabled: {exc}")
                risk_head_enabled = False

    if risk_head_enabled:
        print(f"risk_head output dim: {len(risk_horizons_effective)}")
        risk_head = torch.nn.Linear(hidden_dim, len(risk_horizons_effective)).to(device)

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

    if auto_tune and device.type in ("cuda", "mps"):
        print(f"Auto-tuning batch size for {device.type.upper()}...")
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
                    self_contrastive_temp,
                    self_contrastive_max_graphs,
                    self_contrastive_view_mode,
                    self_contrastive_view_noise_std,
                    self_contrastive_ff_weight,
                    self_contrastive_ff_neg_mode,
                    self_contrastive_ff_noise_std,
                    self_contrastive_ff_target,
                    distance_forward_weight,
                    distance_forward_margin,
                    distance_forward_max_graphs,
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
                empty_device_cache(device)

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
                        self_contrastive_temp,
                        self_contrastive_max_graphs,
                        self_contrastive_view_mode,
                        self_contrastive_view_noise_std,
                        self_contrastive_ff_weight,
                        self_contrastive_ff_neg_mode,
                        self_contrastive_ff_noise_std,
                        self_contrastive_ff_target,
                        distance_forward_weight,
                        distance_forward_margin,
                        distance_forward_max_graphs,
                    )
                    best_bs = test_bs
                    break
                except RuntimeError as exc:
                    if _is_oom(exc):
                        test_bs = int(test_bs / auto_tune_factor)
                        continue
                    raise
                finally:
                    empty_device_cache(device)

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
    optim_params = list(model.parameters())
    if risk_head is not None:
        optim_params.extend(list(risk_head.parameters()))
    optim = _build_optimizer(optim_params, lr=lr, device=device, use_fused=fused_optimizer)

    if log_csv:
        log_path = Path(log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            f.write(
                "epoch,loss,g_pos,g_neg,hallucinate_ratio,gate_ratio,hall_hardness,"
                "hall_close_ratio,energy_penalty,risk_loss,dist_forward_loss,goodness_target_used,"
                "neg_mix_end_used,neg_gate_margin_used,hall_lr_used,hall_steps_used,"
                "hall_node_fraction_used\n"
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
        epoch_goodness_target = float(goodness_target)
        epoch_neg_mix_end = float(neg_mix_end)
        epoch_neg_gate_margin = float(neg_gate_margin)
        hall_cfg = _hall_cfg_for_epoch(epoch)
        hall_cfg_layer = _hall_cfg_for_epoch(
            epoch,
            corr_override=layerwise_hall_corr,
            mean_override=layerwise_hall_mean,
            std_override=layerwise_hall_std,
        )
        epoch_hall_lr = float(hall_cfg.lr)
        epoch_hall_steps = int(hall_cfg.steps)
        epoch_hall_node_fraction = float(hall_cfg.node_fraction)
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        batches = 0
        energy_penalty_sum = 0.0
        risk_loss_sum = 0.0
        risk_batches = 0
        dist_forward_sum = 0.0

        hall_used = 0
        total_used = 0
        hall_gated = 0
        hall_close_count = 0
        hall_close_total = 0
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

            step_idx = batches + 1
            apply_distance = (
                distance_forward_weight > 0
                and (step_idx % distance_forward_interval == 0)
            )
            step_scaler = scaler if (amp_enabled and (use_mode == "self_contrastive" or ff_layerwise)) else None

            if ff_multiscale:
                if step_scaler is not None:
                    with _autocast_if_needed(True, amp_dtype):
                        layers_pos = model(x, batch.edge_index, edge_weight=edge_weight, return_all=True)
                else:
                    layers_pos = model(x, batch.edge_index, edge_weight=edge_weight, return_all=True)
                hall_active = False
                dist_loss_val = 0.0

                if use_mode == "self_contrastive":
                    total_used += 1
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        x_view = _make_self_contrastive_view(
                            x,
                            batch.batch,
                            view_mode=self_contrastive_view_mode,
                            view_noise_std=self_contrastive_view_noise_std,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        layers_view = model(
                            x_view, batch.edge_index, edge_weight=edge_weight, return_all=True
                        )
                        batch_loss = 0.0
                        g_pos_last = 0.0
                        g_neg_last = 0.0
                        z_pos_last = None
                        z_neg_last = None
                        for h_p, h_v in zip(layers_pos, layers_view):
                            sc_loss, pos_score, neg_score, z_pos, z_view = _self_contrastive_batch_loss(
                                h_p,
                                h_v,
                                batch.batch,
                                temperature=self_contrastive_temp,
                                max_graphs=self_contrastive_max_graphs,
                            )
                            batch_loss = batch_loss + sc_loss
                            g_pos_last = float(pos_score.detach())
                            g_neg_last = float(neg_score.detach())
                            z_pos_last = z_pos
                            z_neg_last = permute_graph_embeddings(z_view)
                        batch_loss = batch_loss / max(1, len(layers_pos))
                        if apply_distance and z_pos_last is not None and z_neg_last is not None:
                            dist_loss_val = pairwise_distance_forward_loss(
                                z_pos_last,
                                z_neg_last,
                                margin=distance_forward_margin,
                                max_graphs=distance_forward_max_graphs,
                            )
                            batch_loss = batch_loss + distance_forward_weight * dist_loss_val
                        if self_contrastive_ff_weight > 0:
                            x_neg_aux = make_negative(
                                x,
                                batch.batch,
                                mode=self_contrastive_ff_neg_mode,
                                noise_std=self_contrastive_ff_noise_std,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            layers_neg_aux = model(
                                x_neg_aux,
                                batch.edge_index,
                                edge_weight=edge_weight,
                                return_all=True,
                            )
                            g_pos_aux = goodness(
                                layers_pos[-1], batch.batch, temperature=goodness_temp
                            )
                            g_neg_aux = goodness(
                                layers_neg_aux[-1], batch.batch, temperature=goodness_temp
                            )
                            ff_aux = ff_loss(
                                g_pos_aux,
                                g_neg_aux,
                                target=self_contrastive_ff_target,
                            )
                            batch_loss = batch_loss + self_contrastive_ff_weight * ff_aux
                else:
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
                        if hall_min_delta and hall_min_delta > 0:
                            delta = (
                                x_neg_hall[:, :returns_len] - x[:, :returns_len]
                            ).abs().mean()
                            hall_close_total += 1
                            if float(delta) < hall_min_delta:
                                hall_close_count += 1
                                x_neg_hall = make_negative(
                                    x,
                                    batch.batch,
                                    mode="shuffle+noise",
                                    noise_std=max(float(noise_std), float(hall_fallback_noise)),
                                    window_len=returns_len,
                                    summary_dim=summary_dim,
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

                    if apply_distance:
                        z_pos = global_mean_pool(layers_pos[-1], batch.batch)
                        z_neg_h = global_mean_pool(layers_neg_h[-1], batch.batch)
                        z_neg_t = global_mean_pool(layers_neg_t[-1], batch.batch)
                        dist_loss_h = pairwise_distance_forward_loss(
                            z_pos,
                            z_neg_h,
                            margin=distance_forward_margin,
                            max_graphs=distance_forward_max_graphs,
                        )
                        dist_loss_t = pairwise_distance_forward_loss(
                            z_pos,
                            z_neg_t,
                            margin=distance_forward_margin,
                            max_graphs=distance_forward_max_graphs,
                        )
                        dist_loss_val = 0.5 * (dist_loss_h + dist_loss_t)
                        batch_loss = batch_loss + distance_forward_weight * dist_loss_val

                    g_pos_last = goodness(
                        layers_pos[-1], batch.batch, temperature=goodness_temp
                    ).mean().item()
                    g_neg_h_last = goodness(
                        layers_neg_h[-1], batch.batch, temperature=goodness_temp
                    ).mean().item()
                    g_neg_t_last = goodness(
                        layers_neg_t[-1], batch.batch, temperature=goodness_temp
                    ).mean().item()
                    g_neg_last = (g_neg_h_last + g_neg_t_last) / 2.0

                energy_penalty_val = 0.0
                energy_penalty_weight_eff = (
                    energy_penalty_weight * self_contrastive_energy_penalty_scale
                    if use_mode == "self_contrastive"
                    else energy_penalty_weight
                )
                if energy_penalty_weight_eff > 0:
                    if energy_penalty_mode == "all":
                        energy_penalty_val = sum(
                            h.pow(2).mean() for h in layers_pos
                        ) / max(1, len(layers_pos))
                    else:
                        energy_penalty_val = layers_pos[-1].pow(2).mean()
                    batch_loss = batch_loss + energy_penalty_weight_eff * energy_penalty_val

                risk_loss_val = None
                if risk_head is not None and risk_targets_by_horizon is not None:
                    embed = global_mean_pool(layers_pos[-1], batch.batch)
                    risk_loss_val = _compute_multi_horizon_risk_loss(
                        risk_head=risk_head,
                        embeddings=embed,
                        graph_idx=batch.graph_idx,
                        risk_targets_by_horizon=risk_targets_by_horizon,
                        device=device,
                        risk_loss_type=str(risk_loss_type).strip().lower(),
                    )
                    if risk_loss_val is not None:
                        batch_loss = batch_loss + risk_loss_weight * risk_loss_val

                _optimizer_step(
                    optim=optim,
                    loss=batch_loss,
                    grad_clip=grad_clip,
                    clip_params=optim_params,
                    scaler=step_scaler,
                )

                total_loss += batch_loss.item()
                total_pos += g_pos_last
                total_neg += g_neg_last
                if hall_active:
                    hall_hardness_sum += (g_neg_last - g_pos_last)
                    hall_hardness_count += 1
                if energy_penalty_weight_eff > 0:
                    energy_penalty_sum += float(energy_penalty_val.detach())
                if risk_loss_val is not None:
                    risk_loss_sum += float(risk_loss_val.detach())
                    risk_batches += 1
                if distance_forward_weight > 0 and isinstance(dist_loss_val, torch.Tensor):
                    dist_forward_sum += float(dist_loss_val.detach())
            elif ff_layerwise:
                if ff_blockwise:
                    block_mode = "shuffle" if use_mode == "self_contrastive" else use_mode
                    hall_active = block_mode == "hallucinate"
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        layers_pos = model(x, batch.edge_index, edge_weight=edge_weight, return_all=True)

                    if hall_active:
                        x_neg = hallucinate_negative(
                            model,
                            x,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            batch.batch,
                            hall_cfg_layer,
                            edge_weight=edge_weight,
                        )
                        if hall_min_delta and hall_min_delta > 0:
                            delta = (x_neg[:, :returns_len] - x[:, :returns_len]).abs().mean()
                            hall_close_total += 1
                            if float(delta) < hall_min_delta:
                                hall_close_count += 1
                                x_neg = make_negative(
                                    x,
                                    batch.batch,
                                    mode="shuffle+noise",
                                    noise_std=max(float(layerwise_noise_std), float(hall_fallback_noise)),
                                    window_len=returns_len,
                                    summary_dim=summary_dim,
                                )
                        hall_used += 1
                    else:
                        x_neg = make_negative(
                            x,
                            batch.batch,
                            mode=layerwise_neg_mode,
                            noise_std=layerwise_noise_std,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                    total_used += 1

                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        layers_neg = model(x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True)
                    if hall_active:
                        last_idx = ff_block_endpoints[-1]
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            g_pos_probe = goodness(
                                layers_pos[last_idx], batch.batch, temperature=goodness_temp
                            ).mean().item()
                            g_neg_probe = goodness(
                                layers_neg[last_idx], batch.batch, temperature=goodness_temp
                            ).mean().item()
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
                            with _autocast_if_needed(step_scaler is not None, amp_dtype):
                                layers_neg = model(
                                    x_neg, batch.edge_index, edge_weight=edge_weight, return_all=True
                                )

                    block_loss = 0.0
                    block_gpos = 0.0
                    block_gneg = 0.0
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        for li in ff_block_endpoints:
                            g_pos = goodness(layers_pos[li], batch.batch, temperature=goodness_temp)
                            g_neg = goodness(layers_neg[li], batch.batch, temperature=goodness_temp)
                            block_loss = block_loss + ff_loss(g_pos, g_neg, target=goodness_target)
                            block_gpos += g_pos.mean().item()
                            block_gneg += g_neg.mean().item()
                    block_loss = block_loss / max(1, len(ff_block_endpoints))
                    _optimizer_step(
                        optim=optim,
                        loss=block_loss,
                        grad_clip=grad_clip,
                        clip_params=optim_params,
                        scaler=step_scaler,
                    )

                    avg_g_pos = block_gpos / max(1, len(ff_block_endpoints))
                    avg_g_neg = block_gneg / max(1, len(ff_block_endpoints))
                    total_loss += block_loss.item()
                    total_pos += avg_g_pos
                    total_neg += avg_g_neg
                    if hall_active:
                        hall_hardness_sum += (avg_g_neg - avg_g_pos)
                        hall_hardness_count += 1
                else:
                    x_in = x
                    layer_losses = 0.0
                    layer_gpos = 0.0
                    layer_gneg = 0.0
                    for li in range(len(model.layers)):
                        layer_mode = use_mode
                        if use_mode == "hallucinate" and li > 0:
                            layer_mode = "shuffle"
                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
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
                            if hall_min_delta and hall_min_delta > 0:
                                delta = (
                                    x_neg[:, :returns_len] - x_in[:, :returns_len]
                                ).abs().mean()
                                hall_close_total += 1
                                if float(delta) < hall_min_delta:
                                    hall_close_count += 1
                                    x_neg = make_negative(
                                        x_in,
                                        batch.batch,
                                        mode="shuffle+noise",
                                        noise_std=max(float(layerwise_noise_std), float(hall_fallback_noise)),
                                        window_len=returns_len,
                                        summary_dim=summary_dim,
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
                            with _autocast_if_needed(step_scaler is not None, amp_dtype):
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

                        with _autocast_if_needed(step_scaler is not None, amp_dtype):
                            h_neg = model.forward_layer(x_neg, batch.edge_index, edge_weight, li)
                            g_neg = goodness(h_neg, batch.batch, temperature=goodness_temp)
                            loss = ff_loss(g_pos, g_neg, target=goodness_target)
                        _optimizer_step(
                            optim=optim,
                            loss=loss,
                            grad_clip=grad_clip,
                            clip_params=optim_params,
                            scaler=step_scaler,
                        )

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
                hall_active = False
                dist_loss_val = 0.0

                if use_mode == "self_contrastive":
                    total_used += 1
                    with _autocast_if_needed(step_scaler is not None, amp_dtype):
                        h_pos = model(x, batch.edge_index, edge_weight=edge_weight)
                        x_view = _make_self_contrastive_view(
                            x,
                            batch.batch,
                            view_mode=self_contrastive_view_mode,
                            view_noise_std=self_contrastive_view_noise_std,
                            window_len=returns_len,
                            summary_dim=summary_dim,
                        )
                        h_view = model(x_view, batch.edge_index, edge_weight=edge_weight)
                        loss, pos_score, neg_score, z_pos, z_view = _self_contrastive_batch_loss(
                            h_pos,
                            h_view,
                            batch.batch,
                            temperature=self_contrastive_temp,
                            max_graphs=self_contrastive_max_graphs,
                        )
                        g_pos_val = float(pos_score.detach())
                        g_neg_val = float(neg_score.detach())
                        if apply_distance:
                            z_neg_dist = permute_graph_embeddings(z_view)
                            dist_loss_val = pairwise_distance_forward_loss(
                                z_pos,
                                z_neg_dist,
                                margin=distance_forward_margin,
                                max_graphs=distance_forward_max_graphs,
                            )
                            loss = loss + distance_forward_weight * dist_loss_val
                        if self_contrastive_ff_weight > 0:
                            x_neg_aux = make_negative(
                                x,
                                batch.batch,
                                mode=self_contrastive_ff_neg_mode,
                                noise_std=self_contrastive_ff_noise_std,
                                window_len=returns_len,
                                summary_dim=summary_dim,
                            )
                            h_neg_aux = model(x_neg_aux, batch.edge_index, edge_weight=edge_weight)
                            g_pos_aux = goodness(h_pos, batch.batch, temperature=goodness_temp)
                            g_neg_aux = goodness(h_neg_aux, batch.batch, temperature=goodness_temp)
                            ff_aux = ff_loss(
                                g_pos_aux,
                                g_neg_aux,
                                target=self_contrastive_ff_target,
                            )
                            loss = loss + self_contrastive_ff_weight * ff_aux
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
                        if hall_min_delta and hall_min_delta > 0:
                            delta = (x_neg[:, :returns_len] - x[:, :returns_len]).abs().mean()
                            hall_close_total += 1
                            if float(delta) < hall_min_delta:
                                hall_close_count += 1
                                x_neg = make_negative(
                                    x,
                                    batch.batch,
                                    mode="shuffle+noise",
                                    noise_std=max(float(noise_std), float(hall_fallback_noise)),
                                    window_len=returns_len,
                                    summary_dim=summary_dim,
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
                    g_pos_val = g_pos.mean().item()
                    g_neg_val = g_neg.mean().item()
                    if apply_distance:
                        z_pos = global_mean_pool(h_pos, batch.batch)
                        z_neg = global_mean_pool(h_neg, batch.batch)
                        dist_loss_val = pairwise_distance_forward_loss(
                            z_pos,
                            z_neg,
                            margin=distance_forward_margin,
                            max_graphs=distance_forward_max_graphs,
                        )
                        loss = loss + distance_forward_weight * dist_loss_val

                energy_penalty_val = 0.0
                energy_penalty_weight_eff = (
                    energy_penalty_weight * self_contrastive_energy_penalty_scale
                    if use_mode == "self_contrastive"
                    else energy_penalty_weight
                )
                if energy_penalty_weight_eff > 0:
                    energy_penalty_val = h_pos.pow(2).mean()
                    loss = loss + energy_penalty_weight_eff * energy_penalty_val

                risk_loss_val = None
                if risk_head is not None and risk_targets_by_horizon is not None:
                    embed = global_mean_pool(h_pos, batch.batch)
                    risk_loss_val = _compute_multi_horizon_risk_loss(
                        risk_head=risk_head,
                        embeddings=embed,
                        graph_idx=batch.graph_idx,
                        risk_targets_by_horizon=risk_targets_by_horizon,
                        device=device,
                        risk_loss_type=str(risk_loss_type).strip().lower(),
                    )
                    if risk_loss_val is not None:
                        loss = loss + risk_loss_weight * risk_loss_val
                _optimizer_step(
                    optim=optim,
                    loss=loss,
                    grad_clip=grad_clip,
                    clip_params=optim_params,
                    scaler=step_scaler,
                )

                total_loss += loss.item()
                total_pos += g_pos_val
                total_neg += g_neg_val
                if hall_active:
                    hall_hardness_sum += (g_neg_val - g_pos_val)
                    hall_hardness_count += 1
                if energy_penalty_weight_eff > 0:
                    energy_penalty_sum += float(energy_penalty_val.detach())
                if risk_loss_val is not None:
                    risk_loss_sum += float(risk_loss_val.detach())
                    risk_batches += 1
                if distance_forward_weight > 0 and isinstance(dist_loss_val, torch.Tensor):
                    dist_forward_sum += float(dist_loss_val.detach())
            batches += 1

        hall_ratio = hall_used / total_used if total_used else 0.0
        gate_ratio = hall_gated / total_used if total_used else 0.0
        hall_close_ratio = hall_close_count / hall_close_total if hall_close_total else 0.0
        hall_hardness = hall_hardness_sum / hall_hardness_count if hall_hardness_count else 0.0
        energy_penalty_epoch = energy_penalty_sum / batches if batches else 0.0
        risk_loss_epoch = risk_loss_sum / risk_batches if risk_batches else 0.0
        dist_forward_epoch = dist_forward_sum / batches if batches else 0.0
        epoch_loss = total_loss / batches if batches else 0.0
        epoch_pos = total_pos / batches if batches else 0.0
        epoch_neg = total_neg / batches if batches else 0.0

        target_updated = False
        if adaptive_target_enabled and batches and epoch >= adaptive_target_warmup:
            midpoint = 0.5 * (epoch_pos + epoch_neg) + adaptive_target_margin
            new_target = _clamp(
                (1.0 - adaptive_target_alpha) * goodness_target
                + adaptive_target_alpha * midpoint,
                adaptive_target_min,
                adaptive_target_max,
            )
            if abs(new_target - goodness_target) > 1e-8:
                goodness_target = new_target
                target_updated = True

        adapt_event = ""
        if (
            adaptive_hall_enabled
            and hall_used > 0
            and neg_mode in ("hallucinate", "schedule", "mix")
        ):
            hall_overused = hall_ratio >= adaptive_hall_ratio_high
            needs_harder = (
                hall_close_ratio >= adaptive_hall_close_high
                or hall_hardness <= adaptive_hall_hardness_low
            )
            too_hard = (
                hall_close_ratio < adaptive_hall_close_high
                and hall_hardness >= adaptive_hall_hardness_high
            )

            if hall_overused and hall_hardness <= adaptive_hall_hardness_low:
                hall_node_fraction = _clamp(
                    hall_node_fraction - adaptive_hall_node_inc,
                    0.0,
                    1.0,
                )
                neg_gate_margin = _clamp(
                    neg_gate_margin + adaptive_gate_margin_step,
                    adaptive_gate_margin_min,
                    adaptive_gate_margin_max,
                )
                if neg_mode == "mix":
                    neg_mix_end = _clamp(
                        neg_mix_end - adaptive_mix_step,
                        neg_mix_start,
                        adaptive_mix_end_max,
                    )
                _sync_hall_curriculum_end()
                adapt_event = "rebalance_mix"
            elif needs_harder:
                hall_steps = int(_clamp(float(hall_steps + adaptive_hall_steps_inc), 1.0, float(adaptive_hall_steps_max)))
                hall_lr = _clamp(hall_lr * adaptive_hall_lr_mult, 1e-4, adaptive_hall_lr_max)
                hall_mean = _clamp(hall_mean * adaptive_hall_reg_mult, adaptive_hall_reg_min, 5.0)
                hall_std = _clamp(hall_std * adaptive_hall_reg_mult, adaptive_hall_reg_min, 5.0)
                hall_corr = _clamp(hall_corr * adaptive_hall_reg_mult, adaptive_hall_reg_min, 5.0)
                hall_node_fraction = _clamp(hall_node_fraction + adaptive_hall_node_inc, 0.0, 1.0)
                if hall_min_delta > 0:
                    hall_min_delta = _clamp(
                        hall_min_delta * adaptive_hall_min_delta_mult,
                        adaptive_hall_min_delta_min,
                        hall_min_delta,
                    )
                neg_gate_margin = _clamp(
                    neg_gate_margin - adaptive_gate_margin_step,
                    adaptive_gate_margin_min,
                    adaptive_gate_margin_max,
                )
                if neg_mode == "mix":
                    neg_mix_end = _clamp(
                        neg_mix_end + adaptive_mix_step,
                        neg_mix_start,
                        adaptive_mix_end_max,
                    )
                _sync_hall_curriculum_end()
                adapt_event = "harder_neg"
            elif too_hard:
                hall_lr = _clamp(hall_lr / max(adaptive_hall_lr_mult, 1e-6), 1e-4, adaptive_hall_lr_max)
                hall_mean = _clamp(
                    hall_mean / max(adaptive_hall_reg_mult, 1e-6),
                    adaptive_hall_reg_min,
                    5.0,
                )
                hall_std = _clamp(
                    hall_std / max(adaptive_hall_reg_mult, 1e-6),
                    adaptive_hall_reg_min,
                    5.0,
                )
                hall_corr = _clamp(
                    hall_corr / max(adaptive_hall_reg_mult, 1e-6),
                    adaptive_hall_reg_min,
                    5.0,
                )
                hall_node_fraction = _clamp(hall_node_fraction - adaptive_hall_node_inc, 0.0, 1.0)
                neg_gate_margin = _clamp(
                    neg_gate_margin + adaptive_gate_margin_step,
                    adaptive_gate_margin_min,
                    adaptive_gate_margin_max,
                )
                if neg_mode == "mix":
                    neg_mix_end = _clamp(
                        neg_mix_end - adaptive_mix_step,
                        neg_mix_start,
                        adaptive_mix_end_max,
                    )
                _sync_hall_curriculum_end()
                adapt_event = "easier_neg"

        if adapt_event or target_updated:
            print(
                f"epoch {epoch}: adapt={adapt_event or 'target_only'} "
                f"target={goodness_target:.3f} mix_end={neg_mix_end:.3f} "
                f"gate_margin={neg_gate_margin:.3f} hall_steps={hall_steps} "
                f"hall_lr={hall_lr:.4f} hall_node_fraction={hall_node_fraction:.2f}"
            )

        if log_csv:
            with Path(log_csv).open("a") as f:
                f.write(
                    f"{epoch},{epoch_loss:.6f},"
                    f"{epoch_pos:.6f},{epoch_neg:.6f},"
                    f"{hall_ratio:.4f},{gate_ratio:.4f},{hall_hardness:.6f},"
                    f"{hall_close_ratio:.4f},{energy_penalty_epoch:.6f},{risk_loss_epoch:.6f},"
                    f"{dist_forward_epoch:.6f},"
                    f"{epoch_goodness_target:.6f},{epoch_neg_mix_end:.6f},{epoch_neg_gate_margin:.6f},"
                    f"{epoch_hall_lr:.6f},{epoch_hall_steps},{epoch_hall_node_fraction:.6f}\n"
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
            if "dist_forward_loss" in df.columns:
                plt.plot(df["epoch"], df["dist_forward_loss"], label="dist_forward")
            if "goodness_target_used" in df.columns:
                plt.plot(df["epoch"], df["goodness_target_used"], label="goodness_target")
            if "neg_mix_end_used" in df.columns:
                plt.plot(df["epoch"], df["neg_mix_end_used"], label="neg_mix_end")
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
