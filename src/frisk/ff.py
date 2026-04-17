from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def _segment_logsumexp(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    max_vals = torch.full((num_segments,), -torch.inf, device=values.device, dtype=values.dtype)
    if hasattr(max_vals, "scatter_reduce_"):
        max_vals.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    else:
        for gid in range(num_segments):
            mask = segment_ids == gid
            if mask.any():
                max_vals[gid] = values[mask].max()
    shifted = torch.exp(values - max_vals.index_select(0, segment_ids))
    sum_vals = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    sum_vals.index_add_(0, segment_ids, shifted)
    return max_vals + torch.log(sum_vals.clamp_min(1e-12))


def _segment_mean(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    sum_vals = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    cnt_vals = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    sum_vals.index_add_(0, segment_ids, values)
    cnt_vals.index_add_(0, segment_ids, torch.ones_like(values))
    return sum_vals / cnt_vals.clamp_min(1.0)


def goodness(
    h: torch.Tensor,
    batch: torch.Tensor,
    temperature: float = 1.0,
    critic: torch.nn.Module | None = None,
    norm: Literal["none", "layernorm"] = "none",
    reducer: Literal["logsumexp", "mean"] = "logsumexp",
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if h.numel() == 0:
        return torch.empty(0, device=h.device, dtype=h.dtype)
    if norm not in {"none", "layernorm"}:
        raise ValueError("norm must be 'none' or 'layernorm'")
    if reducer not in {"logsumexp", "mean"}:
        raise ValueError("reducer must be 'logsumexp' or 'mean'")

    if norm == "layernorm":
        h = F.layer_norm(h, (h.size(-1),))

    if critic is not None and reducer == "logsumexp":
        graph_energy_fn = getattr(critic, "graph_energy", None)
        if callable(graph_energy_fn):
            return graph_energy_fn(h, batch, temperature=temperature)

    if critic is not None:
        node_energy_fn = getattr(critic, "node_energy", None)
        if callable(node_energy_fn):
            node_energy = node_energy_fn(h)
        else:
            node_energy = critic(h)
            if node_energy.ndim == 2 and node_energy.size(1) == 1:
                node_energy = node_energy.squeeze(1)
            if node_energy.ndim != 1:
                raise ValueError(
                    f"Critic output must be [num_nodes] or [num_nodes, 1], got {tuple(node_energy.shape)}"
                )
            node_energy = F.softplus(node_energy)
    else:
        node_energy = (h * h).mean(dim=1)
    _, segment_ids = torch.unique(batch, sorted=True, return_inverse=True)
    num_segments = int(segment_ids.max().item()) + 1
    if reducer == "mean":
        return _segment_mean(node_energy, segment_ids, num_segments)
    scaled = node_energy / temperature
    lse = _segment_logsumexp(scaled, segment_ids, num_segments)
    return temperature * lse


def make_negative(
    x: torch.Tensor,
    batch: torch.Tensor,
    mode: Literal[
        "shuffle",
        "noise",
        "shuffle+noise",
        "time_flip",
        "shuffle+time_flip",
        "time_flip+noise",
        "block_bootstrap",
        "cross_asset_mix",
        "phase_randomize",
        "sector_swap",
        "factor_hard",
    ] = "shuffle",
    noise_std: float = 0.05,
    window_len: int | None = None,
    summary_dim: int = 0,
    sector_idx: int | None = None,
    factor_start_idx: int | None = None,
    factor_dim: int = 0,
    hard_topk: int = 3,
) -> torch.Tensor:
    out = x.clone()
    if out.numel() == 0:
        return out
    batch_idx = batch if batch.device == out.device else batch.to(out.device)

    def _split_window_and_tail(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if window_len is None or window_len <= 0 or window_len >= tensor.size(1):
            return tensor, tensor.new_zeros((tensor.size(0), 0))
        return tensor[:, :window_len], tensor[:, window_len:]

    def _merge_window_and_tail(window_part: torch.Tensor, tail_part: torch.Tensor) -> torch.Tensor:
        if tail_part.numel() == 0:
            return window_part
        return torch.cat([window_part, tail_part], dim=1)

    def _time_flip(tensor: torch.Tensor) -> torch.Tensor:
        if window_len is None:
            return torch.flip(tensor, dims=[1])
        w, tail = _split_window_and_tail(tensor)
        flipped = torch.flip(w, dims=[1])
        if summary_dim > 0 and tail.numel() > 0:
            s = tail[:, :summary_dim]
            rest = tail[:, summary_dim:]
            return _merge_window_and_tail(flipped, torch.cat([s, rest], dim=1))
        return _merge_window_and_tail(flipped, tail)

    def _block_bootstrap(tensor: torch.Tensor) -> torch.Tensor:
        w, tail = _split_window_and_tail(tensor)
        wlen = w.size(1)
        if wlen <= 1:
            return tensor
        block = max(2, min(5, wlen // 2 if wlen < 8 else 4))
        nblocks = (wlen + block - 1) // block
        out_w = torch.empty_like(w)
        max_start = max(1, wlen - block + 1)
        for i in range(w.size(0)):
            starts = torch.randint(0, max_start, (nblocks,), device=w.device)
            idx = []
            for s in starts.tolist():
                idx.extend(range(s, min(wlen, s + block)))
                if len(idx) >= wlen:
                    break
            out_w[i] = w[i, idx[:wlen]]
        return _merge_window_and_tail(out_w, tail)

    def _cross_asset_mix(tensor: torch.Tensor) -> torch.Tensor:
        mixed = tensor.clone()
        for gid in batch_idx.unique():
            idx = (batch_idx == gid).nonzero(as_tuple=False).view(-1)
            if idx.numel() <= 1:
                continue
            peer = idx[torch.randperm(idx.numel(), device=out.device)]
            alpha = 0.25 + 0.5 * torch.rand((idx.numel(), 1), device=out.device, dtype=out.dtype)
            mixed.index_copy_(
                0,
                idx,
                alpha * mixed.index_select(0, idx) + (1.0 - alpha) * mixed.index_select(0, peer),
            )
        return mixed

    def _phase_randomize(tensor: torch.Tensor) -> torch.Tensor:
        w, tail = _split_window_and_tail(tensor)
        wlen = w.size(1)
        if wlen <= 2:
            return tensor
        spec = torch.fft.rfft(w, dim=1)
        if spec.size(1) > 2:
            rand_phase = 2.0 * torch.pi * torch.rand(
                spec.size(0), spec.size(1) - 2, device=spec.device, dtype=w.dtype
            )
            spec[:, 1:-1] = spec[:, 1:-1].abs() * torch.exp(1j * rand_phase)
        w_rand = torch.fft.irfft(spec, n=wlen, dim=1)
        return _merge_window_and_tail(w_rand, tail)

    def _sector_swap(tensor: torch.Tensor) -> torch.Tensor:
        # Swap samples within the same graph and inferred sector bucket.
        if sector_idx is None or int(sector_idx) < 0 or int(sector_idx) >= tensor.size(1):
            return tensor
        out_sec = tensor.clone()
        sec = tensor[:, int(sector_idx)]
        did_swap = False
        for gid in batch_idx.unique():
            idx = (batch_idx == gid).nonzero(as_tuple=False).view(-1)
            if idx.numel() <= 1:
                continue
            sec_vals = sec.index_select(0, idx)
            sec_keys = torch.round(sec_vals * 1_000.0).to(torch.long)
            for key in sec_keys.unique():
                grp = idx[(sec_keys == key).nonzero(as_tuple=False).view(-1)]
                if grp.numel() <= 1:
                    continue
                perm = grp[torch.randperm(grp.numel(), device=out_sec.device)]
                out_sec.index_copy_(0, grp, out_sec.index_select(0, perm))
                did_swap = True
        if not did_swap:
            return tensor
        return out_sec

    def _factor_hard(tensor: torch.Tensor) -> torch.Tensor:
        # Replace return windows using nearest-neighbor nodes in factor space.
        w, tail = _split_window_and_tail(tensor)
        if w.numel() == 0:
            return tensor

        factors = None
        if (
            factor_start_idx is not None
            and int(factor_start_idx) >= 0
            and int(factor_dim) > 0
            and int(factor_start_idx) + int(factor_dim) <= tensor.size(1)
        ):
            factors = tensor[:, int(factor_start_idx) : int(factor_start_idx) + int(factor_dim)]
        elif tail.numel() > 0:
            fd = min(4, tail.size(1))
            factors = tail[:, :fd]
        else:
            factors = torch.stack([w.mean(dim=1), w.std(dim=1, unbiased=False)], dim=1)

        out_w = w.clone()
        k_req = max(1, int(hard_topk))
        for gid in batch_idx.unique():
            idx = (batch_idx == gid).nonzero(as_tuple=False).view(-1)
            n = int(idx.numel())
            if n <= 1:
                continue
            local_f = factors.index_select(0, idx)
            local_f = (local_f - local_f.mean(dim=0, keepdim=True)) / (
                local_f.std(dim=0, unbiased=False, keepdim=True) + 1e-6
            )
            dist = torch.cdist(local_f, local_f, p=2)
            dist.fill_diagonal_(float("inf"))
            k = min(k_req, n - 1)
            nn = torch.topk(dist, k=k, largest=False).indices
            nn_col = torch.randint(0, k, (n,), device=nn.device)
            src_local = nn[torch.arange(n, device=nn.device), nn_col]
            src_idx = idx.index_select(0, src_local)
            out_w.index_copy_(0, idx, w.index_select(0, src_idx))
        return _merge_window_and_tail(out_w, tail)

    if mode in ("time_flip", "shuffle+time_flip", "time_flip+noise"):
        out = _time_flip(out)
    elif mode == "block_bootstrap":
        out = _block_bootstrap(out)
    elif mode == "cross_asset_mix":
        out = _cross_asset_mix(out)
    elif mode == "phase_randomize":
        out = _phase_randomize(out)
    elif mode == "sector_swap":
        out = _sector_swap(out)
    elif mode == "factor_hard":
        out = _factor_hard(out)

    if mode in ("shuffle", "shuffle+noise", "shuffle+time_flip"):
        # Group-preserving shuffle: each node is reassigned to another node in the same graph.
        rand = torch.rand(batch_idx.numel(), device=out.device, dtype=torch.float64)
        try:
            order = torch.argsort(batch_idx, stable=True)
            shuffled = torch.argsort(batch_idx.to(torch.float64) + rand, stable=True)
        except TypeError:
            order = torch.argsort(batch_idx)
            shuffled = torch.argsort(batch_idx.to(torch.float64) + rand)

        source_for_dest = torch.empty_like(order)
        source_for_dest[order] = shuffled
        out = out.index_select(0, source_for_dest)

    if mode in ("noise", "shuffle+noise", "time_flip+noise") and noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return out


def ff_loss(
    g_pos: torch.Tensor,
    g_neg: torch.Tensor,
    target: float = 1.0,
    margin: float = 0.0,
    margin_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    loss_type: Literal["softplus_margin", "symba"] = "softplus_margin",
) -> torch.Tensor:
    if loss_type == "symba":
        scale = max(0.0, float(margin_weight))
        if scale <= 0:
            scale = 1.0
        loss = F.softplus(scale * (g_neg - g_pos + float(margin)))
    else:
        # Encourage g_pos > target and g_neg < target
        loss_pos = F.softplus(target - g_pos)
        loss_neg = F.softplus(g_neg - target)
        loss = loss_pos + loss_neg
        if margin > 0 and margin_weight > 0:
            gap = g_pos - g_neg
            loss = loss + float(margin_weight) * F.softplus(float(margin) - gap)
    if sample_weight is not None:
        weight = sample_weight.to(device=loss.device, dtype=loss.dtype).reshape(-1)
        if weight.numel() != loss.numel():
            raise ValueError(
                f"sample_weight shape mismatch: expected {loss.numel()} values, got {weight.numel()}"
            )
        return (loss * weight).sum() / weight.sum().clamp_min(1e-12)
    return loss.mean()


def permute_graph_embeddings(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError("Expected z to have shape [num_graphs, dim]")
    n = z.size(0)
    if n <= 1:
        return z
    base = torch.arange(n, device=z.device)
    perm = base.clone()
    # Try randomized derangements first; fall back to a deterministic roll.
    for _ in range(8):
        candidate = torch.randperm(n, device=z.device)
        if not torch.any(candidate == base):
            perm = candidate
            break
    else:
        perm = torch.roll(base, shifts=1, dims=0)
    return z.index_select(0, perm)


def self_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.2,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if z_a.ndim != 2 or z_b.ndim != 2:
        raise ValueError("Expected z_a and z_b to have shape [num_graphs, dim]")
    if z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if z_a.size(0) == 0:
        zero = torch.zeros((), device=z_a.device, dtype=z_a.dtype)
        return zero, zero, zero

    z_a_n = F.normalize(z_a, p=2, dim=1)
    z_b_n = F.normalize(z_b, p=2, dim=1)
    logits_ab = (z_a_n @ z_b_n.T) / float(temperature)
    labels = torch.arange(z_a_n.size(0), device=z_a.device, dtype=torch.long)
    loss_ab = F.cross_entropy(logits_ab, labels, reduction="none")
    loss_ba = F.cross_entropy(logits_ab.T, labels, reduction="none")
    if sample_weight is not None:
        weight = sample_weight.to(device=loss_ab.device, dtype=loss_ab.dtype).reshape(-1)
        if weight.numel() != loss_ab.numel():
            raise ValueError(
                f"sample_weight shape mismatch: expected {loss_ab.numel()} values, got {weight.numel()}"
            )
        loss = 0.5 * (
            (loss_ab * weight).sum() / weight.sum().clamp_min(1e-12)
            + (loss_ba * weight).sum() / weight.sum().clamp_min(1e-12)
        )
    else:
        loss = 0.5 * (loss_ab.mean() + loss_ba.mean())

    sim = z_a_n @ z_b_n.T
    pos_sim = sim.diag().mean()
    if sim.size(0) > 1:
        mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        neg_sim = sim[mask].mean()
    else:
        neg_sim = torch.zeros((), device=sim.device, dtype=sim.dtype)
    return loss, pos_sim, neg_sim


def self_contrastive_retrieval_accuracy(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    if z_a.ndim != 2 or z_b.ndim != 2:
        raise ValueError("Expected z_a and z_b to have shape [num_graphs, dim]")
    if z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must have the same shape")
    if z_a.size(0) == 0:
        return torch.zeros((), device=z_a.device, dtype=z_a.dtype)

    z_a_n = F.normalize(z_a, p=2, dim=1)
    z_b_n = F.normalize(z_b, p=2, dim=1)
    sim = z_a_n @ z_b_n.T
    labels = torch.arange(sim.size(0), device=sim.device, dtype=torch.long)
    acc_ab = (sim.argmax(dim=1) == labels).float().mean()
    acc_ba = (sim.argmax(dim=0) == labels).float().mean()
    return 0.5 * (acc_ab + acc_ba)


def pairwise_distance_forward_loss(
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    margin: float = 0.15,
    max_graphs: int = 0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if z_pos.ndim != 2 or z_neg.ndim != 2:
        raise ValueError("Expected z_pos and z_neg to have shape [num_graphs, dim]")
    if z_pos.shape != z_neg.shape:
        raise ValueError("z_pos and z_neg must have the same shape")
    if z_pos.size(0) == 0:
        return torch.zeros((), device=z_pos.device, dtype=z_pos.dtype)

    if max_graphs and z_pos.size(0) > int(max_graphs):
        idx = torch.randperm(z_pos.size(0), device=z_pos.device)[: int(max_graphs)]
        z_pos = z_pos.index_select(0, idx)
        z_neg = z_neg.index_select(0, idx)
        if sample_weight is not None:
            sample_weight = sample_weight.to(device=z_pos.device).index_select(0, idx)

    d_neg = torch.norm(z_pos - z_neg, p=2, dim=1)
    if z_pos.size(0) == 1:
        loss = F.relu(float(margin) - d_neg)
        if sample_weight is not None:
            weight = sample_weight.to(device=loss.device, dtype=loss.dtype).reshape(-1)
            return (loss * weight).sum() / weight.sum().clamp_min(1e-12)
        return loss.mean()

    dist_pp = torch.cdist(z_pos, z_pos, p=2)
    eye = torch.eye(dist_pp.size(0), dtype=torch.bool, device=dist_pp.device)
    nearest_pos = dist_pp.masked_fill(eye, float("inf")).min(dim=1).values
    loss = F.relu(float(margin) + nearest_pos - d_neg)
    if sample_weight is not None:
        weight = sample_weight.to(device=loss.device, dtype=loss.dtype).reshape(-1)
        if weight.numel() != loss.numel():
            raise ValueError(
                f"sample_weight shape mismatch: expected {loss.numel()} values, got {weight.numel()}"
            )
        return (loss * weight).sum() / weight.sum().clamp_min(1e-12)
    return loss.mean()


def rank_spread_loss(
    scores: torch.Tensor,
    top_frac: float = 0.2,
    margin: float = 0.1,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("Expected scores to have shape [num_graphs]")
    if scores.numel() == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    n = scores.numel()
    if n == 1:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    frac = float(max(1e-3, min(0.49, top_frac)))
    k = max(1, int(round(frac * n)))
    k = min(k, n // 2)
    if sample_weight is None:
        top = torch.topk(scores, k=k, largest=True).values.mean()
        bot = torch.topk(scores, k=k, largest=False).values.mean()
    else:
        weight = sample_weight.to(device=scores.device, dtype=scores.dtype).reshape(-1)
        if weight.numel() != scores.numel():
            raise ValueError(
                f"sample_weight shape mismatch: expected {scores.numel()} values, got {weight.numel()}"
            )
        top_idx = torch.topk(scores, k=k, largest=True).indices
        bot_idx = torch.topk(scores, k=k, largest=False).indices
        top_w = weight.index_select(0, top_idx)
        bot_w = weight.index_select(0, bot_idx)
        top = (scores.index_select(0, top_idx) * top_w).sum() / top_w.sum().clamp_min(1e-12)
        bot = (scores.index_select(0, bot_idx) * bot_w).sum() / bot_w.sum().clamp_min(1e-12)
    return F.softplus(float(margin) - (top - bot))
