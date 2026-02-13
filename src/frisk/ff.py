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


def goodness(
    h: torch.Tensor,
    batch: torch.Tensor,
    temperature: float = 1.0,
    critic: torch.nn.Module | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if h.numel() == 0:
        return torch.empty(0, device=h.device, dtype=h.dtype)

    if critic is not None:
        graph_energy_fn = getattr(critic, "graph_energy", None)
        if callable(graph_energy_fn):
            return graph_energy_fn(h, batch, temperature=temperature)
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
    scaled = node_energy / temperature
    lse = _segment_logsumexp(scaled, segment_ids, int(segment_ids.max().item()) + 1)
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
    ] = "shuffle",
    noise_std: float = 0.05,
    window_len: int | None = None,
    summary_dim: int = 0,
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

    if mode in ("time_flip", "shuffle+time_flip", "time_flip+noise"):
        out = _time_flip(out)
    elif mode == "block_bootstrap":
        out = _block_bootstrap(out)
    elif mode == "cross_asset_mix":
        out = _cross_asset_mix(out)
    elif mode == "phase_randomize":
        out = _phase_randomize(out)

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
) -> torch.Tensor:
    # Encourage g_pos > target and g_neg < target
    loss_pos = F.softplus(target - g_pos)
    loss_neg = F.softplus(g_neg - target)
    loss = loss_pos + loss_neg
    if margin > 0 and margin_weight > 0:
        gap = g_pos - g_neg
        loss = loss + float(margin_weight) * F.softplus(float(margin) - gap)
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
    loss = 0.5 * (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ab.T, labels))

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

    d_neg = torch.norm(z_pos - z_neg, p=2, dim=1)
    if z_pos.size(0) == 1:
        return F.relu(float(margin) - d_neg).mean()

    dist_pp = torch.cdist(z_pos, z_pos, p=2)
    eye = torch.eye(dist_pp.size(0), dtype=torch.bool, device=dist_pp.device)
    nearest_pos = dist_pp.masked_fill(eye, float("inf")).min(dim=1).values
    return F.relu(float(margin) + nearest_pos - d_neg).mean()
