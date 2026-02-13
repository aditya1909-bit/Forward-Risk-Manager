from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


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


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: Literal["gcn", "sage", "gat"] = "gcn",
        gat_heads: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        conv_mode = str(conv_type).strip().lower()
        if conv_mode not in {"gcn", "sage", "gat"}:
            raise ValueError("conv_type must be one of: gcn, sage, gat")
        heads = max(1, int(gat_heads))

        def _make_layer(in_channels: int, out_channels: int):
            if conv_mode == "sage":
                return SAGEConv(in_channels, out_channels)
            if conv_mode == "gat":
                return GATConv(
                    in_channels,
                    out_channels,
                    heads=heads,
                    concat=False,
                    add_self_loops=False,
                    dropout=max(0.0, float(dropout)),
                )
            # Keep normalization off in GCN mode: FF goodness relies on energy magnitude.
            return GCNConv(in_channels, out_channels, add_self_loops=False, normalize=False)

        self.layers = nn.ModuleList()
        self.layers.append(_make_layer(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(_make_layer(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        return_all: bool = False,
    ) -> torch.Tensor:
        h = x
        outputs: list[torch.Tensor] = []
        for layer in self.layers:
            if isinstance(layer, GCNConv):
                h = layer(h, edge_index, edge_weight=edge_weight)
            else:
                h = layer(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
            outputs.append(h)
        if return_all:
            return outputs  # type: ignore[return-value]
        return h

    def forward_layer(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        layer_idx: int,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        if isinstance(layer, GCNConv):
            h = layer(x, edge_index, edge_weight=edge_weight)
        else:
            h = layer(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class GCNEncoder(GraphEncoder):
    """Backward-compatible alias for existing training/eval scripts."""


class EnergyCritic(nn.Module):
    """Critic that maps node embeddings to graph energy (goodness)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 0,
        num_layers: int = 2,
        dropout: float = 0.0,
        positive_activation: Literal["softplus", "square"] = "softplus",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if positive_activation not in {"softplus", "square"}:
            raise ValueError("positive_activation must be 'softplus' or 'square'")

        width = int(hidden_dim) if int(hidden_dim) > 0 else int(in_dim)
        self.positive_activation = positive_activation
        self.dropout = nn.Dropout(max(0.0, float(dropout)))

        layers: list[nn.Linear] = []
        if num_layers <= 1:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, width))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(width, width))
            layers.append(nn.Linear(width, 1))
        self.layers = nn.ModuleList(layers)

    def _positive(self, x: torch.Tensor) -> torch.Tensor:
        if self.positive_activation == "square":
            return x * x
        return F.softplus(x)

    def node_energy(self, h: torch.Tensor) -> torch.Tensor:
        y = h
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i != last:
                y = F.relu(y)
                y = self.dropout(y)
        if y.ndim == 2 and y.size(1) == 1:
            y = y.squeeze(1)
        if y.ndim != 1:
            raise ValueError(f"Expected critic output shape [num_nodes], got {tuple(y.shape)}")
        return self._positive(y)

    def graph_energy(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if h.numel() == 0:
            return torch.empty(0, device=h.device, dtype=h.dtype)
        node_energy = self.node_energy(h)
        _, segment_ids = torch.unique(batch, sorted=True, return_inverse=True)
        scaled = node_energy / float(temperature)
        lse = _segment_logsumexp(scaled, segment_ids, int(segment_ids.max().item()) + 1)
        return float(temperature) * lse
