from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, RGCNConv, SAGEConv, global_mean_pool


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


class ResidualEdgeWeightAdapter(nn.Module):
    """Learned residual edge reweighting on top of prior graph weights."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        max_delta: float = 0.25,
        detach_features: bool = True,
    ):
        super().__init__()
        hid = max(4, int(hidden_dim))
        self.max_delta = max(0.0, float(max_delta))
        self.detach_features = bool(detach_features)
        self.net = nn.Sequential(
            nn.Linear(2 * int(in_dim), hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        x_used = x.detach() if self.detach_features else x
        feat = torch.cat([x_used.index_select(0, src), x_used.index_select(0, dst)], dim=1)
        delta = torch.tanh(self.net(feat).squeeze(-1))
        if edge_weight is None:
            base = torch.ones_like(delta)
        else:
            base = edge_weight
            if base.ndim == 2 and base.size(1) == 1:
                base = base.squeeze(1)
            base = base.to(dtype=delta.dtype, device=delta.device)
        scale = torch.tanh(self.scale) * float(self.max_delta)
        return base + scale * delta


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: Literal["gcn", "sage", "gat", "rgcn"] = "gcn",
        gat_heads: int = 2,
        rgcn_num_relations: int = 8,
        residual_edge_enabled: bool = False,
        residual_edge_hidden_dim: int = 32,
        residual_edge_max_delta: float = 0.25,
        residual_edge_detach_features: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        conv_mode = str(conv_type).strip().lower()
        if conv_mode not in {"gcn", "sage", "gat", "rgcn"}:
            raise ValueError("conv_type must be one of: gcn, sage, gat, rgcn")
        heads = max(1, int(gat_heads))
        num_relations = max(2, int(rgcn_num_relations))
        self.conv_mode = conv_mode
        self.rgcn_num_relations = int(num_relations)

        def _make_layer(in_channels: int, out_channels: int):
            if conv_mode == "sage":
                return SAGEConv(in_channels, out_channels)
            if conv_mode == "rgcn":
                return RGCNConv(in_channels, out_channels, num_relations=int(num_relations))
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
        self.residual_edge_in_dim = int(in_dim)
        self.residual_edge_adapter = (
            ResidualEdgeWeightAdapter(
                in_dim=in_dim,
                hidden_dim=residual_edge_hidden_dim,
                max_delta=residual_edge_max_delta,
                detach_features=residual_edge_detach_features,
            )
            if bool(residual_edge_enabled)
            else None
        )

    def _effective_edge_weight(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.residual_edge_adapter is None:
            return edge_weight
        if int(x.size(-1)) != int(self.residual_edge_in_dim):
            return edge_weight
        return self.residual_edge_adapter(x, edge_index, edge_weight=edge_weight)

    def _effective_edge_type(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.conv_mode != "rgcn":
            return None
        num_edges = int(edge_index.size(1))
        if edge_type is None:
            if edge_weight is not None:
                ew = edge_weight
                if ew.ndim == 2 and ew.size(1) == 1:
                    ew = ew.squeeze(1)
                if ew.ndim == 1 and int(ew.numel()) == num_edges:
                    # Fallback relation typing: signed edges.
                    sign_type = (ew.to(device=edge_index.device) < 0).to(torch.long) + 1
                    return sign_type.clamp_min(0).clamp_max(self.rgcn_num_relations - 1)
            return torch.zeros(num_edges, dtype=torch.long, device=edge_index.device)
        et = edge_type
        if et.ndim == 2 and et.size(1) == 1:
            et = et.squeeze(1)
        if et.ndim != 1 or int(et.numel()) != num_edges:
            return torch.zeros(num_edges, dtype=torch.long, device=edge_index.device)
        et = et.to(device=edge_index.device, dtype=torch.long)
        return et.clamp_min(0).clamp_max(self.rgcn_num_relations - 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        return_all: bool = False,
    ) -> torch.Tensor:
        h = x
        edge_weight_eff = self._effective_edge_weight(x, edge_index, edge_weight)
        edge_type_eff = self._effective_edge_type(edge_index, edge_weight_eff, edge_type)
        outputs: list[torch.Tensor] = []
        for layer in self.layers:
            if isinstance(layer, GCNConv):
                h = layer(h, edge_index, edge_weight=edge_weight_eff)
            elif isinstance(layer, RGCNConv):
                h = layer(h, edge_index, edge_type_eff)
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
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        edge_weight_eff = self._effective_edge_weight(x, edge_index, edge_weight)
        edge_type_eff = self._effective_edge_type(edge_index, edge_weight_eff, edge_type)
        if isinstance(layer, GCNConv):
            h = layer(x, edge_index, edge_weight=edge_weight_eff)
        elif isinstance(layer, RGCNConv):
            h = layer(x, edge_index, edge_type_eff)
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


class SequenceEnergyCritic(nn.Module):
    """Sequence critic over graph embeddings (graph-order within a batch)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 0,
        num_layers: int = 1,
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
        self.rnn = nn.GRU(
            input_size=int(in_dim),
            hidden_size=width,
            num_layers=int(num_layers),
            dropout=max(0.0, float(dropout)) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(width, 1)

    def _positive(self, x: torch.Tensor) -> torch.Tensor:
        if self.positive_activation == "square":
            return x * x
        return F.softplus(x)

    def sequence_energy(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z_in = z.unsqueeze(0)  # [1, T, D]
        elif z.ndim == 3:
            z_in = z
        else:
            raise ValueError(f"Expected z shape [T, D] or [B, T, D], got {tuple(z.shape)}")
        y, _ = self.rnn(z_in)
        e = self.head(y).squeeze(-1)
        e = self._positive(e)
        return e.squeeze(0) if z.ndim == 2 else e


class EnergyCriticEnsemble(nn.Module):
    """Averages graph energies from multiple critic members."""

    def __init__(
        self,
        critics: list[nn.Module],
        member_weights: list[float] | None = None,
    ):
        super().__init__()
        if not critics:
            raise ValueError("critics must be non-empty")
        self.members = nn.ModuleList(critics)
        if member_weights is None:
            weights = torch.ones(len(critics), dtype=torch.float32)
        else:
            if len(member_weights) != len(critics):
                raise ValueError("member_weights length must match critics")
            weights = torch.tensor(member_weights, dtype=torch.float32)
        weights = weights.clamp_min(0.0)
        if float(weights.sum().item()) <= 0:
            weights = torch.ones_like(weights)
        self.register_buffer("member_weights", weights / weights.sum())

    def member_graph_energy(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        outs = []
        for member in self.members:
            graph_energy_fn = getattr(member, "graph_energy", None)
            if not callable(graph_energy_fn):
                raise ValueError("Ensemble member must expose graph_energy(...)")
            outs.append(graph_energy_fn(h, batch, temperature=temperature))
        return torch.stack(outs, dim=0)

    def member_node_energy(self, h: torch.Tensor) -> torch.Tensor:
        outs = []
        for member in self.members:
            node_energy_fn = getattr(member, "node_energy", None)
            if not callable(node_energy_fn):
                raise ValueError("Ensemble member must expose node_energy(...)")
            outs.append(node_energy_fn(h))
        return torch.stack(outs, dim=0)

    def graph_energy(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        ge = self.member_graph_energy(h, batch, temperature=temperature)
        w = self.member_weights.to(device=ge.device, dtype=ge.dtype).view(-1, 1)
        return (w * ge).sum(dim=0)

    def node_energy(self, h: torch.Tensor) -> torch.Tensor:
        ne = self.member_node_energy(h)
        w = self.member_weights.to(device=ne.device, dtype=ne.dtype).view(-1, 1)
        return (w * ne).sum(dim=0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.node_energy(h)


class CompositeEnergyCritic(nn.Module):
    """Combines node-based and sequence-based graph energy."""

    def __init__(
        self,
        base_critic: nn.Module,
        sequence_critic: SequenceEnergyCritic | None = None,
        sequence_weight: float = 0.0,
    ):
        super().__init__()
        self.base_critic = base_critic
        self.sequence_critic = sequence_critic
        self.sequence_weight = float(sequence_weight)

    def graph_energy(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        base_graph_energy = getattr(self.base_critic, "graph_energy", None)
        if not callable(base_graph_energy):
            raise ValueError("base_critic must expose graph_energy(...)")
        g = base_graph_energy(h, batch, temperature=temperature)
        if self.sequence_critic is None or self.sequence_weight == 0:
            return g
        z = global_mean_pool(h, batch)
        seq_energy = self.sequence_critic.sequence_energy(z)
        if seq_energy.shape != g.shape:
            raise ValueError(
                f"sequence energy shape mismatch: seq={tuple(seq_energy.shape)} vs base={tuple(g.shape)}"
            )
        return g + float(self.sequence_weight) * seq_energy.to(dtype=g.dtype, device=g.device)

    def node_energy(self, h: torch.Tensor) -> torch.Tensor:
        node_energy_fn = getattr(self.base_critic, "node_energy", None)
        if callable(node_energy_fn):
            return node_energy_fn(h)
        y = self.base_critic(h)
        if y.ndim == 2 and y.size(1) == 1:
            y = y.squeeze(1)
        if y.ndim != 1:
            raise ValueError(f"Expected critic output shape [num_nodes], got {tuple(y.shape)}")
        return F.softplus(y)

    def member_node_energy(self, h: torch.Tensor) -> torch.Tensor:
        member_node_energy_fn = getattr(self.base_critic, "member_node_energy", None)
        if callable(member_node_energy_fn):
            return member_node_energy_fn(h)
        node_energy = self.node_energy(h)
        return node_energy.unsqueeze(0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.node_energy(h)
