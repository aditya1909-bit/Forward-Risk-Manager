from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim, add_self_loops=False, normalize=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False))
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, edge_weight=edge_weight)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
        return h

    def forward_layer(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        layer_idx: int,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        norm = self.norms[layer_idx]
        h = layer(x, edge_index, edge_weight=edge_weight)
        h = norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h
