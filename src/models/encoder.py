from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGEEncoder(nn.Module):
    """Lightweight mean-aggregation GraphSAGE encoder (no PyG dependency)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        for layer_idx, layer in enumerate(self.layers):
            if edge_index.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                neighbor_sum = torch.zeros_like(x)
                neighbor_count = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
                neighbor_sum.index_add_(0, dst, x[src])
                neighbor_count.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=x.device, dtype=x.dtype))
                neighbor_mean = neighbor_sum / neighbor_count.clamp(min=1.0)
                aggregated = 0.5 * x + 0.5 * neighbor_mean
            else:
                aggregated = x
            x = layer(aggregated)
            if layer_idx < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
