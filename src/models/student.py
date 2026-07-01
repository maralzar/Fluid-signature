from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoder import GraphSAGEEncoder


class ReasoningSignatureModel(nn.Module):
    """GNN student with semantic/topological split and RBT decoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        sem_dim: int = 64,
        topo_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        rbt_shape: tuple[int, int, int] = (8, 5, 6),
        topo_bias_eps: float = 0.05,
    ) -> None:
        super().__init__()
        self.rbt_shape = rbt_shape
        self.rbt_dim = rbt_shape[0] * rbt_shape[1] * rbt_shape[2]
        self.topo_bias_eps = topo_bias_eps

        self.encoder = GraphSAGEEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.sem_head = nn.Linear(hidden_dim, sem_dim)
        self.topo_head = nn.Linear(hidden_dim, topo_dim)
        self.decoder = nn.Sequential(
            nn.Linear(sem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.rbt_dim),
        )
        self.topo_to_rbt = nn.Linear(topo_dim, self.rbt_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x, edge_index)
        z_sem = self.sem_head(hidden)
        z_topo = self.topo_head(hidden)
        rbt_flat = self.decoder(z_sem) + self.topo_bias_eps * self.topo_to_rbt(z_topo)
        rbt_hat = rbt_flat.view(-1, *self.rbt_shape)
        return {
            "hidden": hidden,
            "z_sem": z_sem,
            "z_topo": z_topo,
            "rbt_hat": rbt_hat,
        }
