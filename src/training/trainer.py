from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.models.student import ReasoningSignatureModel
from src.training.losses import (
    behavior_reconstruction_loss,
    contrastive_behavior_loss,
    orthogonality_loss,
    topology_preservation_loss,
)


def compute_losses(
    outputs: dict[str, torch.Tensor],
    rbt_target: torch.Tensor,
    edge_index: torch.Tensor,
    weights: dict[str, float],
    full_outputs: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    full = full_outputs or outputs
    l_beh = behavior_reconstruction_loss(outputs["rbt_hat"], rbt_target)
    l_con = contrastive_behavior_loss(outputs["z_sem"], rbt_target)
    l_topo = topology_preservation_loss(full["z_topo"], edge_index)
    l_orth = orthogonality_loss(full["z_sem"], full["z_topo"])
    total = (
        weights["behavior"] * l_beh
        + weights["contrastive"] * l_con
        + weights["topology"] * l_topo
        + weights["orthogonality"] * l_orth
    )
    return total, {
        "behavior": float(l_beh.detach()),
        "contrastive": float(l_con.detach()),
        "topology": float(l_topo.detach()),
        "orthogonality": float(l_orth.detach()),
        "total": float(total.detach()),
    }


def train_student(
    model: ReasoningSignatureModel,
    graph_data: dict[str, torch.Tensor],
    rbt_target: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    config: dict[str, Any],
) -> tuple[ReasoningSignatureModel, dict[str, Any]]:
    torch.manual_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = graph_data["x"].to(device)
    edge_index = graph_data["edge_index"].to(device)
    rbt_target = rbt_target.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    weights = config["training"]["loss_weights"]
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = config["training"]["early_stop_patience"]
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(config["training"]["epochs"]):
        model.train()
        outputs = model(x, edge_index)
        train_rbt = rbt_target[train_mask]
        train_outputs = {
            "rbt_hat": outputs["rbt_hat"][train_mask],
            "z_sem": outputs["z_sem"][train_mask],
            "z_topo": outputs["z_topo"][train_mask],
        }
        loss, loss_dict = compute_losses(train_outputs, train_rbt, edge_index, weights, full_outputs=outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(x, edge_index)
            val_loss, val_dict = compute_losses(
                {
                    "rbt_hat": val_outputs["rbt_hat"][val_mask],
                    "z_sem": val_outputs["z_sem"][val_mask],
                    "z_topo": val_outputs["z_topo"][val_mask],
                },
                rbt_target[val_mask],
                edge_index,
                weights,
                full_outputs=val_outputs,
            )
        history.append({"epoch": epoch, **loss_dict, "val_behavior": val_dict["behavior"]})
        if val_dict["behavior"] < best_val:
            best_val = val_dict["behavior"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best_val_behavior": best_val, "epochs_ran": len(history), "history": history}


@torch.no_grad()
def encode_graph(
    model: ReasoningSignatureModel,
    graph_data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    model.eval()
    x = graph_data["x"].to(device)
    edge_index = graph_data["edge_index"].to(device)
    return model(x, edge_index)


def make_train_val_masks(num_nodes: int, train_split: float, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)
    split = int(num_nodes * train_split)
    train_idx = perm[:split]
    val_idx = perm[split:]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask


def save_checkpoint(model: ReasoningSignatureModel, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)
