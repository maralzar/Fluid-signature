from __future__ import annotations

import torch
import torch.nn.functional as F


def behavior_reconstruction_loss(rbt_hat: torch.Tensor, rbt_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(rbt_hat, rbt_target)


def contrastive_behavior_loss(
    z_sem: torch.Tensor,
    rbt_target: torch.Tensor,
    temperature: float = 0.2,
    top_k: int = 3,
) -> torch.Tensor:
    flat = rbt_target.reshape(rbt_target.shape[0], -1)
    flat = F.normalize(flat, dim=1)
    behavior_sim = flat @ flat.t()
    num_nodes = behavior_sim.shape[0]
    if num_nodes <= 1:
        return torch.zeros((), device=z_sem.device)

    z_norm = F.normalize(z_sem, dim=1)
    z_sim = z_norm @ z_norm.t()
    off_diag = ~torch.eye(num_nodes, device=z_sem.device, dtype=torch.bool)
    similarity_targets = behavior_sim.clamp(min=0.0, max=1.0) * 2.0 - 1.0
    pairwise_loss = F.mse_loss(z_sim[off_diag], similarity_targets[off_diag])

    sim = behavior_sim
    k = min(top_k, num_nodes - 1)
    _, neighbors = torch.topk(sim, k=k + 1, dim=1)
    neighbors = neighbors[:, 1:]
    logits = (z_norm @ z_norm.t()) / temperature
    self_mask = torch.eye(num_nodes, device=z_sem.device, dtype=torch.bool)
    contrastive_logits = logits.masked_fill(self_mask, float("-inf"))
    losses = []
    for idx in range(num_nodes):
        pos = neighbors[idx]
        pos_logits = logits[idx, pos]
        denom = torch.logsumexp(contrastive_logits[idx], dim=0)
        losses.append(-(pos_logits - denom).mean())
    return pairwise_loss + 0.1 * torch.stack(losses).mean()


def topology_preservation_loss(
    z_topo: torch.Tensor,
    edge_index: torch.Tensor,
    num_negatives: int = 5,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((), device=z_topo.device)

    src, dst = edge_index[0], edge_index[1]
    pos_score = (z_topo[src] * z_topo[dst]).sum(dim=1)
    num_nodes = z_topo.shape[0]
    neg_dst = torch.randint(0, num_nodes, (src.shape[0], num_negatives), device=z_topo.device)
    neg_score = (z_topo[src].unsqueeze(1) * z_topo[neg_dst]).sum(dim=2)
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss


def orthogonality_loss(z_sem: torch.Tensor, z_topo: torch.Tensor) -> torch.Tensor:
    sem = F.normalize(z_sem, dim=1)
    topo = F.normalize(z_topo, dim=1)
    cross = sem.t() @ topo
    return (cross ** 2).sum()


def compute_losses(
    outputs: dict[str, torch.Tensor],
    rbt_target: torch.Tensor,
    edge_index: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    l_beh = behavior_reconstruction_loss(outputs["rbt_hat"], rbt_target)
    l_con = contrastive_behavior_loss(outputs["z_sem"], rbt_target)
    l_topo = topology_preservation_loss(outputs["z_topo"], edge_index)
    l_orth = orthogonality_loss(outputs["z_sem"], outputs["z_topo"])
    total = (
        weights["behavior"] * l_beh
        + weights["contrastive"] * l_con
        + weights["topology"] * l_topo
        + weights["orthogonality"] * l_orth
    )
    components = {
        "behavior": float(l_beh.detach()),
        "contrastive": float(l_con.detach()),
        "topology": float(l_topo.detach()),
        "orthogonality": float(l_orth.detach()),
        "total": float(total.detach()),
    }
    return total, components
