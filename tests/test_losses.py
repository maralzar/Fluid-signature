import torch

from src.training.losses import contrastive_behavior_loss


def test_contrastive_behavior_loss_penalizes_collapsed_embeddings():
    z_sem = torch.ones(3, 4)
    rbt_target = torch.zeros(3, 2, 2, 1)
    rbt_target[0, 0, 0, 0] = 1.0
    rbt_target[1, 1, 0, 0] = 1.0
    rbt_target[2, 0, 1, 0] = 1.0

    loss = contrastive_behavior_loss(z_sem, rbt_target)

    assert loss > 0.1
