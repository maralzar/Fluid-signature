import torch

from src.models.student import ReasoningSignatureModel
from src.training.eval import build_reasoning_signature_artifact


def test_student_exports_reasoning_signature_artifact():
    model = ReasoningSignatureModel(
        in_dim=4,
        hidden_dim=8,
        sem_dim=5,
        topo_dim=3,
        num_layers=2,
        rbt_shape=(2, 3, 4),
    )
    graph_data = {
        "x": torch.randn(3, 4),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }

    artifact = build_reasoning_signature_artifact(model, graph_data, "toy")

    assert artifact["graph_name"] == "toy"
    assert artifact["z_sem"].shape == (3, 5)
    assert artifact["z_topo"].shape == (3, 3)
    assert artifact["signature"].shape == (3, 8)
    assert artifact["rbt_hat"].shape == (3, 2, 3, 4)
