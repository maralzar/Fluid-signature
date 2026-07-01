import torch

from src.data.rdf_loader import generate_lubm_instance
from src.fluid.summarizer import summarize_graph


def test_summary_numeric_features_are_scale_normalized():
    graph = generate_lubm_instance(1, scale=2)
    summary = summarize_graph(graph)
    numeric = summary.summary_graph["x"][:, :3]

    assert torch.all(numeric >= 0.0)
    assert torch.all(numeric <= 1.0)
