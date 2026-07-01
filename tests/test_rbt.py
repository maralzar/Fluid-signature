import numpy as np
import torch

from src.data.rdf_loader import family_to_index
from src.symbolic.rbt_builder import RBT_CHANNELS, build_rbt_from_events, rbt_channel_index


def test_empty_supernode_zero_tensor():
    events = []
    rbt = build_rbt_from_events(events, num_supernodes=3)
    for sid in range(3):
        assert torch.allclose(rbt[sid], torch.zeros_like(rbt[sid]))


def test_single_rule_peaked_frequency():
    events = [
        {
            "rule_family": "class_axioms",
            "depth": 1,
            "propagation_strength": 1.0,
            "branch_width": 1,
            "supernode_id": 0,
        }
    ]
    rbt = build_rbt_from_events(events, num_supernodes=1)
    tensor = rbt[0].numpy()
    family_idx = family_to_index("class_axioms")
    assert tensor[family_idx, 0, 0] > 0.0
    assert np.isclose(tensor[:, 0, 0].sum(), 1.0, atol=1e-5)


def test_normalization_frequency_channel():
    events = [
        {
            "rule_family": "class_axioms",
            "depth": 1,
            "propagation_strength": 1.0,
            "branch_width": 1,
            "supernode_id": 0,
        },
        {
            "rule_family": "property_assertion",
            "depth": 1,
            "propagation_strength": 1.0,
            "branch_width": 1,
            "supernode_id": 0,
        },
    ]
    rbt = build_rbt_from_events(events, num_supernodes=1)
    depth_slice = rbt[0][:, 0, 0].numpy()
    assert np.isclose(depth_slice.sum(), 1.0, atol=1e-5)


def test_named_semantic_constraint_channel():
    events = [
        {
            "rule_family": "property_assertion",
            "depth": 1,
            "propagation_strength": 2.0,
            "branch_width": 1,
            "semantic_constraint": 1.0,
            "supernode_id": 0,
        }
    ]
    rbt = build_rbt_from_events(events, num_supernodes=1)
    family_idx = family_to_index("property_assertion")
    channel_idx = rbt_channel_index("semantic_constraint")
    assert RBT_CHANNELS[channel_idx] == "semantic_constraint"
    assert rbt[0][family_idx, 0, channel_idx] == 1.0
