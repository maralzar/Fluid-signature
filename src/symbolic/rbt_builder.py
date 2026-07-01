from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

from src.data.rdf_loader import RULE_FAMILIES, family_to_index
from src.utils.normalization import minmax_scale, normalize_frequency

RBT_CHANNELS = [
    "activation_frequency",
    "propagation_strength",
    "branching_factor",
    "rule_interaction",
    "semantic_constraint",
    "rule_centrality",
]


def rbt_channel_index(name: str) -> int:
    return RBT_CHANNELS.index(name)


def build_rbt_from_events(
    events: list[dict[str, Any]],
    num_supernodes: int,
    num_rule_families: int = 8,
    max_depth: int = 5,
    num_channels: int = 6,
) -> dict[int, torch.Tensor]:
    tensors = {
        sid: np.zeros((num_rule_families, max_depth, num_channels), dtype=np.float32)
        for sid in range(num_supernodes)
    }
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[int(event["supernode_id"])].append(event)

    for sid, rows in grouped.items():
        if sid not in tensors:
            continue
        freq = np.zeros((num_rule_families, max_depth), dtype=np.float32)
        raw_freq = np.zeros((num_rule_families, max_depth), dtype=np.float32)
        strength_sum = np.zeros((num_rule_families, max_depth), dtype=np.float32)
        strength_count = np.zeros((num_rule_families, max_depth), dtype=np.float32)
        branch_sum = np.zeros((num_rule_families, max_depth), dtype=np.float32)
        branch_count = np.zeros((num_rule_families, max_depth), dtype=np.float32)
        interaction = np.zeros((num_rule_families, max_depth), dtype=np.float32)
        semantic_sum = np.zeros((num_rule_families, max_depth), dtype=np.float32)
        semantic_count = np.zeros((num_rule_families, max_depth), dtype=np.float32)
        centrality = np.zeros((num_rule_families, max_depth), dtype=np.float32)

        depth_family_events: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            family_idx = family_to_index(row["rule_family"])
            depth_idx = min(int(row["depth"]), max_depth) - 1
            if depth_idx < 0:
                depth_idx = 0
            freq[family_idx, depth_idx] += 1.0
            raw_freq[family_idx, depth_idx] += 1.0
            strength_sum[family_idx, depth_idx] += float(row["propagation_strength"])
            strength_count[family_idx, depth_idx] += 1.0
            branch_sum[family_idx, depth_idx] += float(row["branch_width"])
            branch_count[family_idx, depth_idx] += 1.0
            semantic_sum[family_idx, depth_idx] += float(row.get("semantic_constraint", 0.0))
            semantic_count[family_idx, depth_idx] += 1.0
            depth_family_events[depth_idx].append(family_idx)

        for depth_idx, families in depth_family_events.items():
            counts = Counter(families)
            total = sum(counts.values())
            unique_ratio = (len(counts) - 1) / max(num_rule_families - 1, 1)
            if total <= 1 or unique_ratio <= 0:
                continue
            for family_idx, count in counts.items():
                interaction[family_idx, depth_idx] = (count / total) * unique_ratio

        total_events = raw_freq.sum()
        if total_events > 0:
            family_centrality = raw_freq.sum(axis=1) / total_events
            centrality[:, :] = family_centrality[:, None]

        for depth_idx in range(max_depth):
            counts = freq[:, depth_idx]
            total = counts.sum()
            if total <= 0:
                continue
            probs = counts / total

            freq[:, depth_idx] = normalize_frequency(counts)
            for family_idx in range(num_rule_families):
                if strength_count[family_idx, depth_idx] > 0:
                    strength_sum[family_idx, depth_idx] /= strength_count[family_idx, depth_idx]
                if branch_count[family_idx, depth_idx] > 0:
                    branch_sum[family_idx, depth_idx] /= branch_count[family_idx, depth_idx]
                if semantic_count[family_idx, depth_idx] > 0:
                    semantic_sum[family_idx, depth_idx] /= semantic_count[family_idx, depth_idx]
            for family_idx in range(num_rule_families):
                if num_channels > 0:
                    tensors[sid][family_idx, depth_idx, 0] = freq[family_idx, depth_idx]
                if num_channels > 1:
                    tensors[sid][family_idx, depth_idx, 1] = strength_sum[family_idx, depth_idx]
                if num_channels > 2:
                    tensors[sid][family_idx, depth_idx, 2] = branch_sum[family_idx, depth_idx]
                if num_channels > 3:
                    tensors[sid][family_idx, depth_idx, 3] = interaction[family_idx, depth_idx]
                if num_channels > 4:
                    tensors[sid][family_idx, depth_idx, 4] = semantic_sum[family_idx, depth_idx]
                if num_channels > 5:
                    tensors[sid][family_idx, depth_idx, 5] = centrality[family_idx, depth_idx]

    if num_channels > 1 and tensors:
        strength_all = np.stack([t[:, :, 1] for t in tensors.values()], axis=0)
        scaled_strength = minmax_scale(strength_all)
        for idx, sid in enumerate(tensors):
            tensors[sid][:, :, 1] = scaled_strength[idx]

    if num_channels > 2 and tensors:
        branch_all = np.stack([t[:, :, 2] for t in tensors.values()], axis=0)
        scaled_branch = minmax_scale(branch_all)
        for idx, sid in enumerate(tensors):
            tensors[sid][:, :, 2] = scaled_branch[idx]

    return {sid: torch.tensor(arr, dtype=torch.float32) for sid, arr in tensors.items()}


def rbt_dict_to_tensor(rbt_dict: dict[int, torch.Tensor], num_supernodes: int) -> torch.Tensor:
    shape = next(iter(rbt_dict.values())).shape if rbt_dict else (8, 5, 6)
    stacked = torch.zeros((num_supernodes, *shape), dtype=torch.float32)
    for sid, tensor in rbt_dict.items():
        stacked[int(sid)] = tensor
    return stacked


def flatten_rbt(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)
