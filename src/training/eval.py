from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from src.models.student import ReasoningSignatureModel
from src.symbolic.rbt_builder import flatten_rbt
from src.training.trainer import encode_graph


def rbt_cosine_similarity(source: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    src = flatten_rbt(source).cpu().numpy()
    tgt = flatten_rbt(target).cpu().numpy()
    return cosine_similarity(tgt, src)


def structural_feature_similarity(source_graph: dict[str, torch.Tensor], target_graph: dict[str, torch.Tensor]) -> float:
    src = source_graph["x"].cpu().numpy()
    tgt = target_graph["x"].cpu().numpy()
    min_nodes = min(src.shape[0], tgt.shape[0])
    if min_nodes == 0:
        return 0.0
    src = src[:min_nodes]
    tgt = tgt[:min_nodes]
    return float(np.mean([
        cosine_similarity(src[i:i + 1], tgt[i:i + 1])[0, 0] for i in range(min_nodes)
    ]))


def behavior_alignment_score(
    z_source: torch.Tensor,
    z_target: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> float:
    if not pairs:
        return 0.0
    src = F.normalize(z_source, dim=1).cpu().numpy()
    tgt = F.normalize(z_target, dim=1).cpu().numpy()
    scores = [float(np.dot(tgt[t_idx], src[s_idx])) for t_idx, s_idx in pairs]
    return float(np.mean(scores))


def match_supernodes_by_rbt(rbt_source: torch.Tensor, rbt_target: torch.Tensor) -> list[tuple[int, int]]:
    sim = rbt_cosine_similarity(rbt_source, rbt_target)
    pairs: list[tuple[int, int]] = []
    for target_idx in range(sim.shape[0]):
        source_idx = int(np.argmax(sim[target_idx]))
        pairs.append((target_idx, source_idx))
    return pairs


def bootstrap_significance(
    scores: list[float],
    baseline: list[float],
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(samples):
        sample_scores = rng.choice(scores, size=len(scores), replace=True)
        sample_base = rng.choice(baseline, size=len(baseline), replace=True)
        diffs.append(float(np.mean(sample_scores) - np.mean(sample_base)))
    diffs = np.array(diffs)
    p_value = float(np.mean(diffs <= 0.0))
    return {
        "mean_diff": float(np.mean(scores) - np.mean(baseline)),
        "p_value": p_value,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def random_pair_scores(
    z_source: torch.Tensor,
    z_target: torch.Tensor,
    pairs: list[tuple[int, int]],
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    num_source = z_source.shape[0]
    scores = []
    for target_idx, _ in pairs:
        rand_idx = int(rng.integers(0, num_source))
        z_t = F.normalize(z_target[target_idx], dim=0)
        z_s = F.normalize(z_source[rand_idx], dim=0)
        scores.append(float(torch.dot(z_t, z_s)))
    return scores


@torch.no_grad()
def evaluate_transfer(
    model: ReasoningSignatureModel,
    source_graph: dict[str, torch.Tensor],
    target_graph: dict[str, torch.Tensor],
    rbt_source: torch.Tensor,
    rbt_target: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    source_out = encode_graph(model, source_graph)
    target_out = encode_graph(model, target_graph)
    pairs = match_supernodes_by_rbt(rbt_source, rbt_target)
    source_behavior_mse = float(F.mse_loss(source_out["rbt_hat"].cpu(), rbt_source.cpu()))
    target_behavior_mse = float(F.mse_loss(target_out["rbt_hat"].cpu(), rbt_target.cpu()))

    pair_scores = [
        float(torch.dot(
            F.normalize(target_out["z_sem"][target_idx], dim=0),
            F.normalize(source_out["z_sem"][source_idx], dim=0),
        ))
        for target_idx, source_idx in pairs
    ]
    topo_scores = [
        float(torch.dot(
            F.normalize(target_out["z_topo"][target_idx], dim=0),
            F.normalize(source_out["z_topo"][source_idx], dim=0),
        ))
        for target_idx, source_idx in pairs
    ]
    random_scores = random_pair_scores(
        source_out["z_sem"],
        target_out["z_sem"],
        pairs,
        seed=config["project"]["seed"],
    )

    significance = bootstrap_significance(
        pair_scores,
        random_scores,
        samples=config["transfer"]["bootstrap_samples"],
        seed=config["project"]["seed"],
    )
    topo_vs_random = bootstrap_significance(
        topo_scores,
        random_scores,
        samples=config["transfer"]["bootstrap_samples"],
        seed=config["project"]["seed"] + 1,
    )
    sem_vs_topo = bootstrap_significance(
        pair_scores,
        topo_scores,
        samples=config["transfer"]["bootstrap_samples"],
        seed=config["project"]["seed"] + 2,
    )

    mvp_threshold = config["transfer"]["mvp_threshold"]
    passed = significance["mean_diff"] >= mvp_threshold and significance["p_value"] < 0.05

    return {
        "behavior_alignment": float(np.mean(pair_scores)) if pair_scores else 0.0,
        "source_behavior_mse": source_behavior_mse,
        "target_behavior_mse": target_behavior_mse,
        "topo_matched_alignment": float(np.mean(topo_scores)) if topo_scores else 0.0,
        "random_alignment": float(np.mean(random_scores)) if random_scores else 0.0,
        "structural_feature_similarity": structural_feature_similarity(source_graph, target_graph),
        "num_pairs": len(pairs),
        "significance_vs_random": significance,
        "significance_topo_vs_random": topo_vs_random,
        "significance_sem_vs_topo": sem_vs_topo,
        "mvp_passed": passed,
        "pair_scores": pair_scores,
        "topo_scores": topo_scores,
    }


def run_baselines(
    model: ReasoningSignatureModel,
    source_graph: dict[str, torch.Tensor],
    target_graph: dict[str, torch.Tensor],
    rbt_source: torch.Tensor,
    rbt_target: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    main = evaluate_transfer(model, source_graph, target_graph, rbt_source, rbt_target, config)
    pairs = match_supernodes_by_rbt(rbt_source, rbt_target)

    triple_proxy = []
    for target_idx, source_idx in pairs:
        src_count = float(source_graph["x"][source_idx, 0])
        tgt_count = float(target_graph["x"][target_idx, 0])
        triple_proxy.append(1.0 - abs(src_count - tgt_count) / max(src_count, tgt_count, 1.0))

    main["baselines"] = {
        "topo_matched_alignment": main["topo_matched_alignment"],
        "triple_materialization_proxy": float(np.mean(triple_proxy)) if triple_proxy else 0.0,
        "random_alignment": main["random_alignment"],
    }
    main["significance"] = main["significance_vs_random"]
    return main


@torch.no_grad()
def build_reasoning_signature_artifact(
    model: ReasoningSignatureModel,
    graph_data: dict[str, torch.Tensor],
    graph_name: str,
) -> dict[str, Any]:
    outputs = encode_graph(model, graph_data)
    z_sem = outputs["z_sem"].detach().cpu()
    z_topo = outputs["z_topo"].detach().cpu()
    rbt_hat = outputs["rbt_hat"].detach().cpu()
    return {
        "graph_name": graph_name,
        "signature": torch.cat([z_sem, z_topo], dim=1),
        "z_sem": z_sem,
        "z_topo": z_topo,
        "rbt_hat": rbt_hat,
        "num_supernodes": int(z_sem.shape[0]),
        "sem_dim": int(z_sem.shape[1]),
        "topo_dim": int(z_topo.shape[1]),
    }
