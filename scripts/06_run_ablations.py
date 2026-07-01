from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from src.models.student import ReasoningSignatureModel
from src.training.eval import evaluate_transfer
from src.training.trainer import make_train_val_masks, train_student
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json, save_json


def build_model(config: dict, in_dim: int) -> ReasoningSignatureModel:
    return ReasoningSignatureModel(
        in_dim=in_dim,
        hidden_dim=config["model"]["hidden_dim"],
        sem_dim=config["model"]["sem_dim"],
        topo_dim=config["model"]["topo_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        rbt_shape=(
            config["rbt"]["num_rule_families"],
            config["rbt"]["max_depth"],
            config["rbt"]["num_channels"],
        ),
        topo_bias_eps=config["model"].get("topo_bias_eps", 0.05),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run loss ablation studies.")
    parser.add_argument("--config", default="configs/phase2.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["_root"])
    processed_dir = root / config["project"]["data_dir"] / "processed"
    results_dir = ensure_dir(root / config["project"]["results_dir"])
    train_graph = config["graphs"]["train"]
    transfer_graph = config["graphs"]["transfer"]

    graph_data = torch.load(processed_dir / f"{train_graph}.summary_graph.pt", weights_only=False)
    rbt_target = torch.load(processed_dir / f"{train_graph}.rbt.pt", weights_only=False)
    source_graph = graph_data
    target_graph = torch.load(processed_dir / f"{transfer_graph}.summary_graph.pt", weights_only=False)
    rbt_source = rbt_target
    rbt_transfer = torch.load(processed_dir / f"{transfer_graph}.rbt.pt", weights_only=False)
    stats = load_json(processed_dir / f"{train_graph}.stats.json")
    train_mask, val_mask = make_train_val_masks(
        stats["num_supernodes"],
        config["training"]["train_split"],
        seed=config["project"]["seed"],
    )

    ablation_results = []
    for variant in config.get("ablations", {}).get("variants", []):
        variant_config = copy.deepcopy(config)
        variant_config["training"]["loss_weights"] = variant["loss_weights"]
        torch.manual_seed(variant_config["project"]["seed"])
        model = build_model(variant_config, graph_data["x"].shape[1])
        model, train_info = train_student(
            model, graph_data, rbt_target, train_mask, val_mask, variant_config
        )
        report = evaluate_transfer(
            model, source_graph, target_graph, rbt_source, rbt_transfer, variant_config
        )
        ablation_results.append({
            "variant": variant["name"],
            "loss_weights": variant["loss_weights"],
            "best_val_behavior": train_info["best_val_behavior"],
            "behavior_alignment": report["behavior_alignment"],
            "topo_matched_alignment": report["topo_matched_alignment"],
            "random_alignment": report["random_alignment"],
            "mvp_passed": report["mvp_passed"],
            "significance_vs_random": report["significance_vs_random"],
            "significance_sem_vs_topo": report["significance_sem_vs_topo"],
        })
        print(f"{variant['name']}: alignment={report['behavior_alignment']:.3f}, mvp_passed={report['mvp_passed']}")

    out_path = results_dir / "ablations.json"
    save_json(out_path, {"variants": ablation_results})
    print(f"Ablations saved to {out_path}")


if __name__ == "__main__":
    main()
