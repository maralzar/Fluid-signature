from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.student import ReasoningSignatureModel
from src.training.trainer import make_train_val_masks, save_checkpoint, train_student
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Reasoning Signature student on LUBM1.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["_root"])
    processed_dir = root / config["project"]["data_dir"] / "processed"
    graph_name = config["graphs"]["train"]
    checkpoints_dir = ensure_dir(root / config["project"]["checkpoints_dir"])
    torch.manual_seed(config["project"]["seed"])

    graph_data = torch.load(processed_dir / f"{graph_name}.summary_graph.pt", weights_only=False)
    rbt_target = torch.load(processed_dir / f"{graph_name}.rbt.pt", weights_only=False)
    stats = load_json(processed_dir / f"{graph_name}.stats.json")

    model = ReasoningSignatureModel(
        in_dim=graph_data["x"].shape[1],
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
    train_mask, val_mask = make_train_val_masks(
        stats["num_supernodes"],
        config["training"]["train_split"],
        seed=config["project"]["seed"],
    )
    model, train_info = train_student(model, graph_data, rbt_target, train_mask, val_mask, config)
    checkpoint_path = checkpoints_dir / f"{graph_name}_student.pt"
    save_checkpoint(
        model,
        checkpoint_path,
        {
            "graph_name": graph_name,
            "train_info": train_info,
            "config": config,
            "in_dim": graph_data["x"].shape[1],
        },
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Best val behavior loss: {train_info['best_val_behavior']:.6f}")


if __name__ == "__main__":
    main()
