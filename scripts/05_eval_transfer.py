from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.student import ReasoningSignatureModel
from src.training.eval import build_reasoning_signature_artifact, run_baselines
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
    parser = argparse.ArgumentParser(description="Evaluate cross-graph transfer on LUBM2.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["_root"])
    processed_dir = root / config["project"]["data_dir"] / "processed"
    results_dir = ensure_dir(root / config["project"]["results_dir"])
    train_graph = config["graphs"]["train"]
    transfer_graph = config["graphs"]["transfer"]

    checkpoint = torch.load(
        root / config["project"]["checkpoints_dir"] / f"{train_graph}_student.pt",
        weights_only=False,
    )
    source_graph = torch.load(processed_dir / f"{train_graph}.summary_graph.pt", weights_only=False)
    target_graph = torch.load(processed_dir / f"{transfer_graph}.summary_graph.pt", weights_only=False)
    rbt_source = torch.load(processed_dir / f"{train_graph}.rbt.pt", weights_only=False)
    rbt_target = torch.load(processed_dir / f"{transfer_graph}.rbt.pt", weights_only=False)

    model = build_model(config, checkpoint["metadata"]["in_dim"])
    model.load_state_dict(checkpoint["model_state"])

    report = run_baselines(model, source_graph, target_graph, rbt_source, rbt_target, config)
    source_signatures = build_reasoning_signature_artifact(model, source_graph, train_graph)
    target_signatures = build_reasoning_signature_artifact(model, target_graph, transfer_graph)
    source_signature_path = processed_dir / f"{train_graph}.reasoning_signatures.pt"
    target_signature_path = processed_dir / f"{transfer_graph}.reasoning_signatures.pt"
    torch.save(source_signatures, source_signature_path)
    torch.save(target_signatures, target_signature_path)
    report["train_graph"] = train_graph
    report["transfer_graph"] = transfer_graph
    report["train_val_behavior"] = checkpoint["metadata"]["train_info"]["best_val_behavior"]
    report["signature_artifacts"] = {
        train_graph: str(source_signature_path),
        transfer_graph: str(target_signature_path),
    }
    report_path = results_dir / "mvp_transfer.json"
    save_json(report_path, {k: v for k, v in report.items() if k not in {"pair_scores", "topo_scores"}})
    print(json.dumps({k: v for k, v in report.items() if k not in {"pair_scores", "topo_scores"}}, indent=2))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
