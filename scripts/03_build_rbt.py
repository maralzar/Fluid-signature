from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.data.rdf_loader import load_graph
from src.symbolic.rbt_builder import build_rbt_from_events, rbt_dict_to_tensor
from src.symbolic.teacher import assign_supernodes, events_to_dataframe, run_instrumented_reasoning
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Reasoning Behavior Tensors.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["_root"])
    raw_dir = root / config["project"]["data_dir"] / "raw"
    processed_dir = ensure_dir(root / config["project"]["data_dir"] / "processed")
    graph_name = args.graph or config["graphs"]["train"]

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    graph = load_graph(manifest["graphs"][graph_name]["merged_path"])
    supernode_map = load_json(processed_dir / f"{graph_name}.supernode_map.json")
    stats = load_json(processed_dir / f"{graph_name}.stats.json")

    _, recorder = run_instrumented_reasoning(graph, max_depth=config["symbolic"]["max_reasoning_depth"])
    enriched = assign_supernodes(recorder.events, supernode_map)
    events_df = pd.DataFrame(enriched)
    events_path = processed_dir / f"{graph_name}.reasoning_events.parquet"
    events_df.to_parquet(events_path, index=False)

    rbt_dict = build_rbt_from_events(
        enriched,
        num_supernodes=stats["num_supernodes"],
        num_rule_families=config["rbt"]["num_rule_families"],
        max_depth=config["rbt"]["max_depth"],
        num_channels=config["rbt"]["num_channels"],
    )
    rbt_tensor = rbt_dict_to_tensor(rbt_dict, stats["num_supernodes"])
    torch.save(rbt_tensor, processed_dir / f"{graph_name}.rbt.pt")
    print(f"{graph_name}: {len(recorder.events)} events, {len(enriched)} assigned, RBT shape {tuple(rbt_tensor.shape)}")


if __name__ == "__main__":
    main()
