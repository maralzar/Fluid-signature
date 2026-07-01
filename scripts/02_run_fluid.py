from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.rdf_loader import load_graph
from src.fluid.summarizer import summarize_graph
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FLUID-compatible summarization.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default=None, help="Graph name, e.g. lubm1")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["_root"])
    raw_dir = root / config["project"]["data_dir"] / "raw"
    processed_dir = ensure_dir(root / config["project"]["data_dir"] / "processed")
    graph_name = args.graph or config["graphs"]["train"]

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    graph_path = manifest["graphs"][graph_name]["merged_path"]
    graph = load_graph(graph_path)

    summary = summarize_graph(graph, k_hop=config["fluid"]["k_hop"])
    out_prefix = processed_dir / graph_name
    save_json(out_prefix.with_suffix(".supernode_map.json"), summary.supernode_map)
    save_json(out_prefix.with_suffix(".supernode_members.json"), {str(k): v for k, v in summary.supernode_members.items()})
    save_json(out_prefix.with_suffix(".stats.json"), summary.stats)
    torch.save(summary.summary_graph, out_prefix.with_suffix(".summary_graph.pt"))
    print(f"{graph_name}: {summary.stats}")


if __name__ == "__main__":
    main()
