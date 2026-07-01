from __future__ import annotations

import argparse
from pathlib import Path

from src.data.rdf_loader import prepare_lubm_dataset
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LUBM1/LUBM2 and ontology files.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--scale", type=int, default=3, help="Graph scale factor.")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["_root"])
    raw_dir = root / config["project"]["data_dir"] / "raw"
    ensure_dir(raw_dir)

    graph_scales = config.get("graphs", {}).get("scales")
    manifest = prepare_lubm_dataset(
        raw_dir,
        scale=args.scale,
        graph_scales=graph_scales,
        seed=config["project"]["seed"],
    )
    manifest_path = raw_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"Wrote dataset manifest to {manifest_path}")
    for name, info in manifest["graphs"].items():
        print(f"  {name}: {info['triple_count']} ABox triples, {info['merged_triple_count']} merged triples")


if __name__ == "__main__":
    main()
