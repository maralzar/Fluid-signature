from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = Path(path) if path else root / "configs" / "default.yaml"
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_root"] = str(root)
    return config
