"""YAML config loader using ruamel.yaml or PyYAML."""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg