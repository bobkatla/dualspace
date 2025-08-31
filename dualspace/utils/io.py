"""IO helpers: config loading, simple JSON/NPZ save/load.


TODO: add ml-collections or OmegaConf if needed.
"""
from __future__ import annotations
import json
from typing import Any, Dict
from pathlib import Path
import numpy as np

def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)