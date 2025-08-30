"""IO helpers: config loading, simple JSON/NPZ save/load.


TODO: add ml-collections or OmegaConf if needed.
"""
from __future__ import annotations
import json, os
from typing import Any, Dict
import numpy as np

def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def save_npz(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)