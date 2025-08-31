from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import json
import numpy as np

@dataclass
class Splits:
    train: List[int]
    calib: List[int]
    test: List[int]

    def save(self, out_dir: str | Path, name: str) -> None:
        p = Path(out_dir) / "splits" / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"train": self.train, "calib": self.calib, "test": self.test}, f, indent=2)

def stratified_split(labels: Sequence[int], ratios=(0.7, 0.15, 0.15), seed: int = 1337) -> Splits:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    idx_train: List[int] = []
    idx_calib: List[int] = []
    idx_test: List[int] = []
    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(ratios[0] * n))
        n_calib = int(round(ratios[1] * n))
        n_test = n - n_train - n_calib
        idx_train.extend(idx[:n_train].tolist())
        idx_calib.extend(idx[n_train:n_train+n_calib].tolist())
        idx_test.extend(idx[n_train+n_calib:].tolist())
    return Splits(sorted(idx_train), sorted(idx_calib), sorted(idx_test))

def class_hist(labels: Sequence[int], indices: Sequence[int]) -> Dict[int, int]:
    labels = np.asarray(labels)
    idx = np.asarray(indices)
    vals, counts = np.unique(labels[idx], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}