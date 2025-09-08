"""Conformal calibration of tau_alpha.


Scores
------
- s(x, c) := -log \hat p_psi(phi(x) | e=g(c)) (lower is better)


Procedure
---------
- Compute scores on calibration split (class-conditional by default).
- For each alpha in config, set tau_alpha as the (1-alpha)*(n+1)/n quantile of -scores,
then store log-thresholds for region predicate log p >= tau_alpha.


CLI
---
python -m dualspace.regions.conformal --config configs/cifar10.yaml --split calib
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import numpy as np


def _conformal_index(n: int, alpha: float) -> int:
    """Finite-sample index for coverage >= alpha on one-sided sets."""
    # threshold = ceil(alpha * (n + 1)) - 1  (0-based)
    k = int(np.ceil(alpha * (n + 1))) - 1
    return max(0, min(k, n - 1))


def _pooled_thresholds(scores_all: np.ndarray, alphas: List[float]) -> Dict[str, float]:
    s_sorted = np.sort(scores_all)
    n = len(s_sorted)
    out = {}
    for a in alphas:
        idx = _conformal_index(n, a)
        s_thr = float(s_sorted[idx])
        out[f"{a:.3f}"] = -s_thr
    return out


def _per_class_thresholds(scores: Dict[int, np.ndarray], alphas: List[float]) -> Dict[int, Dict[str, float]]:
    """
    scores[k] = array of nonconformity scores s = -log p(y|e) for class k (calib split).
    Returns per-class log-prob thresholds tau_alpha = -s_alpha (since region is logp >= tau_alpha).
    """
    out: Dict[int, Dict[str, float]] = {}
    for k, s in scores.items():
        s_sorted = np.sort(np.asarray(s))
        n = len(s_sorted)
        out_k: Dict[str, float] = {}
        for a in alphas:
            idx = _conformal_index(n, a)
            s_thr = float(s_sorted[idx])
            out_k[f"{a:.3f}"] = -s_thr
        out[int(k)] = out_k
    return out


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class TauAlpha:
    """A handler for conformal score thresholds."""
    def __init__(self, mode: str, alphas: List[float]):
        if mode not in ["global", "class", "knn"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        self.alphas = alphas
        self.thresholds = {}

    def fit(self, scores: np.ndarray, conditions: np.ndarray | None = None):
        """Fit thresholds from calibration scores."""
        if self.mode == "global":
            self.thresholds = _pooled_thresholds(scores, self.alphas)
        elif self.mode == "class":
            if conditions is None:
                raise ValueError("`conditions` (class labels) must be provided for 'class' mode.")
            scores_by_class = {
                int(k): scores[conditions == k] for k in np.unique(conditions)
            }
            self.thresholds = _per_class_thresholds(scores_by_class, self.alphas)
        elif self.mode == "knn":
            # KNN mode computes thresholds on-the-fly, no fitting needed here.
            # It requires storing calibration scores and embeddings.
            raise NotImplementedError("KNN mode is not yet implemented.")
        return self

    def t(self, alpha: float, condition: int | None = None) -> float:
        """Get the log-probability threshold for a given alpha and condition."""
        alpha_str = f"{alpha:.3f}"
        if self.mode == "global":
            return self.thresholds[alpha_str]
        elif self.mode == "class":
            if condition is None:
                raise ValueError("`condition` (class label) must be provided for 'class' mode.")
            return self.thresholds[str(int(condition))][alpha_str]
        else: # knn
             raise NotImplementedError("KNN mode is not yet implemented.")

    def save(self, path: Path):
        """Save thresholds and config to a JSON file."""
        data = {
            "mode": self.mode,
            "alphas": self.alphas,
            "thresholds": self.thresholds,
        }
        save_json(data, path)

    @classmethod
    def load(cls, path: Path):
        """Load thresholds from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        instance = cls(mode=data["mode"], alphas=data["alphas"])
        # JSON saves integer keys as strings, so we may need to convert them back
        if instance.mode == "class":
            instance.thresholds = {int(k): v for k, v in data["thresholds"].items()}
        else:
            instance.thresholds = data["thresholds"]
        
        return instance
