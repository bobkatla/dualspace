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


def per_class_thresholds(scores: Dict[int, np.ndarray], alphas: List[float]) -> Dict[int, Dict[str, float]]:
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


def pooled_thresholds(scores_all: np.ndarray, alphas: List[float]) -> Dict[str, float]:
    s_sorted = np.sort(scores_all)
    n = len(s_sorted)
    out = {}
    for a in alphas:
        idx = _conformal_index(n, a)
        s_thr = float(s_sorted[idx])
        out[f"{a:.3f}"] = -s_thr
    return out


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
