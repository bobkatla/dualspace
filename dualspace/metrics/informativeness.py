"""Informativeness metrics in φ-space + simple diverse representative selection."""
from __future__ import annotations
import numpy as np
from numpy.linalg import slogdet
from typing import Tuple


def logdet_cov_phi(Y: np.ndarray, eps: float = 1e-4) -> float:
    """
    Proxy for region 'size' / diffuseness in φ-space.
    Y: (n, d_phi) survivors. Returns logdet(cov + eps I). If n<2 -> -inf.
    """
    if Y.shape[0] < 2:
        return float("-inf")
    C = np.cov(Y.T) + eps * np.eye(Y.shape[1], dtype=Y.dtype)
    sign, val = slogdet(C)
    return float(val) if sign > 0 else float("-inf")


def farthest_point_sampling(Y: np.ndarray, m: int, seed: int = 1337) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick m diverse representatives by greedy farthest-point sampling (FPS).
    Returns (indices, Y_subset).
    """
    n = Y.shape[0]
    m = min(m, n)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n))
    chosen = [start]
    d2 = np.sum((Y - Y[start])**2, axis=1)  # (n,)
    for _ in range(1, m):
        i = int(np.argmax(d2))
        chosen.append(i)
        d2 = np.minimum(d2, np.sum((Y - Y[i])**2, axis=1))
    idx = np.array(chosen, dtype=int)
    return idx, Y[idx]
