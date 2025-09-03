"""Informativeness metrics in φ-space + simple diverse representative selection."""
from __future__ import annotations
import numpy as np
from typing import Tuple


def logdet_cov_phi(y: np.ndarray, eps=1e-4) -> float:
    # y: (n, d)
    y = y - y.mean(axis=0, keepdims=True)
    C = (y.T @ y) / max(len(y)-1, 1)
    # ridge for numerical stability
    C.flat[::C.shape[0]+1] += eps
    s, _ = np.linalg.slogdet(C)
    return float(_)


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
