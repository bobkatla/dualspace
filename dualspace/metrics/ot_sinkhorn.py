"""Sinkhorn OT distance in φ-space using POT (pip install pot)."""
from __future__ import annotations
import numpy as np
import ot  # POT


def sinkhorn_phi(X: np.ndarray, Y: np.ndarray, reg: float = 0.05) -> float:
    """
    X, Y: (n,d), (m,d) features in φ-space.
    reg: entropic regularization ε.
    Returns Sinkhorn cost with squared Euclidean ground metric.
    """
    n, m = X.shape[0], Y.shape[0]
    a = np.full(n, 1.0/n, dtype=float)
    b = np.full(m, 1.0/m, dtype=float)
    C = ot.dist(X, Y, metric='euclidean') ** 2  # (n,m)
    P = ot.sinkhorn(a, b, C, reg)
    cost = float((P * C).sum())
    return cost
