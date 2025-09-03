"""Maximum Mean Discrepancy (MMD) in φ-space with RBF kernel mixture."""
from __future__ import annotations
import numpy as np


def _pdists2(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # squared Euclidean distances between rows
    XX = (X**2).sum(1, keepdims=True)
    YY = (Y**2).sum(1, keepdims=True)
    return XX + YY.T - 2 * X @ Y.T


def _rbf_kernel(D2: np.ndarray, sigma2: float) -> np.ndarray:
    return np.exp(-D2 / (2.0 * sigma2))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigmas: list[float] | None = None, unbiased: bool = True) -> float:
    """
    MMD^2 with a mixture of RBF kernels; returns scalar.
    X, Y: (n_x,d), (n_y,d) in φ-space. sigmas are kernel bandwidths.
    """
    if sigmas is None:
        # median heuristic on pooled data
        Z = np.concatenate([X, Y], axis=0)
        D2 = _pdists2(Z, Z)
        med = np.median(D2[D2 > 0])
        sigmas = [np.sqrt(med)] if med > 0 else [1.0, 2.0, 4.0]

    Kxx = 0.0; Kyy = 0.0; Kxy = 0.0
    Dxx = _pdists2(X, X); Dyy = _pdists2(Y, Y); Dxy = _pdists2(X, Y)
    for s in sigmas:
        s2 = s*s
        Kxx += _rbf_kernel(Dxx, s2)
        Kyy += _rbf_kernel(Dyy, s2)
        Kxy += _rbf_kernel(Dxy, s2)

    nx = X.shape[0]; ny = Y.shape[0]
    if unbiased:
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        mmd2 = Kxx.sum()/(nx*(nx-1)) + Kyy.sum()/(ny*(ny-1)) - 2.0*Kxy.mean()
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0*Kxy.mean()
    return float(mmd2)
