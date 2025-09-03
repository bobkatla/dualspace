import numpy as np
import ot

def sinkhorn_cost(Yr: np.ndarray, Yf: np.ndarray, reg=0.1):
    # Yr, Yf: (n,d) real/features in φ-space (float64 improves stability)
    Yr = np.asarray(Yr, dtype=np.float64)
    Yf = np.asarray(Yf, dtype=np.float64)

    n, m = len(Yr), len(Yf)
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m
    # avoid exact zeros
    eps = 1e-12
    a = (a + eps); a /= a.sum()
    b = (b + eps); b /= b.sum()

    # squared Euclidean cost + tiny ridge to keep K bounded
    C = ot.utils.dist(Yr, Yf, metric='euclidean') ** 2
    C += eps

    # log-domain Sinkhorn is much more stable
    # returns the regularized OT cost; use sinkhorn2 to get the cost directly
    G, log = ot.bregman.sinkhorn_log(a, b, C, reg=reg, log=True, stopThr=1e-9, numItermax=10000)
    # cost = <G, C>
    cost = float((G * C).sum())
    return cost
