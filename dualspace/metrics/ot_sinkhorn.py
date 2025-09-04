import numpy as np
import ot

def sinkhorn_cost(Yr: np.ndarray, Yf: np.ndarray, reg: float = 0.1) -> float:
    Yr = np.asarray(Yr, dtype=np.float64)
    Yf = np.asarray(Yf, dtype=np.float64)

    n, m = Yr.shape[0], Yf.shape[0]
    a = np.full(n, 1.0/n, dtype=np.float64)
    b = np.full(m, 1.0/m, dtype=np.float64)

    # squared Euclidean + tiny ridge avoids zeros in kernel
    C = ot.utils.dist(Yr, Yf, metric="euclidean")**2
    C += 1e-12

    # log-domain solver = no exp overflow
    G, log = ot.bregman.sinkhorn_log(
        a, b, C, reg=reg, log=True, stopThr=1e-9, numItermax=20000
    )
    return float((G * C).sum())
