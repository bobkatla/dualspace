"""Informativeness proxies for a kept set in phi-space.


- log-det of covariance (regularized)
- entropy proxy via Gaussian fit
"""
from __future__ import annotations
import torch


# TODO: logdet_cov(Y_kept: (N, D)) -> float