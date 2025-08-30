"""Feature map phi(x) for multivariate time-series (T,D) -> R^{D_phi}.

Design
------
- Small frozen 1D CNN to produce a 128-d feature, then PCA->D_phi.
- Sequence shape convention: (B, D, T).
"""
from __future__ import annotations
import torch
import torch.nn as nn

class TimeseriesPhi(nn.Module):
    def __init__(self, d_phi: int = 32):
        super().__init__()
        # TODO: define 1D CNN feature extractor; freeze for phi extraction
        self.d_phi = d_phi

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B,D,T) -> (B,D_phi)
        raise NotImplementedError