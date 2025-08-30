"""Feature map phi(x) for images using a frozen backbone + PCA.


API
---
class ImagePhi(nn.Module):
forward(x: Float[Tensor, B,3,H,W]) -> Float[Tensor, B, D_phi]

TODO
----
- Wire torchvision resnet18 avgpool features.
- Add PCA fit/transform utilities (sklearn) and persist components.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ImagePhi(nn.Module):
    def __init__(self, d_phi: int = 64, backbone: str = "resnet18"):
        super().__init__()
        # TODO: instantiate and freeze backbone; expose a .transform() that applies PCA
        self.d_phi = d_phi

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B,3,H,W) -> (B,D_phi)
        # TODO: extract features, L2-normalize, PCA-project
        raise NotImplementedError