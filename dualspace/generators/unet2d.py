"""Minimal U-Net 2D backbone for diffusion on CIFAR-10.

Forward signature
-----------------
forward(x_t, t, e) -> eps_pred with shapes matching x_t.


TODO
----
- Implement time embedding, FiLM-like conditioning on e.
- Small/medium variants selectable via config.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class UNet2DSmall(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64, d_c: int = 64):
        super().__init__()
        # TODO: implement blocks
        self.in_ch = in_ch
        self.base = base
        self.d_c = d_c


    def forward(self, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError