"""Conditional diffusion for 1D time-series with classifier-free guidance.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .unet1d import UNet1DSmall


class TimeseriesDiffusion(nn.Module):
    def __init__(self, in_ch: int, d_c: int = 64, T: int = 200):
        super().__init__()
        self.unet = UNet1DSmall(in_ch=in_ch, d_c=d_c)
        self.T = T
        # TODO


    def loss(self, x0: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


    @torch.no_grad()
    def sample(self, e: torch.Tensor, K: int = 256, guidance_scale: float = 2.0, shape=(9,128)) -> torch.Tensor:
        raise NotImplementedError