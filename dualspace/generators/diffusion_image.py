"""Conditional diffusion wrapper for images with classifier-free guidance.


Train loop will optimize MSE between predicted and true noise.


forward_api:
sample(c, K, guidance_scale) -> x: (K, C, H, W)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .unet2d import UNet2DSmall


class ImageDiffusion(nn.Module):
    def __init__(self, in_ch: int = 3, d_c: int = 64, T: int = 200):
        super().__init__()
        self.unet = UNet2DSmall(in_ch=in_ch, d_c=d_c)
        self.T = T
        # TODO: register schedule buffers


    def loss(self, x0: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Compute noise-prediction loss for a batch.
        x0: (B,3,H,W), e: (B,d_c)
        """
        raise NotImplementedError


    @torch.no_grad()
    def sample(self, e: torch.Tensor, K: int = 256, guidance_scale: float = 2.0, shape=(3,32,32)) -> torch.Tensor:
        """Draw K samples conditioned on e.
        Returns (K,3,H,W).
        """
        raise NotImplementedError