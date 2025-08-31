"""Diffusion noise schedule utilities.


- cosine schedule for betas
- helpers to map t in [0,T) to alphas, etc.
"""
from __future__ import annotations
import torch

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021). Returns betas in (0,1)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    f = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)