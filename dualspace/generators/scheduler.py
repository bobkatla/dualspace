"""Diffusion noise schedule utilities.


- cosine schedule for betas
- helpers to map t in [0,T) to alphas, etc.
"""
from __future__ import annotations
import torch


def cosine_beta_schedule(T: int) -> torch.Tensor:
    # TODO: implement
    raise NotImplementedError