"""Condition encoder g(c) -> e.


Inputs
------
- c: one-hot condition tensor of shape (B, C).


Outputs
-------
- e: embedding tensor of shape (B, d_c).


Notes
-----
- Keep tiny and fast (two-layer MLP by default).
- This is learned jointly with the generator when training diffusion.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class ConditionEncoder(nn.Module):
    def __init__(self, num_classes: int, d_c: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden), nn.SiLU(),
            nn.Linear(hidden, d_c)
        )


def forward(self, c: torch.Tensor) -> torch.Tensor:
    return self.net(c)