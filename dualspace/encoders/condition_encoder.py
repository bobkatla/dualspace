"""Condition encoder g(c) -> e with simple diagnostics.
Inputs
------
- c: one-hot condition tensor of shape (B, C).
Outputs
-------
- e: embedding tensor of shape (B, d_c).
Options
--------------------
- hidden (int): hidden width (default 256)
- depth (int): # of hidden layers (default 2)
- dropout (float): dropout prob (default 0.0)
- norm (str|None): "layernorm" or "l2" or None
- bias (bool): include biases (default True)
Notes
-----
- Keep tiny and fast; this will be trained jointly with the generator.
- `null_embedding` parameter is provided for classifier-free guidance later.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class ConditionEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_c: int = 64,
        hidden: int = 256,
        depth: int = 2,
        dropout: float = 0.0,
        norm: str | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_classes
        for i in range(depth):
            layers += [nn.Linear(in_dim, hidden, bias=bias), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, d_c, bias=bias)]
        self.net = nn.Sequential(*layers)

        self.norm_type = (norm or "").lower() or None
        if self.norm_type == "layernorm":
            self.out_norm = nn.LayerNorm(d_c)
        else:
            self.out_norm = nn.Identity()
        # For classifier-free guidance (when dropping conditioning)
        self.null_embedding = nn.Parameter(torch.zeros(d_c))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        e = self.net(c)
        e = self.out_norm(e)
        if self.norm_type == "l2":
            e = torch.nn.functional.normalize(e, dim=-1)
        return e

    def get_null(self, batch: int) -> torch.Tensor:
        return self.null_embedding.unsqueeze(0).expand(batch, -1)