from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_c: int = 64,
        hidden: int = 256,
        depth: int = 2,
        dropout: float = 0.0,
        norm: str | None = None,
        mode: str = "mlp",          # "mlp" | "linear_orth"
        orth_reg: float = 0.0,      # strength for optional orthogonality penalty
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.orth_reg = float(orth_reg)

        if mode == "linear_orth":
            self.proj = nn.Linear(num_classes, d_c, bias=bias)
            nn.init.orthogonal_(self.proj.weight)        # near-orthogonal columns
            if bias:
                nn.init.zeros_(self.proj.bias)
            self.net = None
        else:
            layers: list[nn.Module] = []
            in_dim = num_classes
            for _ in range(depth):
                layers += [nn.Linear(in_dim, hidden, bias=bias), nn.SiLU()]
                if dropout > 0:
                    layers += [nn.Dropout(dropout)]
                in_dim = hidden
            layers += [nn.Linear(in_dim, d_c, bias=bias)]
            self.net = nn.Sequential(*layers)
            self.proj = None

        self.norm_type = (norm or "").lower() or None
        self.out_norm = nn.LayerNorm(d_c) if self.norm_type == "layernorm" else nn.Identity()

        # For classifier-free guidance (when dropping conditioning)
        self.null_embedding = nn.Parameter(torch.zeros(d_c))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear_orth":
            e = self.proj(c)
        else:
            e = self.net(c)
        e = self.out_norm(e)
        if self.norm_type == "l2":
            e = F.normalize(e, dim=-1)
        return e

    def get_null(self, batch: int) -> torch.Tensor:
        return self.null_embedding.unsqueeze(0).expand(batch, -1)

    def orthogonality_loss(self) -> torch.Tensor:
        """
        Optional regularizer encouraging class directions to be orthogonal.
        Only meaningful for mode='linear_orth'. Returns a scalar loss.
        """
        if self.mode != "linear_orth" or self.orth_reg <= 0:
            return torch.tensor(0.0, device=self.null_embedding.device)
        # W: (d_c, C); normalize columns to compute Gram matrix
        W = self.proj.weight  # (d_c, C)
        Wn = F.normalize(W, dim=0)
        G = Wn.t() @ Wn                      # (C, C)
        off_diag = G - torch.eye(G.size(0), device=G.device)
        return self.orth_reg * (off_diag ** 2).mean()
