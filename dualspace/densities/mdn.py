"""Amortized conditional density \hat p_psi(y|e) as a Mixture Density Network.


Inputs
------
- y: (B, D_phi)
- e: (B, d_c)


Outputs
-------
- log_prob(y|e): (B,)


Notes
-----
- Diagonal covariances; numerically stable log-sum-exp.
- Also expose sample() for analysis.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDN(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_comp: int = 6, hidden: int = 256):
        super().__init__()
        self.d_out, self.n_comp = d_out, n_comp
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.fc_pi = nn.Linear(hidden, n_comp)
        self.fc_mu = nn.Linear(hidden, n_comp * d_out)
        self.fc_logvar = nn.Linear(hidden, n_comp * d_out)

    def forward(self, e: torch.Tensor):
        h = self.net(e)
        log_pi = self.fc_pi(h)                # (B,K)
        pi = F.log_softmax(log_pi, dim=-1)
        mu = self.fc_mu(h).view(-1, self.n_comp, self.d_out)
        logvar = self.fc_logvar(h).view(-1, self.n_comp, self.d_out)
        return pi, mu, logvar

    def log_prob(self, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Compute log p(y|e). Returns (B,)."""
        log_pi, mu, logvar = self.forward(e)
        y_exp = y.unsqueeze(1)  # (B,1,D)
        var = logvar.exp()
        log_comp = -0.5 * (((y_exp - mu) ** 2) / var + logvar + torch.log(torch.tensor(2*torch.pi, device=y.device)))
        log_comp = log_comp.sum(dim=-1)       # (B,K)
        log_mix = torch.logsumexp(log_pi + log_comp, dim=-1)  # (B,)
        return log_mix
