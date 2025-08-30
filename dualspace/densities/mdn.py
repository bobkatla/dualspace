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
import torch.distributions as D


class MDN(nn.Module):
    def __init__(self, d_in: int, d_c: int, hidden: int = 256, components: int = 6):
        super().__init__()
        self.components = components
        self.trunk = nn.Sequential(
        nn.Linear(d_in + d_c, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.head_logits = nn.Linear(hidden, components)
        self.head_mu = nn.Linear(hidden, components * d_in)
        self.head_logsigma = nn.Linear(hidden, components * d_in)
        self.d_in = d_in


    def _params(self, y: torch.Tensor, e: torch.Tensor):
        h = self.trunk(torch.cat([y, e], dim=-1))
        logits = self.head_logits(h)
        mu = self.head_mu(h).view(-1, self.components, self.d_in)
        logsig = self.head_logsigma(h).view(-1, self.components, self.d_in)
        return logits, mu, logsig


    def log_prob(self, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        logits, mu, logsig = self._params(y, e)
        # mixture of diagonal Gaussians
        comp = D.Independent(D.Normal(loc=mu, scale=torch.exp(logsig).clamp_min(1e-4)), 1)
        log_probs = comp.log_prob(y.unsqueeze(1).expand_as(mu)) # (B, M)
        return torch.logsumexp(log_probs + torch.log_softmax(logits, dim=-1), dim=-1)