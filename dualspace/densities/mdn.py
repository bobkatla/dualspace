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
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from pathlib import Path


class CondMDN(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_comp: int = 6, hidden: int = 256):
        super().__init__()
        self.d_in, self.d_out, self.n_comp = d_in, d_out, n_comp
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
        # clamp variance to avoid numerical instability
        var = torch.clamp(var, min=1e-6)
        log_comp = -0.5 * (((y_exp - mu) ** 2) / var + torch.log(var) + torch.log(torch.tensor(2*torch.pi, device=y.device)))
        log_comp = log_comp.sum(dim=-1)       # (B,K)
        log_mix = torch.logsumexp(log_pi + log_comp, dim=-1)  # (B,)
        return log_mix

    def fit(self,
            e_train: torch.Tensor, y_train: torch.Tensor,
            e_val: torch.Tensor, y_val: torch.Tensor,
            out_dir: Path,
            lr: float = 1e-3,
            batch_size: int = 512,
            steps: int = 10000,
            patience: int = 20):
        """Train the MDN with early stopping."""
        self.train()
        device = next(self.parameters()).device
        opt = AdamW(self.parameters(), lr=lr)
        ds = TensorDataset(e_train, y_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        best_loss, wait = float('inf'), 0
        history = {"train": [], "val": []}

        pbar = trange(steps, desc="Training MDN")
        for step in pbar:
            for eb, yb in loader:
                eb, yb = eb.to(device), yb.to(device)
                loss = -self.log_prob(yb, eb).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

            # Validation and early stopping
            if step % 100 == 0:
                self.eval()
                with torch.no_grad():
                    val_loss = -self.log_prob(y_val.to(device), e_val.to(device)).mean().item()
                history["val"].append(val_loss)
                pbar.set_postfix(val_nll=f"{val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    self.save(out_dir / "best.pt")
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at step {step}. Best val loss: {best_loss:.4f}")
                        break
                self.train()
        
        return history

    def save(self, path: Path):
        """Saves model state and architecture parameters."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'd_in': self.d_in,
            'd_out': self.d_out,
            'n_comp': self.n_comp,
            'hidden': self.net[0].out_features,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path, map_location=None):
        """Loads a CondMDN model."""
        ckpt = torch.load(path, map_location=map_location)
        model = cls(
            d_in=ckpt['d_in'],
            d_out=ckpt['d_out'],
            n_comp=ckpt['n_comp'],
            hidden=ckpt['hidden']
        )
        model.load_state_dict(ckpt['state_dict'])
        return model
