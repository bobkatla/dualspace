"""Minimal U-Net 2D backbone for diffusion on CIFAR-10.

Forward signature
-----------------
forward(x_t, t, e) -> eps_pred with shapes matching x_t.


TODO
----
- Implement time embedding, FiLM-like conditioning on e.
- Small/medium variants selectable via config.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- helpers ----
def sinusoidal_timestep_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / (half - 1))
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class FiLM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * 2)
    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g, b = self.linear(cond).chunk(2, dim=-1)
        return h * (1 + g.unsqueeze(-1).unsqueeze(-1)) + b.unsqueeze(-1).unsqueeze(-1)

class ResBlock(nn.Module):
    def __init__(self, ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film_t = FiLM(emb_dim, ch)
        self.film_e = FiLM(emb_dim, ch)
    def forward(self, x, t_emb, e_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film_t(h, t_emb)
        h = self.film_e(h, e_emb)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res = ResBlock(out_ch, emb_dim)
    def forward(self, x, t_emb, e_emb):
        x = self.conv(x)
        x = self.res(x, t_emb, e_emb)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch, emb_dim)
    def forward(self, x, skip, t_emb, e_emb):
        x = self.conv(x)
        x = x + skip
        x = self.res(x, t_emb, e_emb)
        return x

class UNet2DSmall(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64, d_c: int = 64, t_dim: int = 128):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.e_mlp = nn.Sequential(
            nn.Linear(d_c, t_dim), nn.SiLU(), nn.Linear(d_c if d_c==t_dim else t_dim, t_dim)
        )
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.d1 = Down(base, base*2, t_dim)
        self.d2 = Down(base*2, base*4, t_dim)
        self.mid = ResBlock(base*4, t_dim)
        self.u2 = Up(base*4, base*2, t_dim)
        self.u1 = Up(base*2, base, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, in_ch, 3, padding=1))
        self.t_dim = t_dim
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(sinusoidal_timestep_embed(t, self.t_dim))
        e_emb = self.e_mlp(e)
        x0 = self.in_conv(x)
        d1 = self.d1(x0, t_emb, e_emb)
        d2 = self.d2(d1, t_emb, e_emb)
        mid = self.mid(d2, t_emb, e_emb)
        u2 = self.u2(mid, d1, t_emb, e_emb)
        u1 = self.u1(u2, x0, t_emb, e_emb)
        out = self.out(u1)
        return out