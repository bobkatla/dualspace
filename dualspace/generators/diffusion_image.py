"""Conditional diffusion wrapper for images with classifier-free guidance.


Train loop will optimize MSE between predicted and true noise.


forward_api:
sample(c, K, guidance_scale) -> x: (K, C, H, W)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet2d import UNet2DSmall
from .scheduler import cosine_beta_schedule


class ImageDiffusion(nn.Module):
    def __init__(self, in_ch: int = 3, d_c: int = 64, T: int = 200, p_uncond: float = 0.1):
        super().__init__()
        self.unet = UNet2DSmall(in_ch=in_ch, d_c=d_c)
        self.T = T
        self.p_uncond = p_uncond
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=alphas.device), alphas_cumprod[:-1]])
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (self.sqrt_alphas_cumprod[t].view(-1,1,1,1) * x0 +
                self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1) * noise)

    def loss(self, x0: torch.Tensor, e: torch.Tensor, null_e: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        # classifier-free: randomly drop conditioning
        drop = (torch.rand(B, device=x0.device) < self.p_uncond).float().view(B,1)
        e_used = e * (1.0 - drop) + null_e * drop
        eps_pred = self.unet(x_t, t, e_used)
        return F.mse_loss(eps_pred, noise)
    
    # replace p_sample() with DDPM posterior
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, e: torch.Tensor,
                guidance_scale: float = 1.5, null_e: torch.Tensor | None = None) -> torch.Tensor:
        t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        eps_c = self.unet(x, t_tensor, e)
        if null_e is not None and guidance_scale != 1.0:
            eps_u = self.unet(x, t_tensor, null_e.expand_as(e))
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        else:
            eps = eps_c

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        abar_t = self.alphas_cumprod[t]
        abar_prev = self.alphas_cumprod_prev[t]
        # x0 prediction (eps-param)
        x0_hat = (x - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)
        x0_hat = x0_hat.clamp(-1, 1)   # dynamic thresholding-lite
        # Posterior q(x_{t-1} | x_t, x0)
        coeff1 = torch.sqrt(abar_prev) * beta_t / (1 - abar_t)
        coeff2 = torch.sqrt(alpha_t)    * (1 - abar_prev) / (1 - abar_t)
        mean = coeff1 * x0_hat + coeff2 * x
        # True posterior variance (improves stability)
        var = beta_t * (1 - abar_prev) / (1 - abar_t)
        if t > 0:
            noise = torch.randn_like(x)
            x_next = mean + torch.sqrt(var) * noise
        else:
            x_next = mean
        return x_next

    @torch.no_grad()
    def sample(self, e: torch.Tensor, K: int = 64, guidance_scale: float = 2.0, shape=(3,32,32), null_e: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.randn((K, *shape), device=e.device)
        if null_e is None:
            null_e = torch.zeros_like(e[:1])
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t, e, guidance_scale, null_e)
        return x