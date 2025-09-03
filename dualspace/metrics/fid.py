"""FID (Fréchet Inception Distance) using torchvision InceptionV3 pool3 features."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class InceptionPool3(nn.Module):
    def __init__(self):
        super().__init__()
        inc = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        inc.Mixed_7c.register_forward_hook(self._hook)  # pool3 before fc
        self.inc = inc.eval()
        for p in self.inc.parameters(): p.requires_grad = False
        self._feat = None

    def _hook(self, module, inp, out):
        # out: (B, 2048, 8, 8), apply global avg pool → (B,2048)
        self._feat = F.adaptive_avg_pool2d(out, (1,1)).flatten(1)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,3,H,W) in [0,1]; resized to 299x299 with inception preprocessing.
        Returns (N,2048).
        """
        if x.dtype != torch.float32:
            x = x.float()
        x = F.interpolate(x, size=(299,299), mode="bilinear", align_corners=False)
        # Inception expects [0,1]; use default weight transforms if needed
        _ = self.inc(x)  # triggers hook
        f = self._feat
        self._feat = None
        return f


def _act_stats(FEATS: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = FEATS.mean(axis=0)
    sig = np.cov(FEATS, rowvar=False)
    return mu, sig


def _frechet_distance(m1, C1, m2, C2, eps: float = 1e-6) -> float:
    # Stable Fréchet distance (FID) implementation
    from scipy.linalg import sqrtm
    diff = m1 - m2
    covmean = sqrtm(C1 @ C2)
    if not np.isfinite(covmean).all():
        # add eps to the diagonal
        C1 = C1 + np.eye(C1.shape[0]) * eps
        C2 = C2 + np.eye(C2.shape[0]) * eps
        covmean = sqrtm(C1 @ C2)
    # sometimes sqrtm returns complex due to precision
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(C1 + C2 - 2*covmean)
    return float(fid)


@torch.no_grad()
def fid_from_tensors(x_real: torch.Tensor, x_fake: torch.Tensor, device: torch.device | None = None) -> float:
    """
    x_real, x_fake: tensors in [0,1], shape (N,3,H,W). Computes FID.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = InceptionPool3().to(device)
    fr = model.features(x_real.to(device)).cpu().numpy()
    ff = model.features(x_fake.to(device)).cpu().numpy()
    m1, C1 = _act_stats(fr); m2, C2 = _act_stats(ff)
    return _frechet_distance(m1, C1, m2, C2)
