"""Feature map phi(x) for images using a frozen ResNet18 + PCA.


API
---
class ImagePhi(nn.Module):
forward(x: Float[Tensor, B,3,H,W]) -> Float[Tensor, B, D_phi]
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.decomposition import PCA
import joblib


class PhiImage(nn.Module):
    def __init__(self, d_out: int = 64):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        layers = list(base.children())[:-1]  # drop fc
        self.backbone = nn.Sequential(*layers)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.d_in = base.fc.in_features
        self.d_out = d_out
        self.pca: PCA | None = None
        self.mean_: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,32,32) in [-1,1]; resize → (B,3,224,224)
        x = torch.nn.functional.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        feats = self.backbone(x).flatten(1)  # (B,512)
        return feats

    def fit_pca(self, X: np.ndarray, out_dir: str | Path):
        self.pca = PCA(n_components=self.d_out, svd_solver="auto", random_state=1337)
        Z = self.pca.fit_transform(X)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca, Path(out_dir) / "pca.joblib")
        np.save(Path(out_dir) / "pca_mean.npy", self.pca.mean_)
        np.save(Path(out_dir) / "pca_var.npy", self.pca.explained_variance_ratio_)
        return Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.pca is not None, "fit_pca() first"
        return self.pca.transform(X)
