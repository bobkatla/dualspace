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
from sklearn.preprocessing import StandardScaler
import joblib
import json


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
        self.scaler: StandardScaler | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,32,32) in [-1,1]; resize → (B,3,224,224)
        x = torch.nn.functional.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        feats = self.backbone(x).flatten(1)  # (B,512)
        return feats

    def fit(self, X_train: np.ndarray, out_dir: Path, var_keep: float = 0.99, min_eig_val: float = 1e-4):
        """Fits standardizer and PCA on training data and saves them."""
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Fit and save standardizer
        self.scaler = StandardScaler()
        X_train_std = self.scaler.fit_transform(X_train)

        scaler_stats = {'mean': self.scaler.mean_.tolist(), 'scale': self.scaler.scale_.tolist()}
        with open(out_dir / "phi_stats.json", "w", encoding="utf-8") as f:
            json.dump(scaler_stats, f, indent=2)

        # 2. Fit PCA with max components to decide how many to trim
        pca_full = PCA(n_components=self.d_out, svd_solver="auto", random_state=1337)
        pca_full.fit(X_train_std)

        # 3. Trim components based on cumulative variance and minimum eigenvalue
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_by_var = np.searchsorted(cum_var, var_keep, side='right') + 1
        n_by_eig = np.sum(pca_full.explained_variance_ >= min_eig_val)
        n_components = min(self.d_out, int(n_by_var), int(n_by_eig))

        # 4. Refit with trimmed components and save
        self.pca = PCA(n_components=n_components, svd_solver="auto", random_state=1337)
        Z_train = self.pca.fit_transform(X_train_std)

        joblib.dump(self.pca, out_dir / "pca.joblib")
        np.savez(
            out_dir / "pca_stats.npz",
            n_components=self.pca.n_components_,
            explained_variance=self.pca.explained_variance_,
            explained_variance_ratio=self.pca.explained_variance_ratio_,
        )
        return Z_train

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies fitted standardizer and PCA."""
        if self.scaler is None or self.pca is None:
            raise RuntimeError("Must call fit() or load() before transforming data.")
        X_std = self.scaler.transform(X)
        return self.pca.transform(X_std)

    def load(self, model_dir: Path):
        """Loads a pre-fitted standardizer and PCA model."""
        with open(model_dir / "phi_stats.json", "r", encoding="utf-8") as f:
            scaler_stats = json.load(f)
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(scaler_stats['mean'])
            self.scaler.scale_ = np.array(scaler_stats['scale'])
            self.scaler.n_features_in_ = len(self.scaler.mean_)
        
        self.pca = joblib.load(model_dir / "pca.joblib")
        self.d_out = self.pca.n_components_
        return self
