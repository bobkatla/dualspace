from __future__ import annotations
from pathlib import Path
import torch
import torchvision.utils as vutils


def save_image_grid(x: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    """Save a grid of images in [0,1] with shape (N,3,H,W)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(x, str(path), nrow=nrow)