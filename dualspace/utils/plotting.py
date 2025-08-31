from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_cosine_heatmap(mat: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.imshow(mat, vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("g(c) cosine similarity")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()