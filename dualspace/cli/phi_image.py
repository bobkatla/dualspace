from __future__ import annotations
import click, yaml
import torch, numpy as np
from pathlib import Path
from tqdm import tqdm
from dualspace.encoders.phi_image import PhiImage
from dualspace.dataio.cifar10 import build_cifar10_loaders

@click.command("fit-phi")
@click.option("--config", type=click.Path(exists=True), required=True)
def fit_phi(config):
    """Extract features with frozen ResNet18 and fit PCA."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}")) / "phi"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders, _, _ = build_cifar10_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phi = PhiImage(d_out=int(cfg.get("d_phi", 64))).to(device)
    feats = []
    for xb, _ in tqdm(loaders["train"], desc="extract φ"):
        xb = xb.to(device)
        with torch.no_grad():
            f = phi(xb).cpu().numpy()
        feats.append(f)
    feats = np.concatenate(feats, axis=0)  # (N,512)

    Z = phi.fit_pca(feats, out_dir)
    np.save(out_dir / "train_proj.npy", Z)
    print(f"[fit-phi] saved PCA with {phi.d_out} comps → {out_dir}")
