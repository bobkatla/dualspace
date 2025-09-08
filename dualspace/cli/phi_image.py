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

    var_keep = float(cfg.get("pca_var_keep", 0.99))
    min_eig_val = float(cfg.get("pca_min_eig_val", 1e-4))
    Z = phi.fit(feats, out_dir, var_keep=var_keep, min_eig_val=min_eig_val)

    np.save(out_dir / "train_proj.npy", Z)
    print(f"[fit-phi] saved standardized PCA with {phi.pca.n_components_} comps to {out_dir}")
