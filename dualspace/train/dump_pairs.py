"""Dump (e=g(c), y=phi(x)) pairs to NPZ for a given split.


- Use real data only for calibration; optionally include generated.
"""
from __future__ import annotations
from pathlib import Path
import yaml, torch, numpy as np
from tqdm import tqdm
import joblib

from dualspace.dataio.cifar10 import build_cifar10_loaders
from dualspace.encoders.condition_encoder import ConditionEncoder
from dualspace.encoders.phi_image import PhiImage
from dualspace.utils.seed import set_seed

def get_dump_pairs(config: str):
    """Project CIFAR-10 images → φ(x), encode conditions → e, save (e,y,labels)."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 1337)))

    out_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}")) / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # data
    loaders, _, _ = build_cifar10_loaders(cfg)
    num_classes = int(cfg.get("num_classes", 10))
    d_c = int(cfg.get("d_c", 64))
    gcfg = cfg.get("g", {})

    # models
    g = ConditionEncoder(num_classes=num_classes, d_c=d_c,
                         hidden=int(gcfg.get("hidden", 256)),
                         depth=int(gcfg.get("depth", 2)),
                         dropout=float(gcfg.get("dropout", 0.0)),
                         norm=gcfg.get("norm", None),
                         mode=gcfg.get("mode", "linear_orth"),
                         orth_reg=float(gcfg.get("orth_reg", 0.0)))
    g.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g.to(device)

    # φ extractor + PCA
    phi = PhiImage(d_out=int(cfg.get("d_phi", 128))).to(device)
    pca = joblib.load(Path(cfg.get("out_dir", "outputs/run")) / "phi" / "pca.joblib")
    phi.pca = pca  # attach PCA

    eye = torch.eye(num_classes, device=device)

    for split in ["train", "calib", "test"]:
        all_e, all_y, all_c = [], [], []
        for xb, yb in tqdm(loaders[split], desc=f"pairs-{split}"):
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                feats = phi(xb).cpu().numpy()     # (B,512)
                y_proj = phi.transform(feats)    # (B,d_phi)
                c_onehot = eye[yb]               # (B,C)
                e = g(c_onehot).cpu().numpy()    # (B,d_c)
            all_e.append(e)
            all_y.append(y_proj)
            all_c.append(yb.cpu().numpy())

        E = np.concatenate(all_e, axis=0)
        Y = np.concatenate(all_y, axis=0)
        C = np.concatenate(all_c, axis=0)
        np.savez_compressed(out_dir / f"{split}.npz", e=E, y=Y, c=C)
        print(f"[dump-pairs] {split}: saved {E.shape}, {Y.shape}, {C.shape} → {out_dir}")
