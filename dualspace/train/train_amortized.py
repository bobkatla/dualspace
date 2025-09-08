"""Train the MDN on dumped pairs.


- Minimize negative log-likelihood on (y,e).
- Save best checkpoint and training curves (JSON/NPZ).
"""
from __future__ import annotations
import yaml
import numpy as np
from pathlib import Path
import torch

from dualspace.densities.mdn import CondMDN
from dualspace.utils.io import save_json
from dualspace.utils.seed import set_seed


def load_pairs(path: Path):
    d = np.load(path)
    return torch.tensor(d["e"], dtype=torch.float32), torch.tensor(d["y"], dtype=torch.float32)


def train_amor(config: str):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 1337)))

    run_out = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    out_dir = run_out / "amortized"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load pairs
    e_train, y_train = load_pairs(run_out / "pairs/train.npz")
    e_calib, y_calib = load_pairs(run_out / "pairs/calib.npz")

    d_in, d_out = e_train.size(1), y_train.size(1)
    mdn = CondMDN(d_in, d_out, n_comp=int(cfg.get("mdn_components", 6)),
              hidden=int(cfg.get("mdn_hidden", 256)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdn.to(device)

    history = mdn.fit(
        e_train, y_train, e_calib, y_calib,
        out_dir=out_dir,
        lr=float(cfg.get("lr_mdn", 1e-3)),
        batch_size=int(cfg.get("batch_size_mdn", 512)),
        steps=int(cfg.get("train_steps_mdn", 10000)),
        patience=int(cfg.get("patience_mdn", 20))
    )

    save_json(out_dir / "history.json", history)
    print(f"[mdn] Training complete. Checkpoints and history saved to {out_dir}")
