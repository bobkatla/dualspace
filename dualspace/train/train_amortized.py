"""Train the MDN on dumped pairs.


- Minimize negative log-likelihood on (y,e).
- Save best checkpoint and training curves (JSON/NPZ).
"""
from __future__ import annotations
import yaml
import numpy as np
from pathlib import Path
from tqdm import trange
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from dualspace.densities.mdn import MDN
from dualspace.utils.io import save_json
from dualspace.utils.seed import set_seed


def load_pairs(path: Path):
    d = np.load(path)
    return torch.tensor(d["e"], dtype=torch.float32), torch.tensor(d["y"], dtype=torch.float32)


def train_amor(config: str):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 1337)))

    out_dir = Path(cfg.get("out_dir", "outputs/run")) / "amortized"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load pairs
    run_out = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    e_train, y_train = load_pairs(run_out / "pairs/train.npz")
    e_calib, y_calib = load_pairs(run_out / "pairs/calib.npz")

    d_in, d_out = e_train.size(1), y_train.size(1)
    mdn = MDN(d_in, d_out, n_comp=int(cfg.get("mdn_components", 6)),
              hidden=int(cfg.get("mdn_hidden", 256)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdn.to(device)

    opt = AdamW(mdn.parameters(), lr=float(cfg.get("lr_mdn", 1e-3)))
    batch_size = int(cfg.get("batch_size_mdn", 512))
    steps = int(cfg.get("train_steps_mdn", 10000))
    log_every = 200

    ds = TensorDataset(e_train, y_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_loss, patience, wait = 1e9, 20, 0
    history = {"train": [], "calib": []}
    for step in trange(steps):
        for eb, yb in loader:
            eb, yb = eb.to(device), yb.to(device)
            logp = mdn.log_prob(yb, eb)
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mdn.parameters(), 1.0)
            opt.step()

        if step % log_every == 0:
            with torch.no_grad():
                train_loss = -mdn.log_prob(y_train.to(device), e_train.to(device)).mean().item()
                calib_loss = -mdn.log_prob(y_calib.to(device), e_calib.to(device)).mean().item()
            history["train"].append(train_loss)
            history["calib"].append(calib_loss)
            print(f"[mdn] step={step} trainNLL={train_loss:.4f} calibNLL={calib_loss:.4f}")
            # early stop
            if calib_loss < best_loss:
                best_loss, wait = calib_loss, 0
                torch.save(mdn.state_dict(), out_dir / "best.pt")
            else:
                wait += 1
                if wait >= patience:
                    print("[mdn] early stop")
                    break

    save_json(out_dir / "history.json", history)
    print(f"[mdn] done. best calibNLL={best_loss:.4f}")
