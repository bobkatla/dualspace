from __future__ import annotations
import yaml, click
import numpy as np
from pathlib import Path
import torch

from dualspace.densities.mdn import MDN
from dualspace.regions.conformal import per_class_thresholds, pooled_thresholds, save_json


@click.command("calibrate")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--pooled", is_flag=True, help="Use pooled (not per-class) thresholds")
def calibrate(config: str, pooled: bool):
    """Conformal calibration: compute tau_alpha from calib (e,y) using the trained MDN."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    pairs = np.load(run_dir / "pairs" / "calib.npz")
    E, Y, C = pairs["e"].astype(np.float32), pairs["y"].astype(np.float32), pairs["c"].astype(int)
    d_in, d_out = E.shape[1], Y.shape[1]

    # Load MDN
    mdn = MDN(d_in, d_out, n_comp=int(cfg.get("mdn_components", 6)), hidden=int(cfg.get("mdn_hidden", 256)))
    state = torch.load(run_dir / "amortized" / "best.pt", map_location="cpu")
    mdn.load_state_dict(state)
    mdn.eval()

    # Compute scores s = -log p(y|e)
    with torch.no_grad():
        logp = mdn.log_prob(torch.from_numpy(Y), torch.from_numpy(E)).cpu().numpy()
    scores = -logp  # lower is better (inside region)

    # Alphas
    alphas = cfg.get("alphas", [0.5, 0.7, 0.8, 0.9, 0.95])

    out_dir = run_dir / "conformal"
    out_dir.mkdir(parents=True, exist_ok=True)

    if pooled:
        taus = pooled_thresholds(scores, alphas)
        save_json({"pooled": taus, "alphas": alphas}, out_dir / "taus.json")
        # coverage on calib (pooled)
        cov = {f"{a:.3f}": float((logp >= taus[f"{a:.3f}"]).mean()) for a in alphas}
        save_json({"coverage_calib_pooled": cov}, out_dir / "coverage_on_calib.json")
        click.echo(f"[calibrate] pooled taus → {out_dir/'taus.json'}")
    else:
        # group by class
        scores_by_k = {}
        logp_by_k = {}
        for k in np.unique(C):
            m = (C == k)
            scores_by_k[int(k)] = scores[m]
            logp_by_k[int(k)] = logp[m]
        taus_pc = per_class_thresholds(scores_by_k, alphas)
        save_json({"per_class": taus_pc, "alphas": alphas}, out_dir / "taus.json")

        # coverage on calib (per-class)
        cov_pc = {}
        for k, lp in logp_by_k.items():
            cov_pc[k] = {f"{a:.3f}": float((lp >= taus_pc[k][f"{a:.3f}"]).mean()) for a in alphas}
        save_json({"coverage_calib_per_class": cov_pc}, out_dir / "coverage_on_calib.json")
        click.echo(f"[calibrate] per-class taus → {out_dir/'taus.json'}")
