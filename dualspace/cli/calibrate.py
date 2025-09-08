from __future__ import annotations
import yaml, click
import numpy as np
from pathlib import Path
import torch

from dualspace.densities.mdn import CondMDN
from dualspace.regions.conformal import TauAlpha


@click.command("calibrate")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--mode", type=click.Choice(["global", "class"]), default="class", help="Calibration mode.")
def calibrate(config: str, mode: str):
    """Conformal calibration: compute tau_alpha from calib (e,y) using the trained MDN."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    pairs = np.load(run_dir / "pairs" / "calib.npz")
    E, Y, C = pairs["e"].astype(np.float32), pairs["y"].astype(np.float32), pairs["c"].astype(int)

    # Load MDN
    mdn = CondMDN.load(run_dir / "amortized" / "best.pt", map_location="cpu")
    mdn.eval()

    # Compute scores s = -log p(y|e)
    with torch.no_grad():
        logp = mdn.log_prob(torch.from_numpy(Y), torch.from_numpy(E)).cpu().numpy()
    scores = -logp  # lower score is better (more likely)

    # Alphas
    alphas = cfg.get("alphas", [0.5, 0.7, 0.8, 0.9, 0.95])
    
    # Fit and save thresholds
    out_dir = run_dir / "conformal"
    tau_handler = TauAlpha(mode=mode, alphas=alphas)
    tau_handler.fit(scores=scores, conditions=C)
    tau_handler.save(out_dir / "taus.json")
    
    click.echo(f"[calibrate] Saved {mode}-mode thresholds to {out_dir/'taus.json'}")
