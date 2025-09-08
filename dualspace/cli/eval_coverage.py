from __future__ import annotations
import json, yaml, click, numpy as np, torch
from pathlib import Path
from dualspace.densities.mdn import CondMDN

@click.command("coverage-curve")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--per-class/--pooled", default=True)
def coverage_curve(config: str, per_class: bool):
    """Empirical coverage on TEST across α ∈ config['alphas']."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    alphas = [float(a) for a in cfg.get("alphas", [0.5,0.7,0.8,0.9,0.95])]

    # load pairs (test)
    test = np.load(run_dir / "pairs" / "test.npz")
    E, Y, C = test["e"].astype(np.float32), test["y"].astype(np.float32), test["c"].astype(int)
    d_in, d_out = E.shape[1], Y.shape[1]

    # load MDN
    mdn = CondMDN.load(run_dir / "amortized" / "best.pt", map_location="cpu")
    mdn.eval()
    with torch.no_grad():
        logp = mdn.log_prob(torch.from_numpy(Y), torch.from_numpy(E)).cpu().numpy()

    # load taus
    with open(run_dir / "conformal" / "taus.json", "r", encoding="utf-8") as f:
        taus = json.load(f)
    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if per_class:
        taus_pc = taus["per_class"]
        cov_pc = {}
        for k in np.unique(C):
            m = (C == k)
            cov_pc[int(k)] = {f"{a:.3f}": float((logp[m] >= float(taus_pc[str(int(k))][f"{a:.3f}"])).mean())
                              for a in alphas}
        results["coverage_test_per_class"] = cov_pc
        # macro-avg
        macro = {f"{a:.3f}": float(np.mean([cov_pc[int(k)][f"{a:.3f}"] for k in cov_pc])) for a in alphas}
        results["coverage_test_macro"] = macro
    else:
        taus_pool = taus["pooled"]
        results["coverage_test_pooled"] = {f"{a:.3f}": float((logp >= float(taus_pool[f"{a:.3f}"])).mean())
                                           for a in alphas}

    # save json
    with open(out_dir / "coverage_curve_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[coverage-curve] wrote {out_dir/'coverage_curve_test.json'}")
