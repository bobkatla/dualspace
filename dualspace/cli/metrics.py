# Example stubs (adapt to your artifact paths)
import click, numpy as np, torch
from dualspace.metrics.informativeness import logdet_cov_phi, farthest_point_sampling
from dualspace.metrics.mmd import mmd_rbf
from dualspace.metrics.ot_sinkhorn import sinkhorn_cost
from dualspace.metrics.fid import fid_from_tensors
from pathlib import Path
import yaml

from dualspace.encoders.phi_image import PhiImage
from dualspace.densities.mdn import CondMDN
from dualspace.regions.conformal import TauAlpha
from dualspace.regions.levelset import LevelSetRegion
from dualspace.utils.io import save_json


PREFERRED_KEYS = ("Y", "y", "X", "features", "emb", "phi", "arr_0")

def _load_array(maybe_npz_or_npy: str) -> np.ndarray:
    p = Path(maybe_npz_or_npy)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")

    if p.suffix == ".npy":
        arr = np.load(p)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{p} did not load to ndarray")
        return arr

    if p.suffix == ".npz":
        data = np.load(p)
        # np.load on .npz returns an NpzFile dict-like
        if hasattr(data, "files"):
            for k in PREFERRED_KEYS:
                if k in data.files:
                    return data[k]
            # Fallback: if there is exactly one array, return it
            if len(data.files) == 1:
                return data[data.files[0]]
            raise KeyError(
                f"{p} is .npz but missing any of keys {PREFERRED_KEYS}; "
                f"found keys={list(data.files)}"
            )
        # rare: plain ndarray stored as .npz
        if isinstance(data, np.ndarray):
            return data

    raise ValueError(f"Unsupported file type for {p} (need .npy or .npz)")

@click.command("metrics")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--alphas", type=str, default="0.9", help="Comma-separated coverage levels to evaluate.")
@click.option("--reports", type=str, default="coverage,informativeness", help="Comma-separated metrics to report.")
def metrics(config: str, alphas: str, reports: str):
    """
    Computes a suite of metrics for the deterministic regions defined by the trained models.
    """
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    run_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_list = [float(a) for a in alphas.split(',')]
    report_list = [r.strip() for r in reports.split(',')]

    # --- Load components to define the region ---
    phi = PhiImage().to(device).eval()
    phi.load(run_dir / "phi")
    mdn = CondMDN.load(run_dir / "amortized" / "best.pt", map_location=device).eval()
    tau_handler = TauAlpha.load(run_dir / "conformal" / "taus.json")
    region = LevelSetRegion(phi=phi, mdn=mdn, tau_handler=tau_handler)

    # --- Load data ---
    # Real test data
    test_pairs = np.load(run_dir / "pairs" / "test.npz")
    e_test, y_test, c_test = test_pairs["e"], test_pairs["y"], test_pairs["c"]

    # Generated data (from region_infer step)
    y_survivors_by_class = {}
    for c in range(int(cfg.get("num_classes", 10))):
        path = run_dir / "region_infer" / f"class{c}_survivors_phi.npz"
        if path.exists():
            y_survivors_by_class[c] = np.load(path)['Y']

    # --- Compute and save metrics ---
    results = {"alphas": alpha_list, "reports": {}}
    for report in report_list:
        results["reports"][report] = {}

    for alpha in alpha_list:
        print(f"--- Evaluating for alpha = {alpha:.3f} ---")
        
        # 1. Coverage (on real test data)
        if 'coverage' in report_list:
            mask = region.contains_y(y_test, e_test, c_test, alpha)
            coverage = mask.mean()
            results["reports"]["coverage"][f"{alpha:.3f}"] = coverage
            print(f"  Coverage: {coverage:.4f}")

        # 2. Informativeness (on generated survivors)
        if 'informativeness' in report_list:
            info_by_class = {}
            for c, y_survivors in y_survivors_by_class.items():
                # The survivors are already filtered at a specific alpha, need to re-check which one.
                # For now, let's assume `region_infer` was run with the same alpha.
                info_by_class[c] = logdet_cov_phi(y_survivors)
            
            avg_info = np.mean(list(info_by_class.values()))
            results["reports"]["informativeness"][f"{alpha:.3f}"] = avg_info
            print(f"  Avg. Informativeness (logdet_cov): {avg_info:.4f}")

        # Add other metrics like MMD, Sinkhorn here if needed.
        # Example: MMD between survivors of class 0 and class 1
        if 'mmd' in report_list and 0 in y_survivors_by_class and 1 in y_survivors_by_class:
            mmd_0_1 = mmd_rbf(y_survivors_by_class[0], y_survivors_by_class[1])
            results["reports"]["mmd"][f"{alpha:.3f}_0v1"] = mmd_0_1
            print(f"  MMD (class 0 vs 1): {mmd_0_1:.4f}")

    save_json(out_dir / "final_metrics.json", results)
    print(f"Metrics saved to {out_dir / 'final_metrics.json'}")
