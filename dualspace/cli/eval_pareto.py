from __future__ import annotations
import json, yaml, click, numpy as np, torch, joblib
from pathlib import Path
from numpy.linalg import slogdet
from dualspace.encoders.condition_encoder import ConditionEncoder
from dualspace.generators.diffusion_image import ImageDiffusion
from dualspace.encoders.phi_image import PhiImage
from dualspace.densities.mdn import MDN

def logdet_cov(Y: np.ndarray, eps: float = 1e-4) -> float:
    # Y: (n, d) in φ-space; regularized covariance
    if Y.shape[0] < 2: return float("-inf")
    C = np.cov(Y.T) + eps * np.eye(Y.shape[1])
    sign, val = slogdet(C)
    return float(val) if sign > 0 else float("-inf")

@click.command("pareto")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--K", type=int, default=256)
@click.option("--per-class/--pooled", default=True)
def pareto(config: str, k: int, per_class: bool):
    """For α grid, sample K drafts/class → keep by τ_α → compute (coverage_test, logdet_cov)."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    alphas = [float(a) for a in cfg.get("alphas", [0.5,0.7,0.8,0.9,0.95])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # models
    num_classes, d_c = int(cfg.get("num_classes", 10)), int(cfg.get("d_c", 64))
    g = ConditionEncoder(num_classes=num_classes, d_c=d_c, mode=cfg.get("g", {}).get("mode","linear_orth")).to(device).eval()
    T = int(cfg.get("T", 200))
    diffusion = ImageDiffusion(in_ch=3, d_c=d_c, T=T).to(device).eval()
    # pick latest ckpt
    ckpt = sorted((run_dir / "ckpts").glob("step_*.pt"))[-1]
    state = torch.load(ckpt, map_location=device)
    diffusion.load_state_dict(state["diffusion"])

    phi = PhiImage(d_out=int(cfg.get("d_phi", 128))).to(device).eval()
    phi.pca = joblib.load(run_dir / "phi" / "pca.joblib")

    mdn = MDN(d_c, int(cfg.get("d_phi", 128)),
              n_comp=int(cfg.get("mdn_components", 6)),
              hidden=int(cfg.get("mdn_hidden", 256))).to(device).eval()
    mdn.load_state_dict(torch.load(run_dir / "amortized" / "best.pt", map_location=device))

    # test pairs for coverage
    test = np.load(run_dir / "pairs" / "test.npz")
    E_test, Y_test, C_test = test["e"].astype(np.float32), test["y"].astype(np.float32), test["c"].astype(int)
    with torch.no_grad():
        logp_test = mdn.log_prob(torch.from_numpy(Y_test).to(device),
                                 torch.from_numpy(E_test).to(device)).cpu().numpy()

    # thresholds
    with open(run_dir / "conformal" / "taus.json", "r", encoding="utf-8") as f:
        taus = json.load(f)
    taus_pc = taus.get("per_class")
    taus_pool = taus.get("pooled")

    eye = torch.eye(num_classes, device=device)
    results = {"alpha_grid": alphas, "per_class": per_class, "points": {}}  # class -> list of dicts

    for c in range(num_classes):
        # sample drafts for class c once (reuse for all α)
        e_c = g(eye[c].unsqueeze(0)).repeat(k, 1)
        null_e = g.get_null(batch=k)
        with torch.no_grad():
            x_samp = diffusion.sample(e_c, K=k, guidance_scale=float(cfg.get("guidance_scale", 1.3)),
                                      shape=(3,32,32), null_e=null_e)
        # φ-projection
        feats = []
        for i in range(0, k, 32):
            with torch.no_grad():
                feats.append(phi(x_samp[i:i+32].to(device)).cpu().numpy())
        Ydraft = phi.transform(np.concatenate(feats, axis=0))  # (K, d_phi)
        Edraft = e_c.detach().cpu().numpy()

        # precompute logp for drafts
        with torch.no_grad():
            logp_draft = mdn.log_prob(torch.from_numpy(Ydraft).to(device),
                                      torch.from_numpy(Edraft).to(device)).cpu().numpy()

        results["points"][str(c)] = []
        for a in alphas:
            thr = (taus_pc[str(c)][f"{a:.3f}"] if per_class else taus_pool[f"{a:.3f}"])
            mask = logp_draft >= float(thr)
            kept = Ydraft[mask]
            # informativeness proxy
            info = logdet_cov(kept) if kept.shape[0] > 1 else float("-inf")
            # test coverage for this class (threshold is same regardless of drafts)
            mtest = (C_test == c)
            cov_test = float((logp_test[mtest] >= float(thr)).mean())
            results["points"][str(c)].append({
                "alpha": a, "coverage_test": cov_test,
                "region_logdetcov": info, "kept": int(mask.sum()), "total": int(k)
            })
        print(f"[pareto] class {c} done.")

    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pareto_points.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[pareto] wrote {out_dir/'pareto_points.json'}")
