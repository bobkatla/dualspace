"""End-to-end inference: given c, sample K drafts, filter by conformal thresholds (tau_alpha),
save survivors + metrics


CLI
---
python -m dualspace.infer.region_infer --config configs/cifar10.yaml --split test
"""
from __future__ import annotations
import click, yaml, torch, numpy as np
from pathlib import Path
import joblib

from dualspace.encoders.condition_encoder import ConditionEncoder
from dualspace.generators.diffusion_image import ImageDiffusion
from dualspace.encoders.phi_image import PhiImage
from dualspace.densities.mdn import MDN
from dualspace.utils.io import save_json
from dualspace.utils.vision import save_image_grid


@click.command("region-infer")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--alpha", type=float, default=0.9, help="Target coverage α")
@click.option("--per-class/--pooled", default=True)
@click.option("--K", type=int, default=256, help="#drafts per condition")
def region_infer(config: str, alpha: float, per_class: bool, k: int):
    """Given conditions, sample K drafts, score in φ-space, keep those above τ_α."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    out_dir = run_dir / "region_infer"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load g(c)
    num_classes, d_c = int(cfg.get("num_classes", 10)), int(cfg.get("d_c", 64))
    gcfg = cfg.get("g", {})
    g = ConditionEncoder(num_classes=num_classes, d_c=d_c,
                         hidden=int(gcfg.get("hidden", 256)),
                         depth=int(gcfg.get("depth", 2)),
                         dropout=float(gcfg.get("dropout", 0.0)),
                         norm=gcfg.get("norm", None),
                         mode=gcfg.get("mode", "linear_orth"))
    g.to(device).eval()

    # Load diffusion (EMA ckpt recommended)
    ckpt = torch.load(run_dir / "ckpts" / "step_020000.pt", map_location=device)
    diffusion = ImageDiffusion(in_ch=3, d_c=d_c, T=int(cfg.get("T", 200)))
    diffusion.load_state_dict(ckpt["diffusion"])
    diffusion.to(device).eval()

    # Load φ with PCA
    phi = PhiImage(d_out=int(cfg.get("d_phi", 128))).to(device)
    phi.pca = joblib.load(run_dir / "phi" / "pca.joblib")

    # Load MDN
    d_in, d_out = d_c, int(cfg.get("d_phi", 128))
    mdn = MDN(d_in, d_out, n_comp=int(cfg.get("mdn_components", 6)), hidden=int(cfg.get("mdn_hidden", 256)))
    state = torch.load(run_dir / "amortized" / "best.pt", map_location=device)
    mdn.load_state_dict(state)
    mdn.to(device).eval()

    # Load taus
    taus = yaml.safe_load(open(run_dir / "conformal" / "taus.json", "r"))
    if per_class:
        taus = taus["per_class"]
    else:
        taus = taus["pooled"]

    eye = torch.eye(num_classes, device=device)
    for c in range(num_classes):
        e = g(eye[c].unsqueeze(0)).repeat(k, 1)
        null_e = g.get_null(batch=k)
        # sample K
        with torch.no_grad():
            x_samp = diffusion.sample(e, K=k, guidance_scale=float(cfg.get("guidance_scale", 1.3)), shape=(3,32,32), null_e=null_e)
        img = (x_samp.clamp(-1,1) + 1.0) / 2.0

        # project φ
        feats = []
        for i in range(0, k, 32):
            xb = x_samp[i:i+32].to(device)
            f = phi(xb).cpu().numpy()
            feats.append(f)
        feats = np.concatenate(feats, axis=0)
        Y = phi.transform(feats)   # (K,d_phi)
        E = e.detach().cpu().numpy()

        # scores
        with torch.no_grad():
            logp = mdn.log_prob(torch.from_numpy(Y).to(device), torch.from_numpy(E).to(device)).cpu().numpy()

        # threshold
        if per_class:
            thr = taus[str(c)][f"{alpha:.3f}"]
        else:
            thr = taus[f"{alpha:.3f}"]

        mask = logp >= thr
        survivors = img[mask]

        # save
        save_image_grid(img[:64], out_dir / f"class{c}_all.png", nrow=8)
        if survivors.size(0) > 0:
            save_image_grid(survivors[:64], out_dir / f"class{c}_survivors.png", nrow=8)

        cov = float(mask.mean())
        save_json(out_dir / f"class{c}_stats.json", {"coverage": cov, "thr": thr, "kept": int(mask.sum()), "total": k})

        print(f"[region-infer] class {c}: kept {mask.sum()}/{k} (cov={cov:.3f})")
