"""End-to-end inference: given c, sample K drafts, filter by conformal thresholds (tau_alpha),
save survivors + metrics


CLI
---
python -m dualspace.infer.region_infer --config configs/cifar10.yaml --split test
"""
from __future__ import annotations
import click, yaml, torch, numpy as np
from pathlib import Path

from dualspace.encoders.condition_encoder import ConditionEncoder
from dualspace.generators.diffusion_image import ImageDiffusion
from dualspace.encoders.phi_image import PhiImage
from dualspace.densities.mdn import CondMDN
from dualspace.regions.conformal import TauAlpha
from dualspace.regions.levelset import LevelSetRegion
from dualspace.utils.io import save_json
from dualspace.utils.vision import save_image_grid


@click.command("region-infer")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--alpha", type=float, default=0.9, help="Target coverage α")
@click.option("--viz-samples", type=int, default=256, help="# drafts to generate for visualization")
@click.option("--region-mode", type=click.Choice(['levelset']), default='levelset')
def region_infer(config: str, alpha: float, viz_samples: int, region_mode: str):
    """
    For a given condition, defines a deterministic region and optionally visualizes it
    by generating samples and filtering them.
    """
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    out_dir = run_dir / "region_infer"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load all components for the LevelSetRegion ---
    # Condition Encoder g(c)
    num_classes = int(cfg.get("num_classes", 10))
    d_c = int(cfg.get("d_c", 64))
    gcfg = cfg.get("g", {})
    g = ConditionEncoder(num_classes=num_classes, d_c=d_c,
                         hidden=int(gcfg.get("hidden", 256)),
                         depth=int(gcfg.get("depth", 2)),
                         mode=gcfg.get("mode", "linear_orth"))
    # No need to load g weights if we use the saved (e,y) pairs.
    # But for sampling new images, we need g and diffusion.
    
    # Diffusion model (for visualization sampling)
    ckpt_path = sorted((run_dir / "ckpts").glob("step_*.pt"))[-1]
    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion = ImageDiffusion(in_ch=3, d_c=d_c, T=int(cfg.get("T", 200)))
    diffusion.load_state_dict(ckpt["diffusion"])
    diffusion.to(device).eval()
    g.to(device).eval() # g is needed for diffusion

    # Phi processor
    phi = PhiImage(d_out=int(cfg.get("d_phi", 128))).to(device).eval()
    phi.load(run_dir / "phi")

    # Amortized density model
    mdn = CondMDN.load(run_dir / "amortized" / "best.pt", map_location=device).eval()

    # Conformal thresholds
    tau_handler = TauAlpha.load(run_dir / "conformal" / "taus.json")
    
    # The deterministic region object
    region = LevelSetRegion(phi=phi, mdn=mdn, tau_handler=tau_handler)

    # --- Main loop: visualize region for each class ---
    eye = torch.eye(num_classes, device=device)
    for c in range(num_classes):
        # Sample K drafts for visualization
        e = g(eye[c].unsqueeze(0)).repeat(viz_samples, 1)
        null_e = g.get_null(batch=viz_samples)
        
        with torch.no_grad():
            x_samp = diffusion.sample(
                e, K=viz_samples, 
                guidance_scale=float(cfg.get("guidance_scale", 1.3)), 
                shape=(3,32,32), 
                null_e=null_e
            )
        img_drafts = (x_samp.clamp(-1,1) + 1.0) / 2.0

        # Project drafts into phi-space to check for containment
        feats = []
        for i in range(0, viz_samples, 32):
            xb = x_samp[i:i+32].to(device)
            f = phi(xb).cpu().numpy()
            feats.append(f)
        feats = np.concatenate(feats, axis=0)
        Y_drafts = phi.transform(feats)
        E_drafts = e.detach().cpu().numpy()
        C_drafts = np.full(viz_samples, c)
        
        # Filter drafts using the deterministic region
        mask = region.contains_y(Y_drafts, E_drafts, C_drafts, alpha)
        survivors = img_drafts[mask]

        # Save artifacts
        np.savez_compressed(out_dir / f"class{c}_drafts_phi.npz", Y=Y_drafts)
        np.savez_compressed(out_dir / f"class{c}_survivors_phi.npz", Y=Y_drafts[mask])
        torch.save(survivors, out_dir / f"class{c}_survivors.pt")
        
        save_image_grid(img_drafts[:64], out_dir / f"class{c}_all_drafts.png", nrow=8)
        if survivors.size(0) > 0:
            save_image_grid(survivors[:64], out_dir / f"class{c}_survivors.png", nrow=8)

        acceptance_rate = float(mask.mean())
        stats = {
            "alpha": alpha,
            "acceptance_rate": acceptance_rate,
            "kept": int(mask.sum()),
            "total_drafts": viz_samples,
            "region_mode": region_mode,
            "threshold": region.tau_handler.t(alpha, c) if tau_handler.mode == 'class' else region.tau_handler.t(alpha)
        }
        save_json(out_dir / f"class{c}_stats.json", stats)
        
        print(f"[region-infer] class {c}: kept {mask.sum()}/{viz_samples} (acceptance rate={acceptance_rate:.3f})")
