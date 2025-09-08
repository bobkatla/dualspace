"""Train conditional diffusion generator (images first).

Phase 0: wiring & sanity only. This bootstraps config, seeds, device, and IO.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import yaml
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from dualspace.utils.config import load_config
from dualspace.utils.env import dump_env_info
from dualspace.dataio.cifar10 import build_cifar10_loaders
from dualspace.encoders.condition_encoder import ConditionEncoder
from dualspace.generators.diffusion_image import ImageDiffusion
from dualspace.utils.seed import set_seed
from dualspace.utils.io import save_json
from dualspace.utils.vision import save_image_grid
from dualspace.utils.ema import EMA
from dualspace.train.urc_loss import conformance_loss, urc_acceptance_loss
from dualspace.regions.levelset import LevelSetRegion
from dualspace.encoders.phi_image import PhiImage
from dualspace.densities.mdn import CondMDN
from dualspace.regions.conformal import TauAlpha
from dualspace.utils.quantile import PerClassFIFOQuantiles
import numpy as np


def check(config_path: str):
    # 1) Load config
    cfg = load_config(config_path)

    # 2) Seed
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    # 3) Resolve out_dir
    out_dir = Path(cfg.get("out_dir", "outputs/test_run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) Save artifacts (no training)
    save_json(out_dir / "run_config.json", cfg)
    save_json(out_dir / "run_info.json", {
        "config_path": config_path,
        "seed": seed,
        "out_dir": str(out_dir),
        "timestamp": datetime.now().isoformat() + "Z",
    })
    info = dump_env_info(out_dir)

    # 5) Console summary
    print("[phase0] Config loaded ✓")
    print(f"[phase0] Seed={seed}")
    print(f"[phase0] Out dir: {out_dir}")
    print("[phase0] Device summary:")
    for k, v in info.items():
        print(f"  - {k}: {v}")
    print("[phase0] Done. (No training executed.)")


def _build_phi_torch_tensors(phi: PhiImage, device: torch.device):
    """Extract scaler/PCA params into torch tensors for fast in-graph transform."""
    scaler_mean = torch.from_numpy(phi.scaler.mean_.astype('float32')).to(device)
    scaler_scale = torch.from_numpy(phi.scaler.scale_.astype('float32')).to(device)
    pca_mean = torch.from_numpy(phi.pca.mean_.astype('float32')).to(device)
    pca_components = torch.from_numpy(phi.pca.components_.astype('float32')).to(device)  # (d_out, d_in)
    return scaler_mean, scaler_scale, pca_mean, pca_components


def _phi_project_torch(phi: PhiImage, x: torch.Tensor,
                       scaler_mean: torch.Tensor, scaler_scale: torch.Tensor,
                       pca_mean: torch.Tensor, pca_components: torch.Tensor) -> torch.Tensor:
    """Compute φ(x) → feats (backbone), then Standardize+PCA with torch ops."""
    with torch.no_grad():
        feats = phi(x).detach()  # (B, d_in), backbone is frozen
    X_std = (feats - scaler_mean) / scaler_scale
    Xc = X_std - pca_mean
    Z = torch.matmul(Xc, pca_components.t())  # (B, d_out)
    return Z


def train_generator(config: str):
    """Train conditional diffusion generator (CIFAR-10)."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 1337)))
    out_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    # URC setup
    region = None
    urc_cfg = cfg.get("urc", {})
    use_urc = urc_cfg.get("enabled", False)

    # data
    loaders, splits, _ = build_cifar10_loaders(cfg)

    # models
    num_classes = int(cfg.get("num_classes", 10))
    d_c = int(cfg.get("d_c", 64))
    gcfg = cfg.get("g", {})
    g = ConditionEncoder(num_classes=num_classes, d_c=d_c,
                        hidden=int(gcfg.get("hidden", 256)),
                        depth=int(gcfg.get("depth", 2)),
                        dropout=float(gcfg.get("dropout", 0.0)),
                        norm=gcfg.get("norm", None),
                        mode=gcfg.get("mode", "linear_orth"),
                        orth_reg=float(gcfg.get("orth_reg", 0.0)))
    T = int(cfg.get("T", 200))
    p_uncond = float(cfg.get("p_uncond", 0.05))
    diffusion = ImageDiffusion(in_ch=3, d_c=d_c, T=T, p_uncond=p_uncond)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g.to(device)
    diffusion.to(device)

    # Persistent EMA model
    ema = EMA(diffusion, decay=float(cfg.get("ema_decay", 0.999)))
    diffusion_ema = ImageDiffusion(in_ch=3, d_c=d_c, T=T)
    diffusion_ema.load_state_dict(diffusion.state_dict())
    diffusion_ema.to(device).eval()
    update_ema_every = int(urc_cfg.get("update_ema_every", 50))
    sample_from_ema = bool(urc_cfg.get("sample_from_ema", True))

    # Online phi + density head for URC
    phi = None
    mdn = None
    mdn_opt = None
    quantiles = None
    alpha_urc = float(urc_cfg.get("alpha", 0.9))
    urc_mode = str(urc_cfg.get("mode", "softplus"))
    urc_weight = float(urc_cfg.get("weight", 0.1))
    weight_sep = float(urc_cfg.get("weight_sep", 0.0))
    sep_margin = float(urc_cfg.get("sep_margin", 1.0))
    warmup_min_per_class = int(urc_cfg.get("warmup_min_per_class", 128))
    quant_win = int(urc_cfg.get("quantile_window", 4096))

    phi_tensors = None
    if use_urc:
        print("[train] URC-online enabled. Loading φ pipeline and initializing density head...")
        phi = PhiImage().to(device).eval()
        phi.load(out_dir / "phi")
        phi_tensors = _build_phi_torch_tensors(phi, device)
        mdn = CondMDN(d_in=d_c, d_out=phi.d_out, n_comp=int(cfg.get("mdn_components", 6)), hidden=int(cfg.get("mdn_hidden", 256))).to(device)
        mdn_opt = AdamW(mdn.parameters(), lr=float(cfg.get("lr_mdn", 1e-3)))
        quantiles = PerClassFIFOQuantiles(num_classes=num_classes, window_size=quant_win)
        print("[train] URC components ready.")

    # optim
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    opt = AdamW(list(g.parameters()) + list(diffusion.parameters()), lr=lr, weight_decay=wd)
    scaler = GradScaler("cuda", enabled=bool(cfg.get("mixed_precision", True)) and device.type=="cuda")
    steps = int(cfg.get("train_steps", 10000))
    log_every = int(cfg.get("log_every", 100))
    sample_every = int(cfg.get("sample_every", 2000))
    save_every = int(cfg.get("save_every_steps", 2000))
    guidance = float(cfg.get("guidance_scale", 2.0))

    # class one-hots for sampling grid (10 classes, 8 per class → 80 samples)
    eye = torch.eye(num_classes, device=device)
    step = 0
    losses = []
    while step < steps:
        for x, y in loaders["train"]:
            step += 1
            x = x.to(device)
            y = y.to(device)
            c = eye[y]
            e = g(c)
            null_e = g.get_null(batch=x.size(0))
            
            # Periodically update persistent EMA
            if step % update_ema_every == 0:
                ema.copy_to(diffusion_ema)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=scaler.is_enabled()):
                loss_diff = diffusion.loss(x, e, null_e)
                loss_ortho = g.orthogonality_loss()
                total_loss = loss_diff + loss_ortho

            # --- URC-online updates ---
            if use_urc:
                # 1) Update density head on real pairs (y_real, e.detach())
                y_real = _phi_project_torch(phi, x, *phi_tensors)
                mdn.train()
                mdn_opt.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=scaler.is_enabled()):
                    logp_real = mdn.log_prob(y_real, e.detach())
                    nll_real = -(logp_real.mean())
                scaler.scale(nll_real).backward()
                scaler.unscale_(mdn_opt)
                torch.nn.utils.clip_grad_norm_(mdn.parameters(), 1.0)
                mdn_opt.step()

                # 2) Update plug-in quantiles with real scores (no grad)
                with torch.no_grad():
                    scores_real = (-logp_real).detach().cpu().numpy()
                    quantiles.update_many(y.detach().cpu().numpy(), scores_real)

                # 3) URC acceptance and separation under autocast
                with autocast("cuda", enabled=scaler.is_enabled()):
                    # Sample once, from EMA if configured
                    with torch.no_grad():
                        if sample_from_ema:
                            x_src = diffusion_ema.sample(e, K=e.size(0), guidance_scale=guidance, shape=(3,32,32), null_e=null_e)
                        else:
                            x_src = diffusion.sample(e, K=e.size(0), guidance_scale=guidance, shape=(3,32,32), null_e=null_e)
                    y_gen = _phi_project_torch(phi, x_src, *phi_tensors)
                    logp_gen = mdn.log_prob(y_gen, e.detach())
                    scores_gen = -logp_gen

                    # thresholds per sample from FIFO (no grad)
                    with torch.no_grad():
                        tau_np = quantiles.get_tau_batch(y.detach().cpu().numpy(), alpha=alpha_urc)
                    tau = torch.from_numpy(tau_np).to(device).float()
                    mask_ready = torch.isfinite(tau)

                    if mask_ready.any():
                        scores_sel = scores_gen[mask_ready]
                        tau_sel = tau[mask_ready]
                        if urc_mode == "softplus":
                            loss_urc = torch.nn.functional.softplus(tau_sel - scores_sel).mean()
                        else:
                            loss_urc = torch.relu(scores_sel - tau_sel + float(urc_cfg.get("margin", 0.0))).mean()
                    else:
                        loss_urc = torch.tensor(0.0, device=device)

                    # Optional separation loss
                    loss_sep = torch.tensor(0.0, device=device)
                    if weight_sep > 0.0 and mask_ready.any():
                        # Negative labels via random permutation
                        y_neg = y[torch.randperm(y.size(0))]
                        e_neg = g(eye[y_neg]).detach()
                        # Option: allow gradients through y_gen to push generator away from wrong regions
                        s_cross = -mdn.log_prob(y_gen, e_neg)
                        with torch.no_grad():
                            tau_cross_np = quantiles.get_tau_batch(y_neg.detach().cpu().numpy(), alpha=alpha_urc)
                        tau_cross = torch.from_numpy(tau_cross_np).to(device).float()
                        tau_cross = tau_cross
                        loss_sep = torch.nn.functional.softplus(sep_margin - (s_cross - tau_cross)).mean()

                    total_loss = total_loss + urc_weight * loss_urc + weight_sep * loss_sep

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(diffusion.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(diffusion)

            if step % log_every == 0:
                msg = f"[train] step {step}/{steps} loss={float(total_loss.detach().cpu().item()):.4f}"
                if use_urc:
                    msg += f" | nll_real={float(nll_real.detach().cpu().item()):.4f}"
                    if 'scores_gen' in locals():
                        with torch.no_grad():
                            mean_s = float(scores_gen.mean().item())
                        msg += f" | mean_s_gen={mean_s:.3f}"
                    if 'tau' in locals():
                        with torch.no_grad():
                            mean_tau = float(torch.nanmean(tau).item())
                        msg += f" | mean_tau={mean_tau:.3f}"
                    if 'scores_gen' in locals() and 'tau' in locals():
                        with torch.no_grad():
                            hit = (scores_gen <= tau).float().mean().item()
                        msg += f" | hit@{alpha_urc:.2f}={hit:.3f}"
                print(msg)
                losses.append(float(total_loss.detach().cpu().item()))

            if step % sample_every == 0:
                with torch.no_grad():
                    e_grid = g(eye).repeat_interleave(8, dim=0)
                    null1 = g.get_null(batch=e_grid.size(0))
                    x_samp = diffusion_ema.sample(e_grid, K=e_grid.size(0), guidance_scale=guidance, shape=(3,32,32), null_e=null1)
                    img = (x_samp.clamp(-1,1) + 1.0) / 2.0
                    mx = x_samp.abs().max().item()
                    nan_frac = torch.isnan(x_samp).float().mean().item()
                    print(f"[sample] max|x|={mx:.2f}, NaN%={nan_frac*100:.3f}")
                    save_image_grid(img, out_dir / "samples" / f"step_{step:06d}.png", nrow=8)

            if step % save_every == 0:
                ck = out_dir / "ckpts" / f"step_{step:06d}.pt"
                torch.save({
                    "g": g.state_dict(),
                    "diffusion": diffusion.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "mdn": (mdn.state_dict() if mdn else None),
                    "quantiles": (quantiles.state_dict() if quantiles else None),
                }, ck)

            if step >= steps:
                break

    save_json(out_dir / "loss_curve.json", {"loss": losses})
    print("[train] done")
