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


def train_generator(config: str):
    """Train conditional diffusion generator (CIFAR-10)."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 1337)))
    out_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

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

    # optim
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    opt = AdamW(list(g.parameters()) + list(diffusion.parameters()), lr=lr, weight_decay=wd)
    scaler = GradScaler("cuda", enabled=bool(cfg.get("mixed_precision", True)) and device.type=="cuda")
    ema = EMA(diffusion, decay=float(cfg.get("ema_decay", 0.999)))
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
            
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=scaler.is_enabled()):
                loss = diffusion.loss(x, e, null_e)
                loss = loss + g.orthogonality_loss()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(diffusion.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(diffusion)

            if step % log_every == 0:
                losses.append(float(loss.detach().cpu().item()))
                print(f"[train] step {step}/{steps} loss={losses[-1]:.4f}")

            if step % sample_every == 0:
                diffusion_ema = ImageDiffusion(in_ch=3, d_c=d_c, T=T)
                diffusion_ema.load_state_dict(diffusion.state_dict())
                ema.copy_to(diffusion_ema)
                diffusion_ema.to(device).eval()
                with torch.no_grad():
                    # sample 8 per class (80 total)
                    e_grid = g(eye).repeat_interleave(8, dim=0)
                    null1 = g.get_null(batch=e_grid.size(0))
                    x_samp = diffusion_ema.sample(e_grid, K=e_grid.size(0), guidance_scale=guidance, shape=(3,32,32), null_e=null1)
                    img = (x_samp.clamp(-1,1) + 1.0) / 2.0
                    mx = x_samp.abs().max().item()
                    nan_frac = torch.isnan(x_samp).float().mean().item()
                    print(f"[sample] max|x|={mx:.2f}, NaN%={nan_frac*100:.3f}")
                    save_image_grid(img, out_dir / "samples" / f"step_{step:06d}.png", nrow=8)
                del diffusion_ema

            if step % save_every == 0:
                ck = out_dir / "ckpts" / f"step_{step:06d}.pt"
                torch.save({
                    "g": g.state_dict(),
                    "diffusion": diffusion.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                }, ck)

            if step >= steps:
                break

    save_json(out_dir / "loss_curve.json", {"loss": losses})
    print("[train] done")
