from __future__ import annotations
import click, yaml, torch
from pathlib import Path
from dualspace.dataio.cifar10 import build_cifar10_loaders

@click.command("export-test-images")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--limit", type=int, default=5000, help="Max # test images to export")
def export_test_images(config: str, limit: int):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}"))
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    loaders, _, _ = build_cifar10_loaders(cfg)
    xs = []
    n = 0
    for xb, _ in loaders["test"]:
        # xb is in [-1,1] → convert to [0,1] for FID
        xs.append(((xb.clamp(-1,1) + 1.0) / 2.0))
        n += xb.size(0)
        if n >= limit: break
    xr = torch.cat(xs, dim=0)[:limit]
    torch.save(xr, run_dir / "metrics" / "test_images.pt")
    print(f"[export-test-images] saved {xr.shape} to {run_dir/'metrics'/'test_images.pt'}")
