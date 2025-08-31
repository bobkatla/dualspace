from __future__ import annotations
import click
from pathlib import Path
from dualspace.utils.config import load_config
from dualspace.utils.io import save_json
from dualspace.dataio.cifar10 import build_cifar10_loaders, compute_channel_stats
from dualspace.utils.vision import save_image_grid
from dualspace.utils.splits import Splits


@click.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--preview", is_flag=True, help="Save a 8x8 preview grid for CIFAR-10")
# @click.option("--har-arrays", type=click.Path(exists=True), required=False, help="Path to npz with X:(N,T,D), y:(N,)")
def prep_data(config, preview):
    """Build stratified splits and save stats/histograms."""
    cfg = load_config(config)
    out_dir = Path(cfg.get("out_dir", "outputs/test_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = cfg.get("dataset", "ERR")
    (out_dir / "stats").mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    if dataset == "cifar10":
        loaders, splits, hists = build_cifar10_loaders(cfg)
        # Save splits JSON
        Splits(splits.train, splits.calib, splits.test).save(out_dir, "cifar10")
        # Save class hist
        save_json(out_dir / "stats" / "cifar10_class_hist.json", hists)
        # Save channel stats (over train only)
        stats = compute_channel_stats(loaders["train"])
        save_json(out_dir / "stats" / "cifar10_channel_stats.json", stats)
        # Optional preview grid
        if preview:
            batch = next(iter(loaders["train"]))[0][:64]
            # de-normalize [-1,1] -> [0,1]
            img = (batch.clamp(-1,1) + 1.0) / 2.0
            save_image_grid(img, out_dir / "stats" / "cifar10_preview.png")
        click.echo(f"[prep-data] CIFAR-10 splits & stats saved under {out_dir}")
    elif dataset == "ERR":
        raise ValueError("config file has not dataset specified")
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")
