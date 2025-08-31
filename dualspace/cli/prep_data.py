from __future__ import annotations
import click
from pathlib import Path
from dualspace.utils.config import load_config
from dualspace.utils.io import save_json
from dualspace.dataio.cifar10 import build_cifar10_loaders


@click.command()
@click.option("--config", type=click.Path(exists=True), required=True)
# @click.option("--har-arrays", type=click.Path(exists=True), required=False, help="Path to npz with X:(N,T,D), y:(N,)")
def prep_data(config):
    """Build stratified splits and save stats/histograms."""
    cfg = load_config(config)
    out_dir = Path(cfg.get("out_dir", "outputs/test_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = cfg.get("dataset", "ERR")

    if dataset == "cifar10":
        loaders, splits, hists = build_cifar10_loaders(cfg)
        splits.save(out_dir, "cifar10")
        save_json(out_dir / "stats" / "cifar10_class_hist.json", hists)
        click.echo(f"[prep-data] CIFAR-10 splits & hist saved under {out_dir}")
    elif dataset == "ERR":
        raise ValueError("config file has not dataset specified")
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")
