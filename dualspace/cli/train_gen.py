from __future__ import annotations
import click
from pathlib import Path
from dualspace.utils.seed import set_seed
from dualspace.utils.config import load_config
from dualspace.utils.env import dump_env_info

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
def train_gen(config: str):
    """Train conditional diffusion generator (CIFAR-10 first)."""
    # TODO: load YAML, build data+models, train loop with EMA/ckpts
    cfg = load_config(config)
    out_dir = Path(cfg.get("out_dir", "outputs/test_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.get("seed", 1337))
    info = dump_env_info(out_dir)
    click.echo(f"[train-gen] Config loaded, env info saved to {out_dir}")
    click.echo(info)
