from __future__ import annotations
import click
from pathlib import Path
from dualspace.utils.seed import set_seed
from dualspace.utils.config import load_config
from dualspace.utils.env import dump_env_info
from dualspace.train.train_generator import train

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
def train_gen(config: str):
    """Train conditional diffusion generator (CIFAR-10 first)."""
    # TODO: load YAML, build data+models, train loop with EMA/ckpts
    train(config)
