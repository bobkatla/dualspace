from __future__ import annotations
import click
from dualspace.train.train_generator import train_generator

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
def train_gen(config: str):
    """Train conditional diffusion generator (CIFAR-10 first)."""
    train_generator(config)
