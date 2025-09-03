"""Train MDN on (e,y) pairs."""
from __future__ import annotations
import click
from dualspace.train.train_amortized import train_amor


@click.command("train-amortized")
@click.option("--config", type=click.Path(exists=True), required=True)
def train_amortized(config: str):
    train_amor(config)
