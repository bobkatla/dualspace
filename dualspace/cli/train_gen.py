from __future__ import annotations
import click

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
def train_gen(config: str):
    """Train conditional diffusion generator (CIFAR-10 first)."""
    # TODO: load YAML, build data+models, train loop with EMA/ckpts
    click.echo(f"[train-gen] using config {config}")
