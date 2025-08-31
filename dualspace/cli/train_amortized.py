from __future__ import annotations
import click

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
def train_amortized(config: str):
    """Train MDN on dumped pairs."""
    click.echo(f"[train-amortized] {config}")
