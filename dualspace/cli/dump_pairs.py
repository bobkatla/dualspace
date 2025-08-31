from __future__ import annotations
import click

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--split", type=click.Choice(["train","calib","test"]), default="train")
def dump_pairs(config: str, split: str):
    """Dump (e=g(c), y=phi(x)) pairs to NPZ for the given split."""
    click.echo(f"[dump-pairs] {split} with {config}")
