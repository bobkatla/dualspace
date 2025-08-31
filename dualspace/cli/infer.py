from __future__ import annotations
import click

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--split", type=click.Choice(["test","calib","train"]), default="test")
def infer(config: str, split: str):
    """Sample K drafts, filter by tau, pick reps, compute metrics & save outputs."""
    click.echo(f"[infer] {split} with {config}")
