from __future__ import annotations
import click

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--split", type=click.Choice(["calib"]), default="calib")
def calibrate(config: str, split: str):
    """Conformal calibration to compute tau_alpha per class."""
    click.echo(f"[calibrate] {split} with {config}")
