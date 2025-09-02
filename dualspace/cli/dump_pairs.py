from __future__ import annotations
import click
from dualspace.train.dump_pairs import get_dump_pairs

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
def dump_pairs(config: str):
    get_dump_pairs(config)
