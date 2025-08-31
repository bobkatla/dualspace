from __future__ import annotations
import click

@click.command()
@click.option("--run", type=str, required=False, help="Run name/out_dir to visualize")
def visualize(run: str | None):
    """Plot coverage curves, Pareto fronts, and sample grids."""
    click.echo(f"[visualize] run={run}")
