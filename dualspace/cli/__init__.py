"""
Command Line Interface for DualSpace.

This module provides the main entry point for the dualspace command line tool.
"""

import click
from dualspace import __version__
from dualspace.cli.calibrate import calibrate
from dualspace.cli.dump_pairs import dump_pairs
from dualspace.cli.infer import infer
from dualspace.cli.metrics import metrics
from dualspace.cli.train_amortized import train_amortized
from dualspace.cli.train_gen import train_gen
from dualspace.cli.visualize import visualize
from dualspace.cli.prep_data import prep_data
from dualspace.cli.diag_g import diag_g
from dualspace.cli.phi_image import fit_phi
from dualspace.infer.region_infer import region_infer
from dualspace.cli.eval_coverage import coverage_curve
from dualspace.cli.eval_pareto import pareto

@click.group()
@click.version_option(version=__version__, prog_name="dualspace")
def main():
    """DualSpace - A framework for building and training conditional generative models."""
    pass


@main.command()
def info():
    """Display information about the DualSpace package."""
    click.echo(f"DualSpace version {__version__}")
    click.echo("Framework for building and training conditional generative models.")


main.add_command(train_gen, name="train-gen")
main.add_command(dump_pairs, name="dump-pairs")
main.add_command(train_amortized, name="train-amortized")
main.add_command(calibrate, name="calibrate")
main.add_command(infer, name="infer")
main.add_command(visualize, name="visualize")
main.add_command(prep_data, name="prep-data")
main.add_command(diag_g, name="diag-g")
main.add_command(fit_phi, name="fit-phi")
main.add_command(region_infer, name="region-infer")
main.add_command(coverage_curve, name="coverage-curve")
main.add_command(pareto, name="pareto")
main.add_command(metrics, name="metrics")