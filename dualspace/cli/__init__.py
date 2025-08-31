"""
Command Line Interface for DualSpace.

This module provides the main entry point for the dualspace command line tool.
"""

import click
from dualspace import __version__
from dualspace.cli.calibrate import calibrate
from dualspace.cli.dump_pairs import dump_pairs
from dualspace.cli.infer import infer
from dualspace.cli.train_amortized import train_amortized
from dualspace.cli.train_gen import train_gen
from dualspace.cli.visualize import visualize

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
