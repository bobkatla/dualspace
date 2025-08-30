"""Dump (e=g(c), y=phi(x)) pairs to NPZ for a given split.


- Use real data only for calibration; optionally include generated.
"""
from __future__ import annotations
import argparse


# TODO: load data split, run g and phi in eval mode, save arrays