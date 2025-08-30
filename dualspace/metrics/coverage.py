"""Coverage computation vs target alphas.


Given held-out (x,c), compute indicator[x in R_alpha(c)] and average per class and overall.
"""
from __future__ import annotations
from typing import Dict
import torch


# TODO: coverage(y, e, mdn, tau_alpha_per_class) -> Dict