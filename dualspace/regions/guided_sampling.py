"""Helpers to sample K drafts and filter by region predicate log p >= tau.


Also expose a simple farthest-point sampling in phi-space to pick diverse reps.
"""
from __future__ import annotations
import torch
from typing import Tuple


# TODO: implement filter_and_select(phi, mdn, e, tau, X, K_select)