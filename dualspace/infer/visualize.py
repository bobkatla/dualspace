"""End-to-end inference: given c, sample K drafts, filter by tau_alpha, pick reps, compute metrics.


CLI
---
python -m dualspace.infer.region_infer --config configs/cifar10.yaml --split test
"""
from __future__ import annotations
import argparse


# TODO