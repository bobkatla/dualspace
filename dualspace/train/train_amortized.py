"""Train the MDN on dumped pairs.


- Minimize negative log-likelihood on (y,e).
- Save best checkpoint and training curves (JSON/NPZ).
"""
from __future__ import annotations
import argparse


# TODO