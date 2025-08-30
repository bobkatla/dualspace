"""CIFAR-10 datamodule-style loader with train/calib/test splits.


- x in [-1, 1], shape (B,3,32,32)
- c as one-hot (B,10)
"""
from __future__ import annotations
from typing import Tuple


# TODO: implement dataset + split function returning torch DataLoaders