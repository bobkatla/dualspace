"""Utilities for deterministic seeding across numpy, torch, and CUDA.

Functions
---------
set_seed(seed: int) -> None
Sets seeds and configures PyTorch for deterministic behavior where feasible.
"""
from __future__ import annotations
import os, random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False