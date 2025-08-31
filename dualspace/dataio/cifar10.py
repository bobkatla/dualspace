from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from dualspace.utils.splits import stratified_split, class_hist, Splits

# Transform: to [-1, 1]
_cifar_transform = T.Compose([
    T.ToTensor(), # [0,1]
    T.Lambda(lambda x: x * 2.0 - 1.0), # [-1,1]
])

def load_cifar10(root: str | Path = "data/cifar10", download: bool = True) -> Tuple[torch.utils.data.Dataset, torch.Tensor]:
    ds = CIFAR10(root=str(root), train=True, transform=_cifar_transform, download=download)
    labels = torch.tensor(ds.targets, dtype=torch.long)
    return ds, labels

def build_cifar10_loaders(cfg: Dict) -> Tuple[Dict[str, DataLoader], Splits, Dict[str, Dict[int,int]]]:
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 8))
    root = cfg.get("data_root", "data/cifar10")
    seed = int(cfg.get("seed", 1337))

    ds_full, labels = load_cifar10(root)
    splits = stratified_split(labels.tolist(), (cfg["split"]["train"], cfg["split"]["calib"], cfg["split"]["test"]), seed)
    loaders = {
        "train": DataLoader(Subset(ds_full, splits.train), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "calib": DataLoader(Subset(ds_full, splits.calib), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(Subset(ds_full, splits.test), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    hists = {
        "train": class_hist(labels, splits.train),
        "calib": class_hist(labels, splits.calib),
        "test": class_hist(labels, splits.test),
    }
    return loaders, splits, hists