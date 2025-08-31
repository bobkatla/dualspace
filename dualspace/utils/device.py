"""Device helpers and summaries."""
from __future__ import annotations
import torch

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device_summary() -> dict:
    d = {"torch_device": str(get_device()), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        d["device_name"] = torch.cuda.get_device_name(0)
        d["capability"] = torch.cuda.get_device_capability(0)
        d["memory_total"] = props.total_memory
    return d
