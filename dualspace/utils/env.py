"""Environment info dump (device, libs, versions)."""
import torch, sys, platform
from pathlib import Path
from .io import save_json


def collect_env_info() -> dict:
    return {
    "python": sys.version,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "num_gpus": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def dump_env_info(out_dir: str | Path):
    out_path = Path(out_dir) / "env_info.json"
    info = collect_env_info()
    save_json(out_path, info)
    return info