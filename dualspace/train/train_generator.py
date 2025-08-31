"""Train conditional diffusion generator (images first).

Phase 0: wiring & sanity only. This bootstraps config, seeds, device, and IO.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime

from dualspace.utils.seed import set_seed
from dualspace.utils.io import save_json
from dualspace.utils.config import load_config
from dualspace.utils.env import dump_env_info


def train(config_path: str):
    # 1) Load config
    cfg = load_config(config_path)

    # 2) Seed
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    # 3) Resolve out_dir
    out_dir = Path(cfg.get("out_dir", "outputs/test_run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) Save artifacts (no training)
    save_json(out_dir / "run_config.json", cfg)
    save_json(out_dir / "run_info.json", {
        "config_path": config_path,
        "seed": seed,
        "out_dir": str(out_dir),
        "timestamp": datetime.now().isoformat() + "Z",
    })
    info = dump_env_info(out_dir)

    # 5) Console summary
    print("[phase0] Config loaded ✓")
    print(f"[phase0] Seed={seed}")
    print(f"[phase0] Out dir: {out_dir}")
    print("[phase0] Device summary:")
    for k, v in info.items():
        print(f"  - {k}: {v}")
    print("[phase0] Done. (No training executed.)")
