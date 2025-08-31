from __future__ import annotations
import json
from pathlib import Path
import click
import torch
import yaml
from dualspace.encoders.condition_encoder import ConditionEncoder
from dualspace.utils.plotting import save_cosine_heatmap

@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
def diag_g(config: str):
    """Diagnose g(c): save per-class centroids and cosine-sim heatmap."""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg.get("out_dir", f"outputs/{cfg.get('run_name','run')}")) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = int(cfg.get("num_classes", 10))
    d_c = int(cfg.get("d_c", 64))
    enc_cfg = cfg.get("g", {})

    g = ConditionEncoder(num_classes=num_classes, d_c=d_c,
                        hidden=int(enc_cfg.get("hidden", 256)),
                        depth=int(enc_cfg.get("depth", 2)),
                        dropout=float(enc_cfg.get("dropout", 0.0)),
                        norm=enc_cfg.get("norm", None)).eval()
    # Identity one-hot → embeddings
    eye = torch.eye(num_classes)
    with torch.no_grad():
        E = g(eye) # (C, d_c)
    # Save centroids
    centroids_path = out_dir / "g_centroids.pt"
    torch.save(E, centroids_path)

    # Cosine similarity
    E_norm = torch.nn.functional.normalize(E, dim=-1)
    cos = E_norm @ E_norm.t() # (C,C)
    save_cosine_heatmap(cos.cpu().numpy(), out_dir / "g_cosine_heatmap.png")

    # Simple JSON summary
    diag = {
        "d_c": d_c,
        "num_classes": num_classes,
        "norm": enc_cfg.get("norm", None),
        "min_cos": float(cos.min().item()),
        "max_cos": float(cos.max().item()),
        "mean_cos_offdiag": float((cos.sum().item() - num_classes) / (num_classes*(num_classes-1))),
        "centroids_path": str(centroids_path),
    }
    with open(out_dir / "g_diag.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    click.echo(f"[diag-g] wrote {out_dir}")   