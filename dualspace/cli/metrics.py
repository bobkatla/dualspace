# Example stubs (adapt to your artifact paths)
import click, numpy as np, torch
from dualspace.metrics.informativeness import logdet_cov_phi, farthest_point_sampling
from dualspace.metrics.mmd import mmd_rbf
from dualspace.metrics.ot_sinkhorn import sinkhorn_cost
from dualspace.metrics.fid import fid_from_tensors
from pathlib import Path

PREFERRED_KEYS = ("Y", "y", "X", "features", "emb", "phi", "arr_0")

def _load_array(maybe_npz_or_npy: str) -> np.ndarray:
    p = Path(maybe_npz_or_npy)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")

    if p.suffix == ".npy":
        arr = np.load(p)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{p} did not load to ndarray")
        return arr

    if p.suffix == ".npz":
        data = np.load(p)
        # np.load on .npz returns an NpzFile dict-like
        if hasattr(data, "files"):
            for k in PREFERRED_KEYS:
                if k in data.files:
                    return data[k]
            # Fallback: if there is exactly one array, return it
            if len(data.files) == 1:
                return data[data.files[0]]
            raise KeyError(
                f"{p} is .npz but missing any of keys {PREFERRED_KEYS}; "
                f"found keys={list(data.files)}"
            )
        # rare: plain ndarray stored as .npz
        if isinstance(data, np.ndarray):
            return data

    raise ValueError(f"Unsupported file type for {p} (need .npy or .npz)")

@click.group("metrics")
def metrics():
    pass

@metrics.command("informativeness")
@click.option("--phi-npz", type=click.Path(exists=True), required=True)
def cli_info(phi_npz):
    Y = _load_array(phi_npz)
    print("logdet_cov_phi:", logdet_cov_phi(Y))

@metrics.command("mmd")
@click.option("--x-npz", type=click.Path(exists=True), required=True)
@click.option("--y-npz", type=click.Path(exists=True), required=True)
def cli_mmd(x_npz, y_npz):
    X = _load_array(x_npz)
    Y = _load_array(y_npz)
    print("MMD^2 (RBF mix):", mmd_rbf(X, Y))

@metrics.command("sinkhorn")
@click.option("--x-npz", type=click.Path(exists=True), required=True)
@click.option("--y-npz", type=click.Path(exists=True), required=True)
@click.option("--reg", type=float, default=0.1)
def cli_sink(x_npz, y_npz, reg):
    X = _load_array(x_npz)
    Y = _load_array(y_npz)
    print("Sinkhorn cost:", sinkhorn_cost(X, Y, reg=reg))

@metrics.command("fid")
@click.option("--real-pt", type=click.Path(exists=True), required=True)
@click.option("--fake-pt", type=click.Path(exists=True), required=True)
def cli_fid(real_pt, fake_pt):
    xr = torch.load(real_pt)  # expect (N,3,H,W) in [0,1]
    xf = torch.load(fake_pt)
    print("FID:", fid_from_tensors(xr, xf))
