# Example stubs (adapt to your artifact paths)
import click, numpy as np, torch
from dualspace.metrics.informativeness import logdet_cov_phi, farthest_point_sampling
from dualspace.metrics.mmd import mmd_rbf
from dualspace.metrics.ot_sinkhorn import sinkhorn_phi
from dualspace.metrics.fid import fid_from_tensors

@click.group("metrics")
def metrics():
    pass

@metrics.command("informativeness")
@click.option("--phi-npz", type=click.Path(exists=True), required=True)
def cli_info(phi_npz):
    Y = np.load(phi_npz)["Y"]  # expect {'Y': survivors in φ-space}
    print("logdet_cov_phi:", logdet_cov_phi(Y))

@metrics.command("mmd")
@click.option("--x-npz", type=click.Path(exists=True), required=True)
@click.option("--y-npz", type=click.Path(exists=True), required=True)
def cli_mmd(x_npz, y_npz):
    X = np.load(x_npz)["Y"]; Y = np.load(y_npz)["Y"]
    print("MMD^2 (RBF mix):", mmd_rbf(X, Y))

@metrics.command("sinkhorn")
@click.option("--x-npz", type=click.Path(exists=True), required=True)
@click.option("--y-npz", type=click.Path(exists=True), required=True)
@click.option("--reg", type=float, default=0.05)
def cli_sink(x_npz, y_npz, reg):
    X = np.load(x_npz)["Y"]; Y = np.load(y_npz)["Y"]
    print("Sinkhorn cost:", sinkhorn_phi(X, Y, reg=reg))

@metrics.command("fid")
@click.option("--real-pt", type=click.Path(exists=True), required=True)
@click.option("--fake-pt", type=click.Path(exists=True), required=True)
def cli_fid(real_pt, fake_pt):
    xr = torch.load(real_pt)  # expect (N,3,H,W) in [0,1]
    xf = torch.load(fake_pt)
    print("FID:", fid_from_tensors(xr, xf))
