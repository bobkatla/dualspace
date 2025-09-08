from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np

# A type hint for the region object to avoid circular imports
if False:
    from dualspace.regions.levelset import LevelSetRegion


def conformance_loss(
    x_gen: torch.Tensor,
    e_gen: torch.Tensor,
    c_gen: torch.Tensor,
    region: "LevelSetRegion",
    alpha: float,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Deprecated in URC-online: kept for backward compatibility when using CP thresholds.
    """
    with torch.no_grad():
        phi_gen_np = region.phi.transform(region.phi(x_gen).cpu().numpy())
    phi_gen = torch.from_numpy(phi_gen_np).to(x_gen.device)
    logp_gen = region.mdn.log_prob(phi_gen, e_gen)
    if region.tau_handler.mode == "class":
        thresholds_np = np.array([region.tau_handler.t(alpha, c) for c in c_gen.cpu().numpy()])
        thresholds = torch.from_numpy(thresholds_np).to(logp_gen.device).float()
    elif region.tau_handler.mode == "global":
        thr = region.tau_handler.t(alpha)
        thresholds = torch.full_like(logp_gen, thr)
    else:
        raise NotImplementedError("Conformance loss requires 'class' or 'global' calibration mode.")
    scores = -logp_gen
    calibrated_score_thresholds = -thresholds
    loss = F.relu(scores - calibrated_score_thresholds + margin)
    return loss.mean()


def urc_acceptance_loss(
    x_gen: torch.Tensor,
    e_gen: torch.Tensor,
    c_gen: torch.Tensor,
    phi,   # PhiImage
    mdn,   # CondMDN
    quantiles,  # PerClassFIFOQuantiles
    alpha: float,
    mode: str = "softplus",
    margin: float = 0.0,
) -> torch.Tensor:
    """
    URC acceptance loss using on-line plug-in thresholds t~_alpha from training scores.

    Args:
        x_gen: Generated images (B, C, H, W)
        e_gen: Condition embeddings (B, d_c)
        c_gen: Integer class labels (B,)
        phi: PhiImage with fitted scaler+PCA
        mdn: CondMDN density head
        quantiles: PerClassFIFOQuantiles (per-class)
        alpha: target coverage level for plug-in thresholds
        mode: 'softplus' (default) or 'hinge'
        margin: hinge margin if mode=='hinge'
    Returns:
        Scalar loss tensor
    """
    device = x_gen.device

    with torch.no_grad():
        y_np = phi.transform(phi(x_gen).cpu().numpy())
    y = torch.from_numpy(y_np).to(device)

    logp = mdn.log_prob(y, e_gen)
    scores = -logp  # s = -log p

    # Get per-sample plug-in thresholds (no grad)
    c_cpu = c_gen.detach().cpu().numpy()
    tau_arr = quantiles.get_tau_batch(c_cpu, alpha=alpha)  # array of s-thresholds
    tau = torch.from_numpy(tau_arr).to(device).float()

    if mode == "softplus":
        # maximize acceptance: softplus(tau - s)
        loss = torch.nn.functional.softplus(tau - scores).mean()
    elif mode == "hinge":
        loss = torch.relu(scores - tau + margin).mean()
    else:
        raise ValueError(f"Invalid URC mode: {mode}")
    return loss
