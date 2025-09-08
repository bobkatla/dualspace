from __future__ import annotations
import torch
import numpy as np

from dualspace.encoders.phi_image import PhiImage
from dualspace.densities.mdn import CondMDN
from dualspace.regions.conformal import TauAlpha


class LevelSetRegion:
    """
    Represents a deterministic, calibrated region R_alpha(c) defined by the level sets
    of a conditional density model in phi-space.
    """
    def __init__(self, phi: PhiImage, mdn: CondMDN, tau_handler: TauAlpha):
        self.phi = phi
        self.mdn = mdn
        self.tau_handler = tau_handler
        # Assume mdn parameters can tell us the device
        self.device = next(mdn.parameters()).device

    @torch.no_grad()
    def contains_y(self, y: np.ndarray, e: np.ndarray, c: np.ndarray | None, alpha: float) -> np.ndarray:
        """
        Checks if feature-space points `y` are in the region for given embeddings `e`
        and conditions `c` at coverage level `alpha`.

        Args:
            y: (N, d_phi) projected features.
            e: (N, d_c) condition embeddings.
            c: (N,) original conditions (e.g., class labels). Required for 'class' mode.
            alpha: Target coverage level.

        Returns:
            (N,) boolean mask, True for points inside the region.
        """
        self.mdn.eval()
        
        y_torch = torch.from_numpy(y).to(self.device)
        e_torch = torch.from_numpy(e).to(self.device)
        
        logp = self.mdn.log_prob(y_torch, e_torch).cpu().numpy()
        
        if self.tau_handler.mode == 'global':
            thr = self.tau_handler.t(alpha)
            return logp >= thr
        elif self.tau_handler.mode == 'class':
            if c is None:
                raise ValueError("`c` (class labels) must be provided for 'class' mode.")
            # Vectorized lookup for thresholds
            thresholds = np.array([self.tau_handler.t(alpha, cond) for cond in c])
            return logp >= thresholds
        else: # knn
             raise NotImplementedError("KNN mode for TauAlpha is not yet implemented.")

    def size_proxy(self, y_samples: np.ndarray, e_samples: np.ndarray, c_samples: np.ndarray, alpha: float) -> float:
        """
        Computes an informativeness proxy (log-determinant of covariance) for the region,
        estimated from a set of samples.

        Args:
            y_samples: A batch of samples in phi-space.
            e_samples: Corresponding condition embeddings.
            c_samples: Corresponding original conditions.
            alpha: Target coverage level.

        Returns:
            Scalar log-determinant of the covariance of the samples that fall within the region.
        """
        mask = self.contains_y(y_samples, e_samples, c_samples, alpha)
        y_contained = y_samples[mask]
        
        if y_contained.shape[0] < y_contained.shape[1]: # Need more points than dims
            return -np.inf
            
        # Add a ridge for numerical stability
        cov = np.cov(y_contained, rowvar=False) + 1e-6 * np.eye(y_contained.shape[1])
        _sign, logdet = np.linalg.slogdet(cov)
        
        return logdet if _sign > 0 else -np.inf
