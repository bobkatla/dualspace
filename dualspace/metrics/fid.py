import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg

@torch.no_grad()
def _inception_pool3_features(x_bchw, model, device):
    # x in [0,1] or [-1,1]? For CIFAR-10, if in [-1,1], map to [0,1]
    if x_bchw.min() < 0:
        x_bchw = (x_bchw + 1.0) * 0.5
    # Resize to 299 for InceptionV3
    x = F.interpolate(x_bchw, size=(299, 299), mode="bilinear", align_corners=False)
    x = x.to(device)
    return model(x)  # returns pool3 activations (N, 2048)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionPool3(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT  # == IMAGENET1K_V1
        # IMPORTANT: aux_logits must be True when using pretrained weights
        inc = inception_v3(weights=weights, aux_logits=True)
        inc.eval()
        self.device = device
        self.inc = inc.to(device)
        self.preprocess = weights.transforms()  # resize + normalize

        # Hook on avgpool to capture (N, 2048, 1, 1)
        self._feats = None
        def hook(_, __, output):
            self._feats = output.squeeze(-1).squeeze(-1)  # (N, 2048)
        self.inc.avgpool.register_forward_hook(hook)

    @torch.no_grad()
    def features(self, x_0_1):  # x expected in [0,1]
        # Preprocess to 299 + imagenet normalization
        # weights.transforms() expects PIL/float tensor in [0,1]; it handles resize
        x = F.interpolate(x_0_1, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.preprocess(x)  # channel-wise normalize
        x = x.to(self.device)
        self._feats = None
        _ = self.inc(x)  # triggers hook
        return self._feats  # (N, 2048)

@torch.no_grad()
def activations_in_batches(x_all: torch.Tensor, batch_size: int, device: torch.device):
    # Map from [-1,1] to [0,1] if needed
    if x_all.min() < 0:
        x_all = (x_all + 1.0) * 0.5
    model = InceptionPool3(device)
    feats = []
    for i in range(0, x_all.size(0), batch_size):
        fb = model.features(x_all[i:i+batch_size].contiguous())
        feats.append(fb.cpu())
    return torch.cat(feats, dim=0).numpy()  # (N, 2048)

def _mu_sigma(feats_np):
    mu = feats_np.mean(axis=0)
    sigma = np.cov(feats_np, rowvar=False)
    return mu, sigma

def _fid_from_musig(mu1, sig1, mu2, sig2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if not np.isfinite(covmean).all():
        # add small jitter for numerical stability
        eps = 1e-6
        covmean = linalg.sqrtm((sig1 + eps*np.eye(sig1.shape[0])).dot(sig2 + eps*np.eye(sig2.shape[0])))[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sig1 + sig2 - 2.0 * covmean)
    return float(fid)

@torch.no_grad()
def fid_from_tensors(x_real: torch.Tensor, x_fake: torch.Tensor, batch_size: int = 64, device: str = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fr = activations_in_batches(x_real, batch_size, device)
    ff = activations_in_batches(x_fake, batch_size, device)
    mu_r, sig_r = _mu_sigma(fr)
    mu_f, sig_f = _mu_sigma(ff)
    return _fid_from_musig(mu_r, sig_r, mu_f, sig_f)
