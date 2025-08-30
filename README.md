# dualspace (MVP)


A conditional generator that maps a condition cc (labels/attributes) to a region of plausible outcomes rather than a single sample. Concretely, we 

- (i) train a conditional diffusion model $p_\theta(x\mid e)$ with a learned condition embedding $e=g(c)$;
- (ii) define a feature map $\phi(x)=y$ into a low-dimensional, stable space;
- (iii) fit an amortized conditional density $\hat p_\psi(y\mid e)$ (MDN);
- (iv) calibrate thresholds $\tau_\alpha$ via conformal prediction to guarantee finite-sample coverage $\Pr[x\in \mathcal R_\alpha(c)]\ge \alpha$;
- (v) at inference, sample drafts, score in $\phi$-space, keep those in the calibrated region, and pick diverse representatives.

To begin, it will be an MVP across CIFAR-10 (images) and UCI HAR (time-series), with clean scripts, metrics, and plots.


## Assumptions
 - Data shapes & normalization
    - CIFAR-10: $x ∈ [-1,1]$, tensor (B,3,32,32). Labels one-hot (B,10).
    - HAR: (B,D,T) with D=9 channels, target T=128 (pad/crop if needed), z-score per channel using train stats. Labels one-hot (B,6).
- Condition encoder $g(c)$: MLP, input size = #classes, output d_c=64.
- Diffusion
    - Timesteps T=200, cosine β-schedule, noise-prediction loss (MSE), EMA.
    - Classifier-free guidance via p-drop on conditions; guidance scale γ=2.0.
    - U-Net2D (CIFAR) and U-Net1D (HAR), small variants for MVP.
- Feature map $\phi$
    - Images: frozen ResNet-18 penultimate features (512-d) → PCA to $d_φ=64$.
    - Time-series: small frozen 1D-CNN features (≈128-d) → PCA to $d_φ=32$.
    - PCA fit on train only; transform applied to calib/test/samples.
- Amortized density $\hat p_\psi(y|e)$
    - MDN with diagonal Gaussians; components M=6, hidden 256.
    - Inputs concatenated [y, e].
- Conformal calibration
    - Score $s(x,c)= -\log \hat p_\psi(\phi(x)\mid g(c))$.
    - Label-conditional split conformal by default; store per-class $\tau_\alpha$.
- Inference
    - Drafts per condition: K=256. Filter by $log p̂ ≥ τ_α$.
    - Representative selection via farthest-point sampling in $\phi$-space.
- Metrics
    - Coverage vs target α (overall & per class).
    - Informativeness via log-det(cov) in $\phi$-space (regularized).
    - Fidelity/diversity: FID (CIFAR), MMD($\phi$)-space (both), TSTR/TSNT (HAR).


## Getting Started


```bash
conda activate dualspace
# CIFAR-10 generator
bash scripts/01_train_gen.sh
# Dump (e, y) pairs
bash scripts/02_dump_pairs.sh
# Train amortized density
bash scripts/03_train_amortized.sh
# Calibrate thresholds
bash scripts/04_calibrate.sh
# Inference + metrics
bash scripts/05_region_infer.sh