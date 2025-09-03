#!/bin/bash
set -e

CFG=configs/cifar10.yaml
RUN=outputs/cifar10_mvp

# recompute (if needed) region inference to produce .npz/.pt
uv run dualspace region-infer --config $CFG --alpha 0.9 --per-class --K 256

# coverage + pareto (JSONs)
uv run dualspace coverage-curve --config $CFG --per-class
uv run dualspace pareto          --config $CFG --per-class --K 256

# plots
uv run dualspace visualize coverage --run-dir $RUN
uv run dualspace visualize pareto   --run-dir $RUN

# export real test images for FID once
uv run dualspace export-test-images --config $CFG --limit 5000

# pick a class to evaluate (example: class 0)
FAKE_PHI=$RUN/region_infer/class0_survivors_phi.npz
REAL_PHI=$RUN/pairs/test.npz         # contains Y for all test
FAKE_PT=$RUN/region_infer/class0_survivors.pt
REAL_PT=$RUN/metrics/test_images.pt

# informativeness (logdet(cov_φ))
uv run dualspace metrics informativeness --phi-npz $FAKE_PHI

# MMD in φ-space (survivors vs real test φ)
uv run dualspace metrics mmd       --x-npz $FAKE_PHI --y-npz $REAL_PHI

# Sinkhorn OT in φ-space
uv run dualspace metrics sinkhorn  --x-npz $FAKE_PHI --y-npz $REAL_PHI --reg 0.05

# FID on images (subset of survivors vs test)
uv run dualspace metrics fid       --real-pt $REAL_PT --fake-pt $FAKE_PT
