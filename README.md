# dualspace (MVP)


Minimal, reproducible framework for conditional regions in generative models via feature-space densities and conformal calibration.


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