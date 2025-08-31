#!/usr/bin/env bash
set -euo pipefail
uv run dualspace train-amortized --config configs/cifar10.yaml
