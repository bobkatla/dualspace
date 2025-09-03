#!/usr/bin/env bash
set -euo pipefail
uv run dualspace region-infer --config configs/cifar10.yaml --alpha 0.9 --per-class --K 256
