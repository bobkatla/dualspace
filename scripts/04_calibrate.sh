#!/usr/bin/env bash
set -euo pipefail
uv run dualspace calibrate --config configs/cifar10.yaml --split calib
