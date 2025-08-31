#!/usr/bin/env bash
set -euo pipefail
uv run dualspace dump-pairs --config configs/cifar10.yaml --split train
uv run dualspace dump-pairs --config configs/cifar10.yaml --split calib
uv run dualspace dump-pairs --config configs/cifar10.yaml --split test
