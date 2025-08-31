#!/usr/bin/env bash
set -euo pipefail
uv run dualspace train-gen --config configs/cifar10.yaml
