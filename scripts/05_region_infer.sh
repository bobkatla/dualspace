#!/usr/bin/env bash
set -euo pipefail
uv run dualspace infer --config configs/cifar10.yaml --split test
