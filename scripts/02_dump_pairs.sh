#!/usr/bin/env bash
set -euo pipefail
uv run dualspace dump-pairs --config configs/cifar10.yaml
