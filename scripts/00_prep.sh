#!/usr/bin/env bash
set -euo pipefail
uv run dualspace prep-data --config configs/cifar10.yaml --preview
uv run dualspace diag-g --config configs/cifar10.yaml # just to check g(c)
uv run dualspace fit-phi --config configs/cifar10.yaml 