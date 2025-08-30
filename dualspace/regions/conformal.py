"""Conformal calibration of tau_alpha.


Scores
------
- s(x, c) := -log \hat p_psi(phi(x) | e=g(c)) (lower is better)


Procedure
---------
- Compute scores on calibration split (class-conditional by default).
- For each alpha in config, set tau_alpha as the (1-alpha)*(n+1)/n quantile of -scores,
then store log-thresholds for region predicate log p >= tau_alpha.


CLI
---
python -m dualspace.regions.conformal --config configs/cifar10.yaml --split calib
"""
from __future__ import annotations
import argparse
from typing import Dict, List


# TODO: implement main() that loads MDN + phi + g, computes thresholds per class and saves JSON


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="calib")
    args = parser.parse_args()
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()