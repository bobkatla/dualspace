"""Train conditional diffusion generator (images first).


CLI
---
python -m dualspace.train.train_generator --config configs/cifar10.yaml
"""
from __future__ import annotations
import argparse


# TODO: parse config, build data, build g and diffusion, train loop with EMA and checkpointing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()