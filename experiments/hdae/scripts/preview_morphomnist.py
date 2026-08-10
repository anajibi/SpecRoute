#!/usr/bin/env python
"""Preview a grid of MorphoMNIST++ samples from configs/morphomnist_factors.yaml
without building the full packed dataset.

Run this (and look at the grid) every time you edit the factor config,
before running scripts/build_morphomnist.py for real -- there is no
automated check for "do the images still look like digits", only your eyes
(see TODO-List item 3's bug log: three of four image-quality bugs were only
caught by visual review, not by the determinism/correlation assertions in
data/verify_morphomnist.py).
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST

from experiments.hdae.data.morphomnist import (DEFAULT_FACTOR_CONFIG_PATH, Factors, load_factor_config,
                                               pad_digit, render, sample_targets)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factor-config", default=DEFAULT_FACTOR_CONFIG_PATH)
    p.add_argument("--raw-dir", default="experiments/hdae/data/raw/mnist")
    p.add_argument("-n", "--num-images", type=int, default=50)
    p.add_argument("--cols", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cell-size", type=int, default=96, help="Display size per cell, px (nearest-neighbor upscale).")
    p.add_argument("--output", default="/tmp/morphomnist_preview.png")
    args = p.parse_args()

    config = load_factor_config(args.factor_config)
    pad = config["image"]["pad"]
    canvas = config["image"]["canvas_size"]
    logging.info("factor config=%s canvas_size=%d", args.factor_config, canvas)

    mnist = MNIST(root=args.raw_dir, train=True, download=True)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(mnist), size=args.num_images, replace=False)

    tiles = []
    for idx in indices:
        idx = int(idx)
        img28, digit = mnist[idx]
        img_padded = pad_digit(np.array(img28, dtype=np.uint8), pad)
        factors = Factors(**sample_targets(idx, int(digit), config))
        rgb, achieved = render(img_padded, factors, config)
        tiles.append(rgb)

    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    pad_tiles = tiles + [np.zeros((canvas, canvas, 3), dtype=np.uint8)] * (rows * cols - len(tiles))
    grid = np.concatenate([np.concatenate(pad_tiles[r * cols:(r + 1) * cols], axis=1) for r in range(rows)], axis=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).resize((grid.shape[1] * args.cell_size // canvas, grid.shape[0] * args.cell_size // canvas),
                                 Image.NEAREST).save(out)
    logging.info("wrote %d-image preview grid (%d cols x %d rows) to %s", len(tiles), cols, rows, out)


if __name__ == "__main__":
    main()
