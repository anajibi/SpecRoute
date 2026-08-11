#!/usr/bin/env python
"""Build the packed MorphoMNIST++ dataset (TODO item 3).

Downloads standard MNIST via torchvision's public mirror (self-contained --
no other local project's data or code is used), applies the causally-linked
thickness/intensity perturbation plus the modeled `hue` and the
injected-unobserved factors (see `data/morphomnist.py`), sampled per
`configs/morphomnist_factors.yaml`, and packs everything into a single
HDF5 file: `images` (N,canvas,canvas,3 uint8, chunked along the batch
dimension), `attrs` (N,len(attribute_names) float32), `attribute_names`
(variable-length UTF-8 strings), `partitions` (0=train,1=test) -- same
field names/shapes as `data/celeba_hq.py`'s `*_attrs.npz` convention (just
HDF5 instead of npz, and images live in the same file rather than a
separate LMDB), so the causal SCM / CF1 tooling that already reads that
shape needs only a format-dispatching loader, not new logic.

Written directly into pre-sized HDF5 datasets as each split renders,
rather than building full train+test numpy arrays and concatenating --
avoids a transient ~2x peak-memory spike at 64x64 resolution.

Review a grid before running this: `scripts/preview_morphomnist.py` renders
a preview grid from the same config without writing the full dataset.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from torchvision.datasets import MNIST

from experiments.hdae.data.morphomnist import (ATTRIBUTE_NAMES, DEFAULT_FACTOR_CONFIG_PATH, Factors,
                                               load_factor_config, pad_digit, render, sample_targets)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

IMAGE_CHUNK_ROWS = 256  # chunk along the batch dim so sequential batch reads are ~one chunk read


def build_split_into(mnist_ds, config, index_offset, n, images_ds, attrs_ds, out_offset, log_every=10000):
    """`n` may exceed `len(mnist_ds)` -- base images then cycle (`i % len(mnist_ds)`), each repeat
    getting a genuinely different factor draw since `global_index` (used as the per-image RNG
    seed) is never taken modulo the dataset length, only the base-image lookup is."""
    t0 = time.time()
    n_base = len(mnist_ds)
    for i in range(n):
        img28, digit = mnist_ds[i % n_base]
        img_padded = pad_digit(np.array(img28, dtype=np.uint8), config["image"]["pad"])
        global_index = index_offset + i
        factors = Factors(**sample_targets(global_index, int(digit), config))
        rgb, achieved = render(img_padded, factors, config)
        images_ds[out_offset + i] = rgb
        attrs_ds[out_offset + i] = achieved.to_vector()
        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            logging.info("  %d/%d (%.0f img/s)", i + 1, n, rate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factor-config", default=DEFAULT_FACTOR_CONFIG_PATH)
    p.add_argument("--raw-dir", default="experiments/hdae/data/raw/mnist")
    p.add_argument("--n-train", type=int, default=60000, help="may exceed MNIST's native 60k -- base "
                   "images cycle, each repeat gets a distinct factor draw (see build_split_into)")
    p.add_argument("--n-test", type=int, default=10000, help="may exceed MNIST's native 10k, same as --n-train")
    p.add_argument("--output", default="experiments/hdae/data/packed/morphomnist.h5")
    args = p.parse_args()

    config = load_factor_config(args.factor_config)
    canvas = config["image"]["canvas_size"]
    logging.info("loaded factor config=%s canvas_size=%d", args.factor_config, canvas)

    logging.info("downloading/loading standard MNIST from torchvision into %s", args.raw_dir)
    train_ds = MNIST(root=args.raw_dir, train=True, download=True)
    test_ds = MNIST(root=args.raw_dir, train=False, download=True)
    n_train, n_test = args.n_train, args.n_test
    n_total = n_train + n_test
    if n_train > len(train_ds) or n_test > len(test_ds):
        logging.info("n_train/n_test exceed MNIST's native %d/%d -- base images will repeat with "
                     "distinct sampled factors per repeat", len(train_ds), len(test_ds))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunk_rows = min(IMAGE_CHUNK_ROWS, n_total)

    with h5py.File(output, "w") as f:
        images_ds = f.create_dataset("images", shape=(n_total, canvas, canvas, 3), dtype=np.uint8,
                                     chunks=(chunk_rows, canvas, canvas, 3), compression="lzf")
        attrs_ds = f.create_dataset("attrs", shape=(n_total, len(ATTRIBUTE_NAMES)), dtype=np.float32)
        f.create_dataset("attribute_names", data=np.array(ATTRIBUTE_NAMES, dtype=object),
                         dtype=h5py.string_dtype(encoding="utf-8"))
        partitions_ds = f.create_dataset("partitions", shape=(n_total,), dtype=np.int64)
        partitions_ds[n_train:] = 1

        logging.info("rendering %d train images (attributes=%s)", n_train, ATTRIBUTE_NAMES)
        build_split_into(train_ds, config, index_offset=0, n=n_train,
                         images_ds=images_ds, attrs_ds=attrs_ds, out_offset=0)
        logging.info("rendering %d test images", n_test)
        # Offset test indices past train's so the per-index RNG seed never collides with a train image.
        build_split_into(test_ds, config, index_offset=10_000_000, n=n_test,
                         images_ds=images_ds, attrs_ds=attrs_ds, out_offset=n_train)

    logging.info("wrote %s: images=(%d,%d,%d,3) attrs=(%d,%d) (train=%d test=%d)",
                 output, n_total, canvas, canvas, n_total, len(ATTRIBUTE_NAMES), n_train, n_test)


if __name__ == "__main__":
    main()
