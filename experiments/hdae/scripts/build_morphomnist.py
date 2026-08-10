#!/usr/bin/env python
"""Build the packed MorphoMNIST++ dataset (TODO item 3).

Downloads standard MNIST via torchvision's public mirror (self-contained --
no other local project's data or code is used), applies the causally-linked
thickness/intensity perturbation plus the modeled `hue` and the
injected-unobserved factors (see `data/morphomnist.py`), and packs
everything into a single `.npz`: `images` (N,32,32,3 uint8), `attrs`
(N,len(attribute_names) float32), `attribute_names`, `partitions`
(0=train,1=test) -- same field names as `data/celeba_hq.py`'s
`*_attrs.npz` convention, so the causal SCM / CF1 tooling that already
reads that shape needs no changes to read this one.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from torchvision.datasets import MNIST

from experiments.hdae.data.morphomnist import ATTRIBUTE_NAMES, Factors, pad_to_32, render, sample_targets

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def build_split(mnist_ds, index_offset, n, log_every=10000):
    images = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    attrs = np.zeros((n, len(ATTRIBUTE_NAMES)), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        img28, digit = mnist_ds[i]
        img32 = pad_to_32(np.array(img28, dtype=np.uint8))
        global_index = index_offset + i
        factors = Factors(**sample_targets(global_index, int(digit)))
        rgb, achieved = render(img32, factors)
        images[i] = rgb
        attrs[i] = achieved.to_vector()
        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            logging.info("  %d/%d (%.0f img/s)", i + 1, n, rate)
    return images, attrs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="experiments/hdae/data/raw/mnist")
    p.add_argument("--n-train", type=int, default=60000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--output", default="experiments/hdae/data/packed/morphomnist_32.npz")
    args = p.parse_args()

    logging.info("downloading/loading standard MNIST from torchvision into %s", args.raw_dir)
    train_ds = MNIST(root=args.raw_dir, train=True, download=True)
    test_ds = MNIST(root=args.raw_dir, train=False, download=True)
    n_train = min(args.n_train, len(train_ds))
    n_test = min(args.n_test, len(test_ds))

    logging.info("rendering %d train images (attributes=%s)", n_train, ATTRIBUTE_NAMES)
    train_images, train_attrs = build_split(train_ds, index_offset=0, n=n_train)
    logging.info("rendering %d test images", n_test)
    # Offset test indices past train's so the per-index RNG seed never collides with a train image.
    test_images, test_attrs = build_split(test_ds, index_offset=10_000_000, n=n_test)

    images = np.concatenate([train_images, test_images], axis=0)
    attrs = np.concatenate([train_attrs, test_attrs], axis=0)
    partitions = np.concatenate([np.zeros(n_train, dtype=np.int64), np.ones(n_test, dtype=np.int64)])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, images=images, attrs=attrs,
             attribute_names=np.array(ATTRIBUTE_NAMES, dtype=object), partitions=partitions)
    logging.info("wrote %s: images=%s attrs=%s (train=%d test=%d)", output, images.shape, attrs.shape, n_train, n_test)


if __name__ == "__main__":
    main()
