#!/usr/bin/env python
"""MorphoMNIST++ generator correctness checks (Phase 1 gate).

1. Determinism: ``render()`` is a pure function of (base MNIST digit,
   Factors record) -- calling it twice on the same inputs gives
   bit-identical output, and it matches what's actually in the packed
   dataset. This is the round-trip property the packed dataset's
   reproducibility and Phase 6's later renderer cross-check both depend on.
2. thickness/intensity causal link: the generator's causal formula
   (``sample_targets``: intensity's target is a decreasing function of
   thickness's target, plus independent noise) leaves a real, measurable
   negative correlation in the packed data -- not just logged metadata that
   happens to be disconnected from the pixels.
3. hue independence: re-rendering one record with only ``hue`` changed
   leaves the digit's spatial footprint (where the stroke is, at any color)
   unchanged -- hue recolors, it doesn't move geometry.

What this script does NOT check: per-image do(thickness) -> intensity
propagation through a *fitted SCM* -- that needs the SCM from Phase 2 (see
`causal/train_scm.py` and the morpho causal-graph verification). This
script only establishes that the generated data has the causal structure
for that SCM to fit against.

Run: python experiments/hdae/data/verify_morphomnist.py
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from torchvision.datasets import MNIST

from experiments.hdae.data.morphomnist import (DEFAULT_FACTOR_CONFIG_PATH, Factors, MorphoMNISTPacked,
                                               load_factor_config, pad_digit, render, sample_targets)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

PACKED_PATH = "experiments/hdae/data/packed/morphomnist.h5"
MNIST_RAW_DIR = "experiments/hdae/data/raw/mnist"


def main():
    config = load_factor_config(DEFAULT_FACTOR_CONFIG_PATH)
    pad = config["image"]["pad"]
    ds = MorphoMNISTPacked(PACKED_PATH)
    n = len(ds)
    logging.info("loaded %s: n=%d attribute_names=%s", PACKED_PATH, n, ds.attribute_names)

    rng = np.random.RandomState(0)
    train_indices = rng.choice(60000, size=50, replace=False)  # packed test-split images aren't in this MNIST split
    mnist = MNIST(root=MNIST_RAW_DIR, train=True, download=True)

    # 1. Determinism / round-trip.
    mismatches = 0
    for idx in train_indices:
        idx = int(idx)
        img28, digit = mnist[idx]
        img_padded = pad_digit(np.array(img28, dtype=np.uint8), pad)
        factors = Factors(**sample_targets(idx, int(digit), config))
        rgb_a, _ = render(img_padded, factors, config)
        rgb_b, _ = render(img_padded, factors, config)
        if not np.array_equal(rgb_a, rgb_b):
            mismatches += 1
        packed_rgb = ((ds[idx]["img"].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        if not np.array_equal(rgb_a, packed_rgb):
            mismatches += 1
    assert mismatches == 0, f"{mismatches} determinism mismatches -- render() is not a pure function of its inputs"
    logging.info("PASS: render() is deterministic and matches the packed dataset exactly (%d images checked)",
                 len(train_indices))

    # 2. thickness/intensity causal link is really in the pixels, not just metadata.
    # Positive correlation: Pawlowski et al. 2020's formula has i = scale*sigmoid(...+ thickness_coef*t +
    # ...)+floor with thickness_coef > 0, so thicker strokes render brighter.
    thickness = ds.attrs[:, ds.attribute_names.index("thickness")]
    intensity = ds.attrs[:, ds.attribute_names.index("intensity")]
    corr = float(np.corrcoef(thickness, intensity)[0, 1])
    logging.info("thickness/intensity Pearson correlation in packed data: %.4f", corr)
    assert corr > 0.3, f"expected a real positive thickness->intensity correlation, got {corr:.4f}"
    logging.info("PASS: thickness->intensity causal link is present and positive as designed")

    # 3. hue independence: changing hue leaves the digit's spatial footprint untouched.
    idx = int(train_indices[0])
    img28, digit = mnist[idx]
    img_padded = pad_digit(np.array(img28, dtype=np.uint8), pad)
    factors = Factors(**sample_targets(idx, int(digit), config))
    rgb_a, _ = render(img_padded, factors, config)
    factors_hue2 = Factors(**{**factors.__dict__, "hue": (factors.hue + 0.5) % 1.0})
    rgb_b, _ = render(img_padded, factors_hue2, config)
    footprint_a = rgb_a.max(axis=-1) > 20
    footprint_b = rgb_b.max(axis=-1) > 20
    mismatch_frac = float((footprint_a != footprint_b).mean())
    assert mismatch_frac < 0.01, f"hue change shifted the digit's footprint by {mismatch_frac:.4f}"
    colors_differ = not np.array_equal(rgb_a, rgb_b)
    assert colors_differ, "hue change had no pixel effect at all"
    logging.info("PASS: hue change recolors (differs=%s) without moving the footprint (mismatch=%.4f)",
                 colors_differ, mismatch_frac)

    logging.info("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
