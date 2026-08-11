#!/usr/bin/env python
"""Train one independent CNN predictor per measurable MorphoMNIST++ factor (12 separate networks,
same architecture template, no shared weights -- see `data/attr_predictor.py`).

Targets mirror `data/measure_morphomnist.py`'s `measure_all` exclusions: no `digit` (needs a
classifier, not a regressor) and no `texture_seed` (an arbitrary seed, not a physical quantity).
`hue` and `bg_phase` are circular (predicted as sin/cos); everything else is a plain scalar,
min-max normalized to a range read off the *actual built dataset* (not the sampling config's
requested range -- achieved values can and do escape the request range, e.g. `thickness`'s
vanish-guard).

Splits: `partition==1` (the packed test split, ~10k images) is never touched here -- it is
reserved for `evaluate_attr_predictors.py`'s CNN-vs-deterministic comparison. Validation (for
early stopping / checkpoint selection) is carved out of `partition==0` only.

Run: python experiments/hdae/data/train_morpho_attr_predictors.py
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import Subset

from experiments.hdae.data.attr_predictor import AttrSpec, AugmentConfig, train_attr_predictor
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# name -> "circular" with a period, everything else defaults to "scalar"
CIRCULAR_ATTRS = {"hue": 1.0, "bg_phase": 2 * np.pi}
# rotation/slant: classification over fixed +/-45deg bins instead of MSE regression -- MSE rewards
# hedging toward the mean on an attribute this confounded with unrelated digit shape (see
# PROGRESS-SUMMARY / commit log: regression plateaued at ~0.55-0.57x the naive-mean baseline even
# after regularization + wider capacity). Fixed range (not empirical, unlike every other scalar
# attribute here) because bin edges need to be stable/interpretable, not fitted to this run's data.
CATEGORICAL_ATTRS = {
    "rotation": (-30.0, 30.0, 20), "slant": (-30.0, 30.0, 10),
    "translate_x": (-10.0, 10.0, 20), "translate_y": (-10.0, 10.0, 20),
    # digit: 10 exact classes (bin width 1.0, centers land on 0..9), not a proxy binning of a
    # continuous quantity like rotation/slant -- but the same AttrSpec(kind="categorical")
    # machinery applies directly. No closed-form baseline exists for this one either (same as
    # rotation/slant): measure_morphomnist.py excludes it for exactly this reason.
    "digit": (-0.5, 9.5, 10),
}
TARGET_ATTRS = ["thickness", "intensity", "hue", "slant", "rotation", "scale",
                "translate_x", "translate_y", "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude", "digit"]
RANGE_PAD_FRAC = 0.02  # small headroom beyond the observed min/max so edge samples don't sit exactly at +/-1


def build_specs(ds: MorphoMNISTPacked, train_indices: np.ndarray) -> dict:
    """Empirical (not config-declared) lo/hi per scalar attribute, computed on the training split
    only -- so no information about val/test values leaks into the normalization bounds."""
    specs = {}
    for name in TARGET_ATTRS:
        col = ds.attribute_names.index(name)
        if name in CIRCULAR_ATTRS:
            specs[name] = AttrSpec(name=name, kind="circular", period=CIRCULAR_ATTRS[name])
            continue
        if name in CATEGORICAL_ATTRS:
            lo, hi, num_bins = CATEGORICAL_ATTRS[name]
            specs[name] = AttrSpec(name=name, kind="categorical", lo=lo, hi=hi, num_bins=num_bins)
            continue
        values = ds.attrs[train_indices, col]
        lo, hi = float(values.min()), float(values.max())
        pad = (hi - lo) * RANGE_PAD_FRAC
        specs[name] = AttrSpec(name=name, kind="scalar", lo=lo - pad, hi=hi + pad)
    return specs


def build_summary_from_disk(output_dir: str, ds: MorphoMNISTPacked) -> dict:
    """Rebuilds training_summary.json by scanning output_dir for completed (best.ckpt + spec.json)
    attributes, rather than trusting only what this process itself trained. Safe to call from
    multiple concurrent `--only <attr>` processes -- each writes an idempotent function of
    whatever has finished on disk so far, and a final consolidation pass after all of them exit
    is guaranteed to see everything."""
    results = {}
    for name in TARGET_ATTRS:
        ckpt = Path(output_dir) / name / "best.ckpt"
        spec_path = Path(output_dir) / name / "spec.json"
        if not (ckpt.exists() and spec_path.exists()):
            continue
        spec = json.loads(spec_path.read_text())
        results[name] = {"checkpoint": str(ckpt), "attr_col": ds.attribute_names.index(name), **spec}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist.h5")
    p.add_argument("--output-dir", default="experiments/hdae/outputs/attr_predictors")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of partition==0 carved out for validation")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20, help="20, not 8: some attributes (e.g. bg_freq) spend "
                   "several epochs stuck predicting the training-set mean before breaking away -- 8 stops before "
                   "that happens and silently ships a mean-predictor. See PROGRESS-SUMMARY / commit log.")
    p.add_argument("--lr", type=float, default=3e-4, help="1e-3 was too high combined with patience=8: several "
                   "predictors plateaued at the constant-mean loss and never escaped before stopping")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr-plateau-patience", type=int, default=0,
                   help="epochs of no val_loss improvement before halving LR; 0 disables the scheduler")
    p.add_argument("--augment", action="store_true",
                   help="mild Gaussian-noise + brightness-jitter augmentation on the train split only "
                        "(no blur -- would destroy the edge-orientation signal rotation/slant need)")
    p.add_argument("--only", nargs="*", default=None, help="subset of TARGET_ATTRS to train (default: all 12)")
    p.add_argument("--skip-training", action="store_true",
                   help="only rebuild training_summary.json from whatever's already on disk (used by the "
                        "parallel launcher's final consolidation pass)")
    args = p.parse_args()

    ds = MorphoMNISTPacked(args.packed, preload_images=not args.skip_training)
    partitions = ds.partitions
    all_train = np.nonzero(partitions == 0)[0]
    rng = np.random.RandomState(0)
    rng.shuffle(all_train)
    n_val = int(len(all_train) * args.val_frac)
    val_indices = all_train[:n_val]
    train_indices = all_train[n_val:]
    test_count = int((partitions == 1).sum())
    logging.info("packed=%s n_total=%d train=%d val=%d test(reserved, untouched)=%d",
                args.packed, len(ds), len(train_indices), len(val_indices), test_count)

    specs = build_specs(ds, train_indices)
    targets = args.only or TARGET_ATTRS
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    augment = AugmentConfig(gaussian_noise_prob=0.3, gaussian_noise_std=0.03,
                            brightness_jitter_prob=0.3, brightness_jitter_strength=0.1) if args.augment else None

    if not args.skip_training:
        for name in targets:
            spec = specs[name]
            col = ds.attribute_names.index(name)
            if spec.kind == "scalar":
                detail = f", range=[{spec.lo:.3f},{spec.hi:.3f}]"
            elif spec.kind == "categorical":
                detail = f", range=[{spec.lo:.3f},{spec.hi:.3f}], num_bins={spec.num_bins}"
            else:
                detail = f", period={spec.period:.4f}"
            logging.info("=== training predictor for %r (kind=%s%s) ===", name, spec.kind, detail)
            best_ckpt = train_attr_predictor(
                spec=spec, attr_col=col,
                train_dataset=Subset(ds, train_indices.tolist()), val_dataset=Subset(ds, val_indices.tolist()),
                output_dir=args.output_dir, base_channels=args.base_channels, lr=args.lr,
                weight_decay=args.weight_decay, dropout=args.dropout, lr_plateau_patience=args.lr_plateau_patience,
                augment=augment,
                batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience,
                num_workers=args.num_workers, accelerator=args.accelerator,
            )
            logging.info("=== %r done: best checkpoint=%s ===", name, best_ckpt)

    results = build_summary_from_disk(args.output_dir, ds)
    summary_path = Path(args.output_dir) / "training_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    logging.info("wrote training summary to %s (%d/%d attributes present)",
                summary_path, len(results), len(TARGET_ATTRS))


if __name__ == "__main__":
    main()
