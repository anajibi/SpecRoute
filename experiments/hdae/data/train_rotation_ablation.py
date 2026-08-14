#!/usr/bin/env python
"""Rotation-only classifier on the rotation-ablation dataset (no slant, 5 bins over +/-45deg).

Standalone rather than a --only rotation call into train_morpho_attr_predictors.py: that script's
CATEGORICAL_ATTRS["rotation"] = (-30, 30, 10) is tied to the main morphomnist_70k.h5 dataset (13
attributes currently trained/training against it) -- this ablation uses a different range/bin
count on a different (throwaway, single-purpose) dataset, and overloading the shared dict would
risk a mismatch for the main run. Same model/training code (attr_predictor.py), just a standalone
driver so the two don't share mutable config state.
"""
import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from torch.utils.data import Subset

from experiments.hdae.data.attr_predictor import AttrSpec, AugmentConfig, train_attr_predictor
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist_rotation_ablation.h5")
    p.add_argument("--output-dir", default="experiments/hdae/outputs/attr_predictors_rotation_ablation")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--rotation-lo", type=float, default=-45.0)
    p.add_argument("--rotation-hi", type=float, default=45.0)
    p.add_argument("--rotation-bins", type=int, default=5)
    args = p.parse_args()

    ds = MorphoMNISTPacked(args.packed, preload_images=True)
    partitions = ds.partitions
    all_train = np.nonzero(partitions == 0)[0]
    rng = np.random.RandomState(0)
    rng.shuffle(all_train)
    n_val = int(len(all_train) * args.val_frac)
    val_indices = all_train[:n_val]
    train_indices = all_train[n_val:]
    logging.info("packed=%s n_total=%d train=%d val=%d", args.packed, len(ds), len(train_indices), len(val_indices))

    spec = AttrSpec(name="rotation", kind="categorical", lo=args.rotation_lo, hi=args.rotation_hi,
                    num_bins=args.rotation_bins)
    col = ds.attribute_names.index("rotation")
    logging.info("=== training rotation ablation: range=[%.1f,%.1f] num_bins=%d ===",
                args.rotation_lo, args.rotation_hi, args.rotation_bins)

    augment = AugmentConfig(gaussian_noise_prob=0.3, gaussian_noise_std=0.03,
                            brightness_jitter_prob=0.3, brightness_jitter_strength=0.1) if args.augment else None

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_ckpt = train_attr_predictor(
        spec=spec, attr_col=col,
        train_dataset=Subset(ds, train_indices.tolist()), val_dataset=Subset(ds, val_indices.tolist()),
        output_dir=args.output_dir, base_channels=args.base_channels, lr=args.lr,
        weight_decay=args.weight_decay, dropout=args.dropout, augment=augment,
        batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience,
        num_workers=args.num_workers, accelerator=args.accelerator,
    )
    logging.info("done: best checkpoint=%s", best_ckpt)

    results = {"rotation": {"checkpoint": best_ckpt, "attr_col": col, **dataclasses.asdict(spec)}}
    (Path(args.output_dir) / "training_summary.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
