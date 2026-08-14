#!/usr/bin/env python
"""Precompute per-image flip-intervention targets for thickness/intensity/hue, bin-based.

Replaces the earlier fixed p15/p85 targets (same absolute value for every image in a cohort),
which the earlier gs=5/8 CC/FC/CF1 run used. Problem with that design: a fixed target doesn't
guarantee a meaningful, comparable-magnitude shift for every individual image -- an image already
sitting near p85 barely moves when "intervened" toward p85, while one near the opposite extreme
moves a lot. Same underlying issue as the original out-of-distribution targets, just subtler.

New design, per attribute (thickness/intensity/hue -- digit is untouched, still a forced shift
to (digit+5)%10, not a binning problem):
1. Partition the attribute's values -- computed on the SAME fixed 512-image evaluation cohort
   used by morpho_cf1_eval.py (identical index-selection logic: RandomState(0).choice over the
   test partition, sorted), not the train partition -- into N_BINS population-quantile bins
   (equal member count per bin, via np.quantile edges).
2. Each image's own value places it in a source bin (0..N_BINS-1).
3. Flip rule: images in the low half (bin < N_BINS/2) get a target bin from the high half
   (bin >= N_BINS/2); images in the high half get a target from the low half. Additional
   constraint: the target bin must be at least MIN_BIN_DISTANCE bins away from the source bin
   (for N_BINS=10 this only ever excludes the bin immediately across the low/high boundary --
   e.g. source bin 4 excludes target bin 5 -- the eligible set is never empty).
4. Among eligible target bins, one is sampled uniformly at random per image (fixed seed, so this
   is reproducible) -- not every image in a source bin gets pushed to the identical target bin,
   avoiding the earlier one-fixed-target-per-cohort collapse.
5. Intervention value = the midpoint of the target bin's [lo, hi) quantile edges (not the mean or
   median of members actually in that bin -- the bin's value-range midpoint, per instruction).

Output: one row per (image_index, attribute) with source bin/value and target bin/value, written
to JSON + CSV for review before any model inference runs against these cohorts.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.hdae.data.morphomnist import MorphoMNISTPacked

CONTINUOUS_ATTRS = ["thickness", "intensity"]
# hue removed (2026-08-11) -- now a categorical attribute (10 fixed bin centers, not continuous
# Uniform(0,1)), so it gets a digit-style shift intervention in morpho_cf1_eval_binned.py, not a
# population-decile flip target from this cohort file.


def fixed_eval_indices(ds, index_seed=0, n_images=512):
    """Must match morpho_cf1_eval.py's index selection exactly -- same cohort, same targets."""
    test_idx_all = np.nonzero(ds.partitions == 1)[0]
    rng = np.random.RandomState(index_seed)
    idx = rng.choice(test_idx_all, size=min(n_images, len(test_idx_all)), replace=False)
    idx.sort()
    return idx


def compute_bins(values, n_bins):
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9  # right-inclusive on the last edge so the max value gets bin n_bins-1
    bin_idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
    midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
    counts = [int((bin_idx == i).sum()) for i in range(n_bins)]
    return bin_idx, edges, midpoints, counts


def eligible_targets(source_bin, n_bins, min_dist):
    half = n_bins // 2
    if source_bin < half:
        candidates = range(half, n_bins)
    else:
        candidates = range(0, half)
    return [b for b in candidates if abs(b - source_bin) >= min_dist]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist.h5")
    p.add_argument("--n-images", type=int, default=512)
    p.add_argument("--index-seed", type=int, default=0)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--min-bin-distance", type=int, default=2)
    p.add_argument("--sample-seed", type=int, default=42, help="seed for per-image target-bin sampling")
    p.add_argument("--output-json", default="experiments/hdae/outputs/intervention_cohorts.json")
    p.add_argument("--output-csv", default="experiments/hdae/outputs/intervention_cohorts.csv")
    args = p.parse_args()

    ds = MorphoMNISTPacked(args.packed)
    indices = fixed_eval_indices(ds, args.index_seed, args.n_images)
    print(f"fixed index set: n={len(indices)} seed={args.index_seed} "
         f"first5={indices[:5].tolist()} last5={indices[-5:].tolist()}")

    rng = np.random.RandomState(args.sample_seed)
    cohorts = {}
    rows_csv = []

    for attr in CONTINUOUS_ATTRS:
        col = ds.attribute_names.index(attr)
        values = ds.attrs[indices, col]
        bin_idx, edges, midpoints, counts = compute_bins(values, args.n_bins)

        print(f"\n=== {attr} ===")
        print(f"  bin edges: {[round(float(e), 4) for e in edges]}")
        print(f"  bin counts (should be ~equal, n={args.n_images}/{args.n_bins}={args.n_images/args.n_bins:.1f} each): {counts}")
        print(f"  bin midpoints: {[round(m, 4) for m in midpoints]}")

        source_bins, target_bins, target_values = [], [], []
        for i, sb in enumerate(bin_idx):
            eligible = eligible_targets(int(sb), args.n_bins, args.min_bin_distance)
            tb = int(rng.choice(eligible))
            source_bins.append(int(sb))
            target_bins.append(tb)
            target_values.append(midpoints[tb])

        source_bins = np.array(source_bins)
        target_bins = np.array(target_bins)
        target_values = np.array(target_values)

        print(f"  sample rows (first 8 of {len(indices)}):")
        print(f"    {'img_idx':<10}{'value':<10}{'src_bin':<9}{'tgt_bin':<9}{'tgt_value':<10}")
        for i in range(8):
            print(f"    {int(indices[i]):<10}{values[i]:<10.3f}{source_bins[i]:<9}{target_bins[i]:<9}{target_values[i]:<10.3f}")

        target_bin_dist = np.bincount(target_bins, minlength=args.n_bins).tolist()
        print(f"  target bin distribution: {target_bin_dist}")
        min_gap = np.abs(target_bins - source_bins).min()
        print(f"  min |target_bin - source_bin| observed: {min_gap} (constraint: >= {args.min_bin_distance})")

        cohorts[attr] = {
            "n_bins": args.n_bins, "min_bin_distance": args.min_bin_distance,
            "bin_edges": [float(e) for e in edges], "bin_midpoints": [float(m) for m in midpoints],
            "bin_counts": counts,
            "per_image": [
                {"index": int(indices[i]), "value": float(values[i]), "source_bin": int(source_bins[i]),
                 "target_bin": int(target_bins[i]), "target_value": float(target_values[i])}
                for i in range(len(indices))
            ],
        }
        for i in range(len(indices)):
            rows_csv.append({"index": int(indices[i]), "attribute": attr, "value": float(values[i]),
                             "source_bin": int(source_bins[i]), "target_bin": int(target_bins[i]),
                             "target_value": float(target_values[i])})

    cohorts["_meta"] = {"n_images": args.n_images, "index_seed": args.index_seed,
                        "n_bins": args.n_bins, "min_bin_distance": args.min_bin_distance,
                        "sample_seed": args.sample_seed, "fixed_indices": indices.tolist()}

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(cohorts, indent=2))
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nwrote {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
