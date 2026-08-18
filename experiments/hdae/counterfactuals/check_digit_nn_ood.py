#!/usr/bin/env python
"""Digit-intervention OOD check, nearest-neighbor version.

The marginal-range check (check_counterfactual_ood.py) only asks "is each attribute
individually inside the range seen during training" -- it says nothing about whether the
*combination* has ever been seen. This script answers the sharper question the user actually
asked: for each digit-intervention counterfactual target, does a training image of the TARGET
digit class exist with a *similar* combination of the other attributes?

digit is a root node in the causal graph (causal_graph_morpho.yaml: digit has no incoming or
outgoing edges), so a digit intervention changes nothing else -- the counterfactual target for
image i is just (target_digit, all 11 other attrs unchanged from image i). So the question
reduces cleanly to: within the TRAIN images of target_digit, what is the nearest-neighbor
distance (in normalized attribute space) to image i's own (unchanged) 11-dim attribute vector?

To judge whether that distance is "close" or "far", we compare it against two baselines built
the same way, per target class:
  1. train-self NN distance: for real training images of that class, distance to their nearest
     *other* training image of that class (leave-one-out). This is the typical spacing between
     real, seen examples -- the tightest reasonable notion of "similar".
  2. val-vs-train NN distance: for real, held-out validation images of that class (never trained
     on), distance to their nearest training image of that class. This is the model's normal
     generalization gap -- the amount of "difference from anything trained on" the model
     already has to handle successfully for ordinary reconstruction/generation to work.
If the counterfactual targets' NN distances sit inside the val-vs-train distribution, the model
has plausibly seen "something similar" in the sense that matters (it already generalizes over
gaps of that size). If they are systematically larger, that's a real, specific novelty gap.

Attribute vector (11 dims -> 12 columns): thickness, intensity, hue, rotation, scale,
translate_x, translate_y, bg_freq, bg_amplitude, texture_amplitude, and bg_phase encoded as
(cos, sin) to respect its circularity (a plain z-score would treat phase 0 and phase 2*pi as
maximally far apart, which is wrong). slant is excluded (constant by dataset design) and
texture_seed is excluded (arbitrary RNG seed, not a semantic attribute -- there is no notion of
"similar seed"). All columns z-scored using TRAIN statistics only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree

from experiments.hdae.data.morphomnist import MorphoMNISTPacked

PACKED = "experiments/hdae/data/packed/morphomnist_70k.h5"
COHORTS = "experiments/hdae/outputs/intervention_cohorts.json"
OUT_PATH = "experiments/hdae/outputs/digit_nn_ood_check.json"

LINEAR_ATTRS = ["thickness", "intensity", "hue", "rotation", "scale",
                "translate_x", "translate_y", "bg_freq", "bg_amplitude", "texture_amplitude"]
CIRCULAR_ATTR = "bg_phase"


def build_feature_matrix(attrs, names, mean, std):
    cols = []
    for name in LINEAR_ATTRS:
        cols.append(attrs[:, names.index(name)])
    phase = attrs[:, names.index(CIRCULAR_ATTR)]
    cols.append(np.cos(phase))
    cols.append(np.sin(phase))
    mat = np.stack(cols, axis=1)
    if mean is None:
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        std[std < 1e-8] = 1.0
    return (mat - mean) / std, mean, std


def main():
    ds = MorphoMNISTPacked(PACKED)
    names = ds.attribute_names
    digit_col = names.index("digit")

    train_mask = ds.partitions == 0
    val_mask = ds.partitions == 1  # held-out, never trained on

    train_attrs = ds.attrs[train_mask]
    val_attrs = ds.attrs[val_mask]
    train_feat_all, mean, std = build_feature_matrix(train_attrs, names, None, None)
    val_feat_all, _, _ = build_feature_matrix(val_attrs, names, mean, std)

    train_digits = train_attrs[:, digit_col].astype(int)
    val_digits = val_attrs[:, digit_col].astype(int)

    trees = {}
    train_self_nn = {}
    val_vs_train_nn = {}
    for d in range(10):
        pool = train_feat_all[train_digits == d]
        tree = cKDTree(pool)
        trees[d] = tree
        # leave-one-out: k=2, the nearest neighbor other than the point itself
        dists, _ = tree.query(pool, k=2)
        train_self_nn[d] = dists[:, 1]
        v_pool = val_feat_all[val_digits == d]
        if len(v_pool) > 0:
            vdists, _ = tree.query(v_pool, k=1)
            val_vs_train_nn[d] = vdists
        else:
            val_vs_train_nn[d] = np.array([])

    cohorts = json.loads(Path(COHORTS).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"]
    eval_attrs = ds.attrs[fixed_indices]
    eval_feat, _, _ = build_feature_matrix(eval_attrs, names, mean, std)
    eval_digits = eval_attrs[:, digit_col].astype(int)
    target_digits = (eval_digits + 10 // 2) % 10  # matches the shift used everywhere else in eval

    cf_nn_dist = np.zeros(len(fixed_indices))
    for i in range(len(fixed_indices)):
        d = int(target_digits[i])
        dist, _ = trees[d].query(eval_feat[i], k=1)
        cf_nn_dist[i] = dist

    def pct(arr, qs=(50, 75, 90, 95, 99)):
        if len(arr) == 0:
            return {}
        return {str(q): float(np.percentile(arr, q)) for q in qs}

    all_train_self = np.concatenate(list(train_self_nn.values()))
    all_val_vs_train = np.concatenate([a for a in val_vs_train_nn.values() if len(a) > 0])

    print("=== baseline: train-self NN distance (typical spacing between real, seen images, per class) ===")
    print(f"  overall: {pct(all_train_self)}")
    print("\n=== baseline: val-vs-train NN distance (real held-out image -> nearest train image, per class) ===")
    print(f"  overall: {pct(all_val_vs_train)}")
    print("\n=== digit-intervention counterfactual targets: NN distance to nearest train image of the TARGET class ===")
    print(f"  overall: {pct(cf_nn_dist)}")

    # how does each cf target's distance compare to the val-vs-train distribution for ITS target class?
    frac_within_val_gap = []
    ratio_to_val_median = []
    for i in range(len(fixed_indices)):
        d = int(target_digits[i])
        baseline = val_vs_train_nn[d]
        if len(baseline) == 0:
            continue
        frac_within_val_gap.append(cf_nn_dist[i] <= np.percentile(baseline, 95))
        ratio_to_val_median.append(cf_nn_dist[i] / (np.median(baseline) + 1e-8))
    frac_within_val_gap = np.array(frac_within_val_gap)
    ratio_to_val_median = np.array(ratio_to_val_median)

    print(f"\n{frac_within_val_gap.sum()}/{len(frac_within_val_gap)} "
          f"({frac_within_val_gap.mean()*100:.1f}%) of digit counterfactual targets have a NN "
          f"distance <= the 95th percentile of real held-out images' NN distance to train "
          f"(their target class's normal generalization gap).")
    print("Distribution of (cf target NN dist) / (median real val-vs-train NN dist for that target class):")
    print(f"  {pct(ratio_to_val_median)}")
    print("  (ratio ~1 means the cf target is about as far from the nearest real training image")
    print("   as a genuine, never-seen validation image of that class typically is -- i.e. normal")
    print("   generalization territory. Ratio >>1 means the cf target is further from anything")
    print("   trained on than real unseen data ever is.)")

    per_class_detail = {}
    for d in range(10):
        mask = target_digits == d
        if mask.sum() == 0:
            continue
        per_class_detail[str(d)] = {
            "n_targets": int(mask.sum()),
            "cf_nn_dist_median": float(np.median(cf_nn_dist[mask])),
            "train_self_nn_median": float(np.median(train_self_nn[d])),
            "val_vs_train_nn_median": float(np.median(val_vs_train_nn[d])) if len(val_vs_train_nn[d]) else None,
            "n_train_in_class": int((train_digits == d).sum()),
        }

    out = {
        "train_self_nn_pct": pct(all_train_self),
        "val_vs_train_nn_pct": pct(all_val_vs_train),
        "cf_target_nn_pct": pct(cf_nn_dist),
        "frac_within_val_95th_gap": float(frac_within_val_gap.mean()),
        "ratio_to_val_median_pct": pct(ratio_to_val_median),
        "per_target_class": per_class_detail,
        "n_targets": len(fixed_indices),
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
