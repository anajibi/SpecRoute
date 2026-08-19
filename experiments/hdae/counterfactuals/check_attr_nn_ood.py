#!/usr/bin/env python
"""T13: extend the digit nearest-neighbor OOD check (check_digit_nn_ood.py) to hue, thickness,
and intensity. Same question as before: for each counterfactual target, does a training image
exist (within the same digit class, since digit is never intervened by these three types) with a
similar combination of the OTHER attributes?

hue is a causal-graph root like digit -- the intervention changes only hue, everything else
(including digit) stays at the image's own original value. thickness is a parent of intensity
(the one declared causal edge) -- a thickness intervention also changes intensity via the SCM's
propagated counterfactual value, so the feature vector used for the NN search must reflect that
propagated intensity, not the original. intensity has no children, so an intensity intervention
changes nothing else.

Because digit is never intervened by any of these three, the natural pool restriction is "train
images of the SAME digit class as the source image" (not a target class, as in the digit script)
-- this keeps the comparison meaningful (thickness/intensity/hue combinations are being compared
within the same visual identity), matching the digit script's logic that comparisons should
respect what a human/model would consider "similar".
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

PACKED = "experiments/hdae/data/packed/morphomnist_70k.h5"
CAUSAL_GRAPH = "experiments/hdae/configs/causal_graph_morpho.yaml"
COHORTS = "experiments/hdae/outputs/intervention_cohorts.json"
OUT_PATH = "experiments/hdae/outputs/attr_nn_ood_check.json"

LINEAR_ATTRS = ["thickness", "intensity", "hue", "rotation", "scale",
                "translate_x", "translate_y", "bg_freq", "bg_amplitude", "texture_amplitude"]
CIRCULAR_ATTR = "bg_phase"


def build_feature_matrix(attrs, names, mean, std):
    cols = [attrs[:, names.index(name)] for name in LINEAR_ATTRS]
    phase = attrs[:, names.index(CIRCULAR_ATTR)]
    cols.append(np.cos(phase))
    cols.append(np.sin(phase))
    mat = np.stack(cols, axis=1)
    if mean is None:
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        std[std < 1e-8] = 1.0
    return (mat - mean) / std, mean, std


def pct(arr, qs=(50, 75, 90, 95, 99)):
    if len(arr) == 0:
        return {}
    return {str(q): float(np.percentile(arr, q)) for q in qs}


def main():
    ds = MorphoMNISTPacked(PACKED)
    names = ds.attribute_names
    digit_col = names.index("digit")
    with open(CAUSAL_GRAPH) as f:
        causal_raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(causal_raw)
    scm = SCM.load(causal_raw["scm_checkpoint"], device="cpu")

    train_mask = ds.partitions == 0
    val_mask = ds.partitions == 1
    train_attrs = ds.attrs[train_mask]
    val_attrs = ds.attrs[val_mask]
    train_feat_all, mean, std = build_feature_matrix(train_attrs, names, None, None)
    val_feat_all, _, _ = build_feature_matrix(val_attrs, names, mean, std)
    train_digits = train_attrs[:, digit_col].astype(int)
    val_digits = val_attrs[:, digit_col].astype(int)

    trees, train_self_nn, val_vs_train_nn = {}, {}, {}
    for d in range(10):
        pool = train_feat_all[train_digits == d]
        tree = cKDTree(pool)
        trees[d] = tree
        dists, _ = tree.query(pool, k=2)
        train_self_nn[d] = dists[:, 1]
        v_pool = val_feat_all[val_digits == d]
        val_vs_train_nn[d] = tree.query(v_pool, k=1)[0] if len(v_pool) else np.array([])

    all_train_self = np.concatenate(list(train_self_nn.values()))
    all_val_vs_train = np.concatenate([a for a in val_vs_train_nn.values() if len(a) > 0])

    cohorts = json.loads(Path(COHORTS).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"]
    eval_attrs_np = ds.attrs[fixed_indices]
    eval_digits = eval_attrs_np[:, digit_col].astype(int)
    scm_cols = [names.index(a) for a in graph.attributes]
    scm_attr_index = {n: i for i, n in enumerate(graph.attributes)}
    eval_attrs_t = torch.from_numpy(eval_attrs_np.astype(np.float32))

    continuous_attrs = ["thickness", "intensity"]
    target_by_index = {attr: {row["index"]: row["target_value"] for row in cohorts[attr]["per_image"]}
                       for attr in continuous_attrs}

    results = {}
    for attr in ["hue", "thickness", "intensity"]:
        cf_full = eval_attrs_np.copy()  # start from original, overwrite what changes
        if attr == "hue":
            n_classes = scm.specs["hue"].num_classes
            cur_raw = eval_attrs_t[:, scm_cols[scm_attr_index["hue"]]]
            cur_class = scm.categorical_class_index("hue", cur_raw)
            target_class = (cur_class + n_classes // 2) % n_classes
            target_raw = scm.class_index_to_raw("hue", target_class).numpy()
            cf_full[:, names.index("hue")] = target_raw
        else:
            target_vals = np.array([target_by_index[attr][int(i)] for i in fixed_indices])
            target_tensor = torch.tensor(target_vals).float().view(-1, 1)
            cf_attrs = scm.counterfactual(eval_attrs_t[:, scm_cols].float(), scm_attr_index,
                                          interventions={attr: target_tensor})
            for a in graph.attributes:
                cf_full[:, names.index(a)] = cf_attrs[a].detach().cpu().numpy().reshape(-1)

        cf_feat, _, _ = build_feature_matrix(cf_full, names, mean, std)
        pool_digits = eval_digits  # digit never changes for these 3 intervention types

        cf_nn_dist = np.zeros(len(fixed_indices))
        for i in range(len(fixed_indices)):
            d = int(pool_digits[i])
            dist, _ = trees[d].query(cf_feat[i], k=1)
            cf_nn_dist[i] = dist

        frac_within_val_gap, ratio_to_val_median = [], []
        for i in range(len(fixed_indices)):
            baseline = val_vs_train_nn[int(pool_digits[i])]
            if len(baseline) == 0:
                continue
            frac_within_val_gap.append(cf_nn_dist[i] <= np.percentile(baseline, 95))
            ratio_to_val_median.append(cf_nn_dist[i] / (np.median(baseline) + 1e-8))
        frac_within_val_gap = np.array(frac_within_val_gap)
        ratio_to_val_median = np.array(ratio_to_val_median)

        results[attr] = {
            "cf_target_nn_pct": pct(cf_nn_dist),
            "frac_within_val_95th_gap": float(frac_within_val_gap.mean()),
            "ratio_to_val_median_pct": pct(ratio_to_val_median),
            "n_targets": len(fixed_indices),
        }
        print(f"=== {attr} ===")
        print(f"  {frac_within_val_gap.sum()}/{len(frac_within_val_gap)} "
              f"({frac_within_val_gap.mean()*100:.1f}%) within the 95th pct of real val-vs-train NN dist")
        print(f"  ratio to val median: {pct(ratio_to_val_median)}")

    out = {
        "train_self_nn_pct": pct(all_train_self),
        "val_vs_train_nn_pct": pct(all_val_vs_train),
        "per_attribute": results,
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
