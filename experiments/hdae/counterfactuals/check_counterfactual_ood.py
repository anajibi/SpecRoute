#!/usr/bin/env python
"""For every counterfactual target vector actually fed to the models during eval (all 4
intervention types, full 512-image cohort), check every one of the 14 logged parameters
against the TRAIN partition's observed range. This is model-agnostic -- the target vectors
come from the SCM + cohorts file, identical for every model (k1/k5/k11), so this only needs
to run once, not per-model.

"OOD" here means: does the value fall outside [train_min, train_max] for that attribute --
i.e. would the model need to extrapolate beyond anything it saw during training. This is the
literal, checkable definition; exact joint-density estimation over 14 dimensions from ~56k
training images isn't feasible (curse of dimensionality means almost any full combination is
technically "unseen" in that stricter sense), so per-attribute marginal range is the
meaningful, answerable version of the question.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

PACKED = "experiments/hdae/data/packed/morphomnist_70k.h5"
CAUSAL_GRAPH = "experiments/hdae/configs/causal_graph_morpho.yaml"
COHORTS = "experiments/hdae/outputs/intervention_cohorts.json"
# slant is constant (0.0) by dataset design -- never OOD. texture_seed is an arbitrary RNG
# seed (range ~3e4 to ~2e9), not a semantic parameter -- "OOD" isn't a meaningful concept for
# it (changing the seed doesn't move the image toward or away from anything). Both excluded
# from the check, not silently -- noted here.
SKIP_ATTRS = {"slant", "texture_seed"}


def main():
    device = "cpu"
    ds = MorphoMNISTPacked(PACKED)
    with open(CAUSAL_GRAPH) as f:
        causal_raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(causal_raw)
    scm = SCM.load(causal_raw["scm_checkpoint"], device=device)

    train_mask = ds.partitions == 0
    check_attrs = [a for a in ds.attribute_names if a not in SKIP_ATTRS]
    train_range = {}
    for name in check_attrs:
        col = ds.attribute_names.index(name)
        v = ds.attrs[train_mask, col]
        train_range[name] = (float(v.min()), float(v.max()))

    print("Train-partition range per checked attribute:")
    for name, (lo, hi) in train_range.items():
        print(f"  {name:>16s}: [{lo:.4f}, {hi:.4f}]")

    cohorts = json.loads(Path(COHORTS).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"]
    categorical_attrs = [a for a in graph.attributes if scm.specs[a].kind == "categorical"]
    continuous_attrs = ["thickness", "intensity"]
    target_by_index = {attr: {row["index"]: row["target_value"] for row in cohorts[attr]["per_image"]}
                       for attr in continuous_attrs}

    scm_cols = [ds.attribute_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}

    specs = [{"attr": a, "kind": "categorical", "num_classes": scm.specs[a].num_classes} for a in categorical_attrs]
    specs += [{"attr": a, "kind": "continuous"} for a in continuous_attrs]

    all_attrs_raw = torch.from_numpy(ds.attrs[fixed_indices].astype(np.float32))

    print(f"\nChecking {len(fixed_indices)} images x {len(specs)} intervention types "
         f"= {len(fixed_indices) * len(specs)} counterfactual vectors, {len(check_attrs)} attributes each.\n")

    overall_any_ood = 0
    overall_n = 0
    per_attr_ood_count = {name: 0 for name in check_attrs}
    per_intervention_summary = []

    for spec in specs:
        attr = spec["attr"]
        descendants, observed = sorted(graph.descendants(attr)), None
        observed = [a for a in graph.attributes if a != attr and a not in descendants]

        cur_raw = all_attrs_raw[:, scm_cols[scm_attr_index[attr]]]
        if spec["kind"] == "categorical":
            n_classes = spec["num_classes"]
            cur_class = scm.categorical_class_index(attr, cur_raw)
            target_class = (cur_class + n_classes // 2) % n_classes
            target_tensor = scm.class_index_to_raw(attr, target_class).view(-1, 1)
        else:
            target_vals = [target_by_index[attr][int(i)] for i in fixed_indices]
            target_tensor = torch.tensor(target_vals).float().view(-1, 1)

        cf_attrs = scm.counterfactual(all_attrs_raw[:, scm_cols].float(), scm_attr_index,
                                      interventions={attr: target_tensor})
        for a in observed:
            cf_attrs[a] = all_attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()

        # Full per-image vector for every checked attribute, whether modeled (digit/thickness/
        # intensity/hue) or unmodeled (rotation/scale/.../texture_amplitude -- untouched, still
        # the image's real original value, since interventions never touch unmodeled attributes).
        full_vec = {}
        for name in check_attrs:
            if name in graph.attributes:
                full_vec[name] = cf_attrs[name].detach().cpu().numpy().reshape(-1)
            else:
                col = ds.attribute_names.index(name)
                full_vec[name] = all_attrs_raw[:, col].numpy()

        n_images = len(fixed_indices)
        any_ood = np.zeros(n_images, dtype=bool)
        attr_ood_this_intervention = {}
        for name in check_attrs:
            lo, hi = train_range[name]
            ood = (full_vec[name] < lo) | (full_vec[name] > hi)
            attr_ood_this_intervention[name] = int(ood.sum())
            per_attr_ood_count[name] += int(ood.sum())
            any_ood |= ood

        overall_any_ood += int(any_ood.sum())
        overall_n += n_images
        per_intervention_summary.append({
            "intervention": attr, "n_images": n_images, "n_any_ood": int(any_ood.sum()),
            "frac_any_ood": float(any_ood.mean()),
            "per_attr_ood_count": attr_ood_this_intervention,
        })
        print(f"intervene({attr}): {int(any_ood.sum())}/{n_images} images have >=1 OOD attribute "
             f"({any_ood.mean()*100:.1f}%)")
        nonzero = {k: v for k, v in attr_ood_this_intervention.items() if v > 0}
        if nonzero:
            print(f"  OOD counts by attribute: {nonzero}")
        else:
            print("  no attribute went OOD for any image")

    print(f"\n=== overall ===")
    print(f"{overall_any_ood}/{overall_n} counterfactual vectors ({overall_any_ood/overall_n*100:.1f}%) "
         f"have at least one attribute outside its train-partition range.")
    print("Per-attribute total OOD count across all 4 intervention types:")
    for name, cnt in sorted(per_attr_ood_count.items(), key=lambda kv: -kv[1]):
        if cnt > 0:
            print(f"  {name:>16s}: {cnt} ({cnt/overall_n*100:.2f}%)")

    out = {"per_intervention": per_intervention_summary, "overall_any_ood": overall_any_ood,
          "overall_n": overall_n, "per_attr_total_ood_count": per_attr_ood_count,
          "train_range": train_range, "skipped_attrs": sorted(SKIP_ATTRS)}
    out_path = "experiments/hdae/outputs/counterfactual_ood_check.json"
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
