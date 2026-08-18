#!/usr/bin/env python
"""Variant of morpho_cf1_eval_binned.py with two tolerance changes, scoped exactly as requested:

1. thickness/intensity's CC target-check (and the `valid` gate) tolerance changes from
   max(cnn_mae, 5% of p15-p85 gap) to half the width of the image's own TARGET bin (bin_edges
   from intervention_cohorts.json are unequal-width quantile bins, so this is computed per-image,
   not one fixed number). digit/hue are UNCHANGED -- their existing exact-class-match (via
   scm.categorical_class_index) is mathematically identical to a half-bin-width raw-unit
   tolerance, since a predictor's categorical `predict_raw` always returns its predicted class's
   exact bin center, so neighboring classes are one full bin width apart and a half-bin tolerance
   can never admit a wrong class. Nothing to change there -- said explicitly here, not left
   implicit, since it's easy to misread "no change" as "forgot to change it."

2. FC_unobserved only: the soft std-normalized drift score (continuous_flip_or_drift) is replaced
   with a hard per-attribute gate, |after - before| <= tolerance_mult * cnn_mae(attr), run once for
   tolerance_mult=2 and once for tolerance_mult=3 (two separate result sets). FC_observed and the
   within-CC descendant-consistency check (thickness->intensity) are UNCHANGED -- both are "stay
   near a reference that isn't a cohort target_bin" cases with no natural bin-width tolerance to
   derive, and extending the redesign to them wasn't asked for.

Known, un-fixed limitation carried over from morpho_cf1_eval.py: bg_phase wraps at 2*pi but only
`hue` is in the CIRCULAR dict used here, so bg_phase's hard-gate comparison is wrap-unaware -- a
value near the 0/2*pi boundary can read as a large spurious drift. Not fixed here (would be a
silent methodology change beyond what was asked); flagged in the printed summary instead.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from experiments.hdae.counterfactuals import hdae_adapter  # noqa: F401 -- registers "hdae"
from experiments.hdae.counterfactuals.cf_contract import load_adapter
from experiments.hdae.counterfactuals.run_cf1_eval import cf1
from experiments.hdae.counterfactuals.morpho_cf1_eval import (
    UNMODELED_ATTRS, attribute_partition, compute_train_stats, fc_for_pool_mixed,
    load_all_predictors, predict_all)
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.morphomnist import MorphoMNISTPacked


def fc_for_pool_hard(pred0, pred_cf, mask, attrs, mae_by_attr, mult):
    """FC_unobserved only: hard per-attribute gate at mult * cnn_mae, mirrors fc_for_pool_mixed's
    contract (mean-flip-derived FC = 1 - mean flip fraction) but with a hard tolerance per column
    instead of a std-normalized soft score."""
    if not attrs:
        return 1.0, {}
    if not mask.any():
        return 1.0, {a: 0.0 for a in attrs}
    per_col = {}
    for a in attrs:
        b, e = pred0[a][mask], pred_cf[a][mask]
        tol = mae_by_attr[a] * mult
        per_col[a] = float((np.abs(e - b) > tol).mean())
    fc = float(1.0 - np.mean(list(per_col.values())))
    return fc, per_col


def evaluate_one_intervention(adapter, scm, graph, predictors, spec, ds, indices, attr_names,
                              std_by_attr, mae_by_attr, tol_mult, batch_size, device, T,
                              target_by_index=None, half_bin_tol_by_index=None):
    scm_cols = [attr_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}
    descendants, observed = attribute_partition(graph, spec["attr"])
    observed_kinds = [(a, "categorical" if scm.specs[a].kind == "categorical" else "continuous") for a in observed]

    all_valid, all_success = [], []
    desc_success_sum, desc_valid_count = 0, 0
    fc_s_obs_vals, fc_f_obs_vals = [], []
    fc_s_unobs_vals, fc_f_unobs_vals = [], []

    loader = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    for batch in loader:
        x = batch["img"].to(device)
        attrs_raw = batch["attr"].to(device)
        batch_idx = batch["index"].tolist()
        state = adapter.encode(x, attrs_raw, attr_names)
        recon0 = adapter.render(state)
        pred0 = predict_all(predictors, recon0 * 2 - 1, device)

        if spec["kind"] == "categorical":
            attr = spec["attr"]
            n_classes = spec["num_classes"]
            cur_raw = attrs_raw[:, scm_cols[scm_attr_index[attr]]]
            cur_class = scm.categorical_class_index(attr, cur_raw)
            target_class = (cur_class + n_classes // 2) % n_classes
            target_tensor = scm.class_index_to_raw(attr, target_class).view(-1, 1)
            target_np = target_class.detach().cpu().numpy()
            pred0_class = scm.categorical_class_index(attr, torch.from_numpy(pred0[attr]).to(device)).cpu().numpy()
            valid = pred0_class != target_np
        else:
            target_vals = [target_by_index[spec["attr"]][i] for i in batch_idx]
            target_tensor = torch.tensor(target_vals, device=device).float().view(-1, 1)
            target_np = np.array(target_vals)
            tol_np = np.array([half_bin_tol_by_index[spec["attr"]][i] for i in batch_idx])
            err0 = np.abs(pred0[spec["attr"]] - target_np)
            valid = err0 > tol_np

        cf_attrs = scm.counterfactual(attrs_raw[:, scm_cols].float(), scm_attr_index,
                                      interventions={spec["attr"]: target_tensor})
        for a in observed:
            cf_attrs[a] = attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()
        cf_state = adapter.intervene(state, spec["attr"], spec["direction"], cf_attrs)
        cf = adapter.render(cf_state)
        pred_cf = predict_all(predictors, cf * 2 - 1, device)
        pred_cf_unmodeled = predict_all({k: v for k, v in predictors.items() if k in UNMODELED_ATTRS},
                                        cf * 2 - 1, device)
        pred0_unmodeled = predict_all({k: v for k, v in predictors.items() if k in UNMODELED_ATTRS},
                                      recon0 * 2 - 1, device)

        if spec["kind"] == "categorical":
            pred_cf_cls = scm.categorical_class_index(
                spec["attr"], torch.from_numpy(pred_cf[spec["attr"]]).to(device)).cpu().numpy()
            success = valid & (pred_cf_cls == target_np)
        else:
            err_cf = np.abs(pred_cf[spec["attr"]] - target_np)
            success = valid & (err_cf <= tol_np)

        all_valid.append(valid)
        all_success.append(success)

        for d in descendants:
            scm_target_d = cf_attrs[d].detach().cpu().numpy().reshape(-1)
            derr = np.abs(pred_cf[d] - scm_target_d)
            tol_d = std_by_attr.get(d, 1.0) * 0.25  # unchanged -- see module docstring
            desc_success_sum += int((valid & (derr <= tol_d)).sum())
            desc_valid_count += int(valid.sum())

        fail = valid & ~success
        # FC_observed: unchanged soft mixed-kind score.
        fc_s, _ = fc_for_pool_mixed(pred0, pred_cf, success, observed_kinds, std_by_attr, scm=scm)
        fc_f, _ = fc_for_pool_mixed(pred0, pred_cf, fail, observed_kinds, std_by_attr, scm=scm)
        # FC_unobserved: new hard mult*cnn_mae gate.
        fc_s_u, _ = fc_for_pool_hard(pred0_unmodeled, pred_cf_unmodeled, success, UNMODELED_ATTRS, mae_by_attr, tol_mult)
        fc_f_u, _ = fc_for_pool_hard(pred0_unmodeled, pred_cf_unmodeled, fail, UNMODELED_ATTRS, mae_by_attr, tol_mult)
        fc_s_obs_vals.append((fc_s, int(success.sum())))
        fc_f_obs_vals.append((fc_f, int(fail.sum())))
        fc_s_unobs_vals.append((fc_s_u, int(success.sum())))
        fc_f_unobs_vals.append((fc_f_u, int(fail.sum())))

    valid_arr = np.concatenate(all_valid)
    success_arr = np.concatenate(all_success)
    cc_num = int(success_arr.sum()) + desc_success_sum
    cc_den = int(valid_arr.sum()) * (1 + len(descendants))
    cc = float(cc_num / cc_den) if cc_den else 0.0

    def weighted_mean(pairs):
        total_w = sum(w for _, w in pairs)
        return float(sum(v * w for v, w in pairs) / total_w) if total_w else 1.0

    fc_s_observed = weighted_mean(fc_s_obs_vals)
    fc_f_observed = weighted_mean(fc_f_obs_vals)
    fc_s_unobserved = weighted_mean(fc_s_unobs_vals)
    fc_f_unobserved = weighted_mean(fc_f_unobs_vals)

    return {"attribute": spec["attr"], "direction": spec["direction"],
           "descendants": ";".join(descendants), "n_descendants": len(descendants),
           "observed_attrs": ";".join(observed), "n_observed_attrs": len(observed),
           "CC": cc, "cc_numerator": cc_num, "cc_denominator": cc_den,
           "FC_observed": fc_s_observed, "FC_fail_observed": fc_f_observed,
           "FC_unobserved": fc_s_unobserved, "FC_fail_unobserved": fc_f_unobserved,
           "CF1_observed": cf1(cc, fc_s_observed), "CF1_unobserved": cf1(cc, fc_s_unobserved),
           "n_valid": int(valid_arr.sum()), "n_success": int(success_arr.sum()),
           "cohort_n": len(indices)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph_morpho.yaml")
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist_70k.h5")
    p.add_argument("--predictors-dir", default="experiments/hdae/outputs/attr_predictors_70k")
    p.add_argument("--cohorts", default="experiments/hdae/outputs/intervention_cohorts.json")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--edit-strength", type=float, required=True)
    p.add_argument("--unobserved-tolerance-mult", type=float, required=True,
                   help="FC_unobserved success gate = mult * that attribute's own predictor cnn_mae")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.causal_graph) as f:
        causal_raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(causal_raw)
    scm = SCM.load(causal_raw["scm_checkpoint"], device=device)

    adapter = load_adapter("hdae", args.config, args.ckpt, device, edit_strength=args.edit_strength,
                           T=args.T, compile_model=False)
    if set(adapter.modeled_attrs) != set(graph.attributes):
        raise ValueError(f"adapter.modeled_attrs={adapter.modeled_attrs} != graph.attributes={graph.attributes}")

    ds = MorphoMNISTPacked(args.packed)
    predictors = load_all_predictors(args.predictors_dir, ds)
    comp = json.loads((Path(args.predictors_dir) / "comparison_results.json").read_text())["per_attribute"]
    continuous_attrs = ["thickness", "intensity"]
    categorical_attrs = [a for a in graph.attributes if scm.specs[a].kind == "categorical"]
    train_stats = compute_train_stats(ds, continuous_attrs)
    std_by_attr = {name: train_stats[name]["std"] for name in train_stats}
    for name in UNMODELED_ATTRS:
        col = ds.attribute_names.index(name)
        std_by_attr[name] = float(ds.attrs[ds.partitions == 0, col].std())
    mae_by_attr = {name: comp[name]["cnn_mae"] for name in UNMODELED_ATTRS}

    cohorts = json.loads(Path(args.cohorts).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"]
    target_by_index = {attr: {row["index"]: row["target_value"] for row in cohorts[attr]["per_image"]}
                       for attr in continuous_attrs}
    # half the WIDTH of each image's own target bin (unequal-width quantile bins) -- per image,
    # not one fixed number for the whole attribute.
    half_bin_tol_by_index = {}
    for attr in continuous_attrs:
        edges = cohorts[attr]["bin_edges"]
        half_bin_tol_by_index[attr] = {
            row["index"]: (edges[row["target_bin"] + 1] - edges[row["target_bin"]]) / 2.0
            for row in cohorts[attr]["per_image"]
        }

    specs = [{"attr": a, "direction": "shift", "kind": "categorical", "num_classes": scm.specs[a].num_classes}
            for a in categorical_attrs]
    for attr in continuous_attrs:
        specs.append({"attr": attr, "direction": "flip_binned", "kind": "continuous"})

    print(f"fixed index set: n={len(fixed_indices)} (from {args.cohorts})")
    print(f"unobserved_tolerance_mult={args.unobserved_tolerance_mult} "
         f"(FC_unobserved gate = mult * cnn_mae per attribute)")
    print("NOTE: bg_phase wraps at 2*pi but is not in the circular-diff set used here -- its hard "
         "gate is wrap-unaware, see module docstring.")
    rows = []
    for spec in specs:
        descendants, observed = attribute_partition(graph, spec["attr"])
        print(f"  intervene({spec['attr']}, {spec['direction']}): CC pool={{{spec['attr']}}}+{descendants} "
             f"FC_obs={observed} FC_unobs(8 attrs, hard mult*mae gate)")
        r = evaluate_one_intervention(adapter, scm, graph, predictors, spec, ds, fixed_indices,
                                      ds.attribute_names, std_by_attr, mae_by_attr,
                                      args.unobserved_tolerance_mult, args.batch_size, device, args.T,
                                      target_by_index, half_bin_tol_by_index)
        r.update({"model": args.model_name, "edit_strength": args.edit_strength,
                 "unobserved_tolerance_mult": args.unobserved_tolerance_mult})
        rows.append(r)
        print(f"    -> CC={r['CC']:.4f} FC_obs={r['FC_observed']:.4f} "
             f"FC_unobs={r['FC_unobserved']:.4f} CF1_obs={r['CF1_observed']:.4f} "
             f"CF1_unobs={r['CF1_unobserved']:.4f} n_valid={r['n_valid']}/{r['cohort_n']}")

    with open(out / "per_intervention.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    macro_observed = float(np.mean([r["CF1_observed"] for r in rows]))
    macro_unobserved = float(np.mean([r["CF1_unobserved"] for r in rows]))
    w_sum = sum(r["n_valid"] for r in rows) or 1
    weighted_observed = float(sum(r["CF1_observed"] * r["n_valid"] for r in rows) / w_sum)
    weighted_unobserved = float(sum(r["CF1_unobserved"] * r["n_valid"] for r in rows) / w_sum)
    global_cc = sum(r["cc_numerator"] for r in rows) / (sum(r["cc_denominator"] for r in rows) or 1)
    global_fc_obs = float(np.mean([r["FC_observed"] for r in rows]))
    global_fc_unobs = float(np.mean([r["FC_unobserved"] for r in rows]))

    agg = {"model": args.model_name, "edit_strength": args.edit_strength,
          "unobserved_tolerance_mult": args.unobserved_tolerance_mult,
          "global_CC": global_cc, "global_FC_observed": global_fc_obs, "global_FC_unobserved": global_fc_unobs,
          "macro_CF1_observed": macro_observed, "macro_CF1_unobserved": macro_unobserved,
          "weighted_CF1_observed": weighted_observed, "weighted_CF1_unobserved": weighted_unobserved,
          "n_interventions": len(rows), "n_images": len(fixed_indices)}
    with open(out / "aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)
    print("\n=== aggregate ===")
    print(json.dumps(agg, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
