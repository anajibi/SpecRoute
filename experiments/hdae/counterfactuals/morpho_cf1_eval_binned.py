#!/usr/bin/env python
"""CC / FC_observed / FC_unobserved / CF1 using the precomputed bin-based flip cohorts
(precompute_intervention_cohorts.py) instead of morpho_cf1_eval.py's fixed p15/p85 targets.

Same aggregation math (imported from run_cf1_eval.py), same success/tolerance criteria, same
attribute-partition logic as morpho_cf1_eval.py -- the only change is where the per-attribute
intervention target comes from: looked up per-image from intervention_cohorts.json (opposite
population half, >=2 bins away, target = bin midpoint) rather than one fixed value shared by a
whole cohort. 4 intervention cells (digit shift, thickness/intensity/hue flip), not 7, since each
continuous attribute now has exactly one flip direction per image (already determined by the
cohort file), not a separate high/low pair.
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
    CIRCULAR, UNMODELED_ATTRS, attribute_partition, circular_diff, compute_train_stats,
    fc_for_pool_mixed, load_all_predictors, predict_all)
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.morphomnist import MorphoMNISTPacked


def evaluate_one_intervention_binned(adapter, scm, graph, predictors, spec, ds, indices, attr_names,
                                     std_by_attr, batch_size, device, T, target_by_index=None,
                                     categorical_top_k=1):
    scm_cols = [attr_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}
    descendants, observed = attribute_partition(graph, spec["attr"])
    # categorical-ness comes from the fitted SCM's own node specs (authoritative -- digit and now
    # hue are both categorical root nodes; previously only digit was, back when hue was
    # continuous), not a hardcoded attribute name.
    observed_kinds = [(a, "categorical" if scm.specs[a].kind == "categorical" else "continuous") for a in observed]
    unobs_kinds = [(a, "continuous") for a in UNMODELED_ATTRS]

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
            # raw -> class index -> shift -> back to a valid raw-units target. For digit, raw
            # value already IS the class index (scm.categorical_class_index is then a no-op
            # cast); for hue, raw value is a bin-center float (e.g. 0.05-0.95) that must be
            # binned via lo/hi first -- naive `.long()` truncation collapses nearly all hue
            # values to class 0 and produces an out-of-[0,1]-range target (found the hard way).
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
            if spec["attr"] in CIRCULAR:
                err0 = circular_diff(pred0[spec["attr"]], target_np, CIRCULAR[spec["attr"]])
            else:
                err0 = np.abs(pred0[spec["attr"]] - target_np)
            valid = err0 > spec["tolerance"]

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
            if categorical_top_k > 1:
                # "success" = target class is among the predictor's top-k guesses, not only its
                # single best one -- decouples the *predictor's* own top-1 error (digit's CNN
                # isn't 100% accurate either) from what's actually being measured here (did the
                # generative model's intervention work). Explicit, reported tolerance, not a
                # silently softened metric -- categorical_top_k=1 (default) reproduces the exact
                # prior behavior bit-for-bit.
                topk = predictors[spec["attr"]].predict_topk_classes(cf * 2 - 1, k=categorical_top_k).numpy()
                success = valid & (topk == target_np[:, None]).any(axis=1)
            else:
                pred_cf_cls = scm.categorical_class_index(
                    spec["attr"], torch.from_numpy(pred_cf[spec["attr"]]).to(device)).cpu().numpy()
                success = valid & (pred_cf_cls == target_np)
        else:
            if spec["attr"] in CIRCULAR:
                err_cf = circular_diff(pred_cf[spec["attr"]], target_np, CIRCULAR[spec["attr"]])
            else:
                err_cf = np.abs(pred_cf[spec["attr"]] - target_np)
            success = valid & (err_cf <= spec["tolerance"])

        all_valid.append(valid)
        all_success.append(success)

        for d in descendants:
            scm_target_d = cf_attrs[d].detach().cpu().numpy().reshape(-1)
            if d in CIRCULAR:
                derr = circular_diff(pred_cf[d], scm_target_d, CIRCULAR[d])
            else:
                derr = np.abs(pred_cf[d] - scm_target_d)
            tol_d = std_by_attr.get(d, 1.0) * 0.25
            desc_success_sum += int((valid & (derr <= tol_d)).sum())
            desc_valid_count += int(valid.sum())

        fail = valid & ~success
        fc_s, _ = fc_for_pool_mixed(pred0, pred_cf, success, observed_kinds, std_by_attr, scm=scm)
        fc_f, _ = fc_for_pool_mixed(pred0, pred_cf, fail, observed_kinds, std_by_attr, scm=scm)
        fc_s_u, _ = fc_for_pool_mixed(pred0_unmodeled, pred_cf_unmodeled, success, unobs_kinds, std_by_attr, scm=scm)
        fc_f_u, _ = fc_for_pool_mixed(pred0_unmodeled, pred_cf_unmodeled, fail, unobs_kinds, std_by_attr, scm=scm)
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
           "unobserved_attrs": ";".join(UNMODELED_ATTRS), "n_unobserved_attrs": len(UNMODELED_ATTRS),
           "CC": cc, "cc_numerator": cc_num, "cc_denominator": cc_den,
           "FC_success_observed": fc_s_observed, "FC_fail_observed": fc_f_observed,
           "FC_success_unobserved": fc_s_unobserved, "FC_fail_unobserved": fc_f_unobserved,
           "CF1_observed": cf1(cc, fc_s_observed), "CF1_unobserved": cf1(cc, fc_s_unobserved),
           "n_valid": int(valid_arr.sum()), "n_success": int(success_arr.sum()),
           "cohort_n": len(indices)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph_morpho.yaml")
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist.h5")
    p.add_argument("--predictors-dir", default="experiments/hdae/outputs/attr_predictors")
    p.add_argument("--cohorts", default="experiments/hdae/outputs/intervention_cohorts.json")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--edit-strength", type=float, required=True)
    p.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--categorical-top-k", type=int, default=1, help="success for a categorical "
                   "intervention (currently digit) counts if the target class is in the predictor's "
                   "top-k, not just its top-1 guess -- decouples the predictor's own error from CC/FC. "
                   "1 reproduces the original exact-match behavior exactly.")
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
                           T=args.T, compile_model=args.compile_model)
    if set(adapter.modeled_attrs) != set(graph.attributes):
        raise ValueError(f"adapter.modeled_attrs={adapter.modeled_attrs} != graph.attributes={graph.attributes}")

    ds = MorphoMNISTPacked(args.packed)
    predictors = load_all_predictors(args.predictors_dir, ds)
    comp = json.loads((Path(args.predictors_dir) / "comparison_results.json").read_text())["per_attribute"]
    # thickness/intensity: still continuous, bin-flip cohort targets as before. hue: now
    # categorical (2026-08-11) -- same shift-based spec as digit, not a cohort-driven flip; both
    # its tolerance machinery (train percentiles, cnn_mae) and its cohort entry no longer apply.
    continuous_attrs = ["thickness", "intensity"]
    categorical_attrs = [a for a in graph.attributes if scm.specs[a].kind == "categorical"]
    tolerances_raw = {name: comp[name]["cnn_mae"] for name in continuous_attrs}
    train_stats = compute_train_stats(ds, continuous_attrs)
    std_by_attr = {name: train_stats[name]["std"] for name in train_stats}
    for name in UNMODELED_ATTRS:
        col = ds.attribute_names.index(name)
        std_by_attr[name] = float(ds.attrs[ds.partitions == 0, col].std())

    cohorts = json.loads(Path(args.cohorts).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"]
    target_by_index = {attr: {row["index"]: row["target_value"] for row in cohorts[attr]["per_image"]}
                       for attr in continuous_attrs}

    specs = [{"attr": a, "direction": "shift", "kind": "categorical", "num_classes": scm.specs[a].num_classes}
            for a in categorical_attrs]
    for attr in continuous_attrs:
        gap = abs(train_stats[attr]["p85"] - train_stats[attr]["p15"])
        tol = max(tolerances_raw[attr], 0.05 * gap)
        specs.append({"attr": attr, "direction": "flip_binned", "kind": "continuous", "tolerance": tol})

    print(f"fixed index set: n={len(fixed_indices)} (from {args.cohorts})")
    rows = []
    for spec in specs:
        descendants, observed = attribute_partition(graph, spec["attr"])
        print(f"  intervene({spec['attr']}, {spec['direction']}): CC pool={{{spec['attr']}}}+{descendants} "
             f"FC_obs={observed} FC_unobs(9 attrs)")
        r = evaluate_one_intervention_binned(adapter, scm, graph, predictors, spec, ds, fixed_indices,
                                             ds.attribute_names, std_by_attr, args.batch_size, device,
                                             args.T, target_by_index, categorical_top_k=args.categorical_top_k)
        r.update({"model": args.model_name, "edit_strength": args.edit_strength,
                 "categorical_top_k": args.categorical_top_k})
        rows.append(r)
        print(f"    -> CC={r['CC']:.4f} FC_obs={r['FC_success_observed']:.4f} "
             f"FC_unobs={r['FC_success_unobserved']:.4f} CF1_obs={r['CF1_observed']:.4f} "
             f"CF1_unobs={r['CF1_unobserved']:.4f} n_valid={r['n_valid']}/{r['cohort_n']}")

    with open(out / "morpho_cf1_binned_per_intervention.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    macro_observed = float(np.mean([r["CF1_observed"] for r in rows]))
    macro_unobserved = float(np.mean([r["CF1_unobserved"] for r in rows]))
    w_sum = sum(r["n_valid"] for r in rows) or 1
    weighted_observed = float(sum(r["CF1_observed"] * r["n_valid"] for r in rows) / w_sum)
    weighted_unobserved = float(sum(r["CF1_unobserved"] * r["n_valid"] for r in rows) / w_sum)
    global_cc = sum(r["cc_numerator"] for r in rows) / (sum(r["cc_denominator"] for r in rows) or 1)

    agg = {"model": args.model_name, "edit_strength": args.edit_strength,
          "macro_CF1_observed": macro_observed, "macro_CF1_unobserved": macro_unobserved,
          "weighted_CF1_observed": weighted_observed, "weighted_CF1_unobserved": weighted_unobserved,
          "global_CC": global_cc, "n_interventions": len(rows), "n_images": len(fixed_indices)}
    with open(out / "morpho_cf1_binned_aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)
    print("\n=== aggregate ===")
    print(json.dumps(agg, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
