#!/usr/bin/env python
"""CC / FC_observed / FC_unobserved / CF1 for MorphoMNIST HDAE, at a given guidance scale.

Reuses the exact aggregation math from run_cf1_eval.py (`cf1`, `pareto_frontier`,
`frontier_area`, imported not reimplemented) -- only the per-attribute *success/flip
criterion* is new, because run_cf1_eval.py's is boolean-threshold-only (built for CelebA's 40
binary attributes) and MorphoMNIST's 4 conditioning attributes are categorical (digit) or
continuous (thickness/intensity/hue).

Intervention design (7 cells total, over the same 4 attributes the user asked about):
- digit: forced shift to (current_digit + 5) mod 10 -- one cell, categorical has no
  continuous "direction".
- thickness / intensity / hue: two cells each, "low" and "high" -- a FIXED target per
  direction (the 15th/85th percentile of that attribute on the TRAIN partition), applied to
  the cohort of images whose own (recon0-predicted) value is currently on the opposite side
  of the train-partition median. Fixed target + cohort-selected source mirrors CelebA's
  positive/negative cohort framing (same target for every image in a cohort, only the source
  set differs) -- an earlier version of this script used a per-image-adaptive target instead,
  which correlates the target with the starting value and makes "success" a moving goalpost;
  fixed here after review. The earlier smoke test's low=1.5/high=6.5 (thickness) and
  low=90/high=230 (intensity) targets were also replaced -- both were found to be out of
  distribution (thickness high sat above the p99, intensity low sat at the observed minimum)
  when the empirical distributions were plotted; p15/p85 are solidly in-distribution.

Success/valid criterion, per attribute kind:
- categorical (digit): valid = recon0's predicted class != target class; success = valid AND
  the counterfactual's predicted class == target class. No tolerance (exact class match).
- continuous (thickness/intensity/hue): tolerance = that attribute's own CNN predictor's
  test-set MAE (data/evaluate_attr_predictors.py's comparison_results.json) -- the
  predictor's own measurement noise floor, not an arbitrary number. valid = |recon0 pred -
  target| > tolerance; success = valid AND |cf pred - target| <= tolerance. hue uses circular
  error (wraparound-aware), matching its predictor's own training objective.

CC (Counterfactual Consistency): success rate on {intervened attribute} union its causal
descendants (only thickness has one: intensity) -- descendant "success" = its counterfactual
prediction lands within tolerance of the SCM's own propagated target for it, among images
where the parent's intervention was valid.

FC (Factual Consistency), two disjoint pools:
- FC_observed: the other 3 conditioning attributes that are NOT causal descendants of the
  intervened one (note: intervening on `intensity` leaves `thickness` in this pool --
  do()-interventions don't propagate to ancestors, only descendants).
- FC_unobserved: the 9 non-conditioning MorphoMNIST factors (slant, rotation, scale,
  translate_x/y, bg_freq/phase/amplitude, texture_amplitude) -- outside the causal graph
  entirely, same role as CelebA's other-36-attributes pool.
For a categorical column in a pool: flip = predicted class changed (boolean), same as
CelebA. For a continuous column: drift = clip(|cf pred - recon0 pred| / population std, 0, 1)
-- capped so one badly-behaved column can't make FC go negative. FC = 1 - mean(pool values),
computed separately over the success-mask and fail-mask populations (only FC on successes
feeds CF1, matching run_cf1_eval.py; FC on failures is reported as an extra diagnostic).

CF1 = harmonic mean of CC and FC (`cf1()`, imported), separately for each pool.
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
from experiments.hdae.counterfactuals.run_cf1_eval import cf1, frontier_area, pareto_frontier
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.attr_predictor import load_attr_predictor
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

UNMODELED_ATTRS = ["rotation", "scale", "translate_x", "translate_y",
                   "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude"]
# slant removed (2026-08-11) -- morphomnist_factors.yaml now fixes it at a constant, so there's no
# predictor for it and nothing for the FC_unobserved pool to check.
CIRCULAR = {"hue": 1.0}


def circular_diff(a, b, period):
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def load_all_predictors(predictors_dir, ds):
    summary = json.loads((Path(predictors_dir) / "training_summary.json").read_text())
    out = {}
    for name, info in summary.items():
        out[name] = load_attr_predictor(info["checkpoint"], attr_col=ds.attribute_names.index(name))
    return out


def predict_all(predictors, img_m11, device):
    x = img_m11.to(device)
    return {name: model.predict_raw(x).numpy() for name, model in predictors.items()}


def compute_train_stats(ds, attrs_of_interest):
    train_mask = ds.partitions == 0
    stats = {}
    for name in attrs_of_interest:
        col = ds.attribute_names.index(name)
        vals = ds.attrs[train_mask, col]
        stats[name] = {"median": float(np.median(vals)), "p15": float(np.percentile(vals, 15)),
                       "p85": float(np.percentile(vals, 85)), "std": float(vals.std())}
    return stats


def build_intervention_specs(graph, train_stats, tolerances):
    specs = []
    specs.append({"attr": "digit", "direction": "shift", "kind": "categorical"})
    for attr in ["thickness", "intensity", "hue"]:
        s = train_stats[attr]
        gap = abs(s["p85"] - s["p15"])
        # Floor at 5% of the p15-p85 gap: a predictor's raw test MAE is a fine tolerance when
        # it's a meaningful measurement floor (thickness/intensity), but hue's predictor is so
        # accurate (MAE ~0.0007 on a [0,1] scale) that using it directly makes "success" require
        # landing within a fraction of a percent of the target -- effectively unreachable for any
        # generative sampler, not a real signal about hue control quality. The floor only binds
        # for hue here; thickness/intensity are unaffected (their MAE already exceeds 5% of gap).
        tol = max(tolerances[attr], 0.05 * gap)
        for direction, target in [("high", s["p85"]), ("low", s["p15"])]:
            specs.append({"attr": attr, "direction": direction, "kind": "continuous",
                         "target": target, "median": s["median"], "std": s["std"],
                         "tolerance": tol})
    return specs


def attribute_partition(graph, attr):
    descendants = sorted(graph.descendants(attr))
    observed = [a for a in graph.attributes if a != attr and a not in descendants]
    return descendants, observed


def continuous_flip_or_drift(pred_before, pred_after, std):
    return np.clip(np.abs(pred_after - pred_before) / max(std, 1e-6), 0.0, 1.0)


def fc_for_pool_mixed(pred0, pred_cf, mask, cols_kinds, std_by_attr, scm=None):
    """cols_kinds: list of (name, 'categorical'|'continuous'). Mirrors run_cf1_eval.fc_for_pool's
    contract (mean-flip-derived FC = 1 - mean flip/drift) but per-column-kind, since MorphoMNIST
    pools mix categorical columns (digit, hue) with continuous ones.

    `scm` (required whenever any column is categorical) supplies the raw-value -> class-index
    conversion -- naive round()/clip(0,9) only happens to be correct for digit (raw value already
    IS the class index); hue's raw value is a bin-center float that needs lo/hi-based binning or
    it silently collapses to class 0/1 (see causal/scm.py's categorical_class_index docstring)."""
    if not cols_kinds:
        return 1.0, {}
    if not mask.any():
        return 1.0, {name: 0.0 for name, _ in cols_kinds}
    per_col = {}
    for name, kind in cols_kinds:
        b, e = pred0[name][mask], pred_cf[name][mask]
        if kind == "categorical":
            b_cls = scm.categorical_class_index(name, torch.from_numpy(b)).numpy()
            e_cls = scm.categorical_class_index(name, torch.from_numpy(e)).numpy()
            per_col[name] = float((b_cls != e_cls).mean())
        elif name in CIRCULAR:
            per_col[name] = float(np.clip(circular_diff(e, b, CIRCULAR[name]) / 0.5, 0, 1).mean())
        else:
            per_col[name] = float(continuous_flip_or_drift(b, e, std_by_attr[name]).mean())
    fc = float(1.0 - np.mean(list(per_col.values())))
    return fc, per_col


def evaluate_one_intervention(adapter, scm, graph, predictors, spec, ds, indices, attr_names,
                              std_by_attr, batch_size, device, T):
    scm_cols = [attr_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}
    descendants, observed = attribute_partition(graph, spec["attr"])
    observed_kinds = [(a, "categorical" if a == "digit" else "continuous") for a in observed]
    unobs_kinds = [(a, "continuous") for a in UNMODELED_ATTRS]

    all_valid, all_success = [], []
    desc_success_sum, desc_valid_count = 0, 0
    fc_s_obs_vals, fc_f_obs_vals = [], []
    fc_s_unobs_vals, fc_f_unobs_vals = [], []

    loader = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    for batch in loader:
        x = batch["img"].to(device)
        attrs_raw = batch["attr"].to(device)
        state = adapter.encode(x, attrs_raw, attr_names)
        recon0 = adapter.render(state)
        pred0 = predict_all(predictors, recon0 * 2 - 1, device)

        if spec["kind"] == "categorical":
            cur = attrs_raw[:, scm_cols[scm_attr_index["digit"]]]
            target = (cur.long() + 5) % 10
            target_tensor = target.float().view(-1, 1)
            target_np = target.detach().cpu().numpy()
            valid = np.round(pred0["digit"]).clip(0, 9) != target_np
        else:
            target_tensor = torch.full((x.shape[0], 1), spec["target"], device=device)
            target_np = np.full(x.shape[0], spec["target"])
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
            pred_cf_cls = np.round(pred_cf["digit"]).clip(0, 9)
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
            tol_d = spec.get("descendant_tolerance", {}).get(d, std_by_attr.get(d, 1.0) * 0.25)
            desc_success_sum += int((valid & (derr <= tol_d)).sum())
            desc_valid_count += int(valid.sum())

        fail = valid & ~success
        fc_s, _ = fc_for_pool_mixed(pred0, pred_cf, success, observed_kinds, std_by_attr)
        fc_f, _ = fc_for_pool_mixed(pred0, pred_cf, fail, observed_kinds, std_by_attr)
        fc_s_u, _ = fc_for_pool_mixed(pred0_unmodeled, pred_cf_unmodeled, success, unobs_kinds, std_by_attr)
        fc_f_u, _ = fc_for_pool_mixed(pred0_unmodeled, pred_cf_unmodeled, fail, unobs_kinds, std_by_attr)
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
    p.add_argument("--n-images", type=int, default=5000)
    p.add_argument("--index-seed", type=int, default=0, help="fixed seed -- same across all model/scale runs "
                   "so every combo scores the exact same 5000 images")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--edit-strength", type=float, required=True)
    p.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction, default=False)
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
    tolerances = {name: comp[name]["cnn_mae"] for name in ["thickness", "intensity", "hue"]}
    train_stats = compute_train_stats(ds, ["thickness", "intensity", "hue"])
    std_by_attr = {name: train_stats[name]["std"] for name in train_stats}
    for name in UNMODELED_ATTRS:
        col = ds.attribute_names.index(name)
        std_by_attr[name] = float(ds.attrs[ds.partitions == 0, col].std())

    specs = build_intervention_specs(graph, train_stats, tolerances)
    print("=== per-attribute tolerance / target sanity (post floor-at-5%-of-gap) ===")
    tol_used = {s["attr"]: s["tolerance"] for s in specs if s["kind"] == "continuous"}
    for name, s in train_stats.items():
        gap = abs(s["p85"] - s["p15"])
        print(f"  {name:<10} p15={s['p15']:.3f} p85={s['p85']:.3f} gap={gap:.3f} "
             f"raw_MAE={tolerances[name]:.4f} tolerance_used={tol_used[name]:.4f} "
             f"tolerance/gap={tol_used[name]/gap:.3f}")

    test_idx_all = np.nonzero(ds.partitions == 1)[0]
    rng = np.random.RandomState(args.index_seed)
    fixed_indices = rng.choice(test_idx_all, size=min(args.n_images, len(test_idx_all)), replace=False)
    fixed_indices.sort()
    print(f"fixed index set: n={len(fixed_indices)} seed={args.index_seed} "
         f"first5={fixed_indices[:5].tolist()} last5={fixed_indices[-5:].tolist()}")

    print("\n=== attribute partition per intervention ===")
    rows = []
    for spec in specs:
        descendants, observed = attribute_partition(graph, spec["attr"])
        print(f"  intervene({spec['attr']}, {spec['direction']}):")
        print(f"    CC pool           = {{{spec['attr']}}} + descendants {descendants} "
             f"[n={1 + len(descendants)}]")
        print(f"    FC_observed pool  = {observed} [n={len(observed)}]")
        print(f"    FC_unobserved pool= {UNMODELED_ATTRS} [n={len(UNMODELED_ATTRS)}]")
        r = evaluate_one_intervention(adapter, scm, graph, predictors, spec, ds, fixed_indices.tolist(),
                                      ds.attribute_names, std_by_attr, args.batch_size, device, args.T)
        r.update({"model": args.model_name, "edit_strength": args.edit_strength})
        rows.append(r)
        print(f"    -> CC={r['CC']:.4f} FC_obs={r['FC_success_observed']:.4f} "
             f"FC_unobs={r['FC_success_unobserved']:.4f} CF1_obs={r['CF1_observed']:.4f} "
             f"CF1_unobs={r['CF1_unobserved']:.4f} n_valid={r['n_valid']}/{r['cohort_n']}")

    with open(out / "morpho_cf1_per_intervention.csv", "w", newline="") as f:
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
    with open(out / "morpho_cf1_aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)
    print("\n=== aggregate ===")
    print(json.dumps(agg, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
