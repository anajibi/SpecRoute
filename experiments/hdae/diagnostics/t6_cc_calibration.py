#!/usr/bin/env python
"""T6: CC ceiling (TPR) and chance floor (FPR) calibration for the toleranced eval's CC metric.

TPR = P(scored pass | image truly AT target), measured on real held-out (val) images: predict
each image's own attributes with the CNN predictor, score predicted-vs-TRUE with the exact same
per-image half-bin tolerance the eval uses (population-quantile bins from intervention_cohorts
.json), per target bin, reweighted by the eval's ACTUAL target-bin distribution (from the fixed
512-image cohort) rather than the general population's bin distribution.

FPR = P(scored pass | image truly at the SOURCE value, target sampled as usual): for the fixed
512-image eval cohort itself, score the predictor's reading of each image's TRUE (unedited)
source value against that image's actual stored counterfactual target and tolerance.

c (true compliance) = (CC_obs - FPR) / (TPR - FPR), inverted per T6's spec (NOT the naive
CC/TPR ratio, which is biased and can exceed 1). Bootstrap CIs; identifiability guard when
TPR-FPR is small.

digit/hue (categorical, exact match): TPR = predictor's own held-out class accuracy (no
tolerance involved). FPR = P(predictor predicts the ACTUAL sampled target class | true class is
the source, source/target pairs as used in the real eval) -- for digit/hue this is the shift-by-
n/2 pattern used everywhere else in this investigation.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.attr_predictor import load_attr_predictor
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

PACKED = "experiments/hdae/data/packed/morphomnist_70k.h5"
COHORTS = "experiments/hdae/outputs/intervention_cohorts.json"
CAUSAL_GRAPH = "experiments/hdae/configs/causal_graph_morpho.yaml"
PREDICTORS_DIR = "experiments/hdae/outputs/attr_predictors_70k"
OUT_PATH = "experiments/hdae/outputs/diagnostics_t6_calibration.json"
N_TPR_SAMPLE = 3000
N_BOOTSTRAP = 500
CONTINUOUS = ["thickness", "intensity"]
CATEGORICAL = ["digit", "hue"]

# CC observed for k11 @ 45k tol2x, the most recent full eval (digit/hue exact-match unaffected
# by the tolerance redesign; thickness/intensity use this same eval's tolerance definition)
CC_OBS = {"digit": 0.4634, "hue": 0.9355, "thickness": 0.2604, "intensity": 0.4083}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MorphoMNISTPacked(PACKED)
    names = ds.attribute_names
    summary = json.loads((Path(PREDICTORS_DIR) / "training_summary.json").read_text())
    all_attrs = CONTINUOUS + CATEGORICAL
    predictors = {a: load_attr_predictor(summary[a]["checkpoint"], attr_col=names.index(a)).to(device)
                 for a in all_attrs}
    with open(CAUSAL_GRAPH) as f:
        causal_raw = yaml.safe_load(f)
    scm = SCM.load(causal_raw["scm_checkpoint"], device=device)

    cohorts = json.loads(Path(COHORTS).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"]

    val_mask = ds.partitions == 1
    val_idx_all = np.where(val_mask)[0]
    rng = np.random.RandomState(0)
    tpr_sample_idx = rng.choice(val_idx_all, size=min(N_TPR_SAMPLE, len(val_idx_all)), replace=False)

    def predict_batch(attr, indices, batch=256):
        preds = []
        for start in range(0, len(indices), batch):
            chunk = indices[start:start + batch]
            imgs = torch.stack([ds[int(i)]["img"] for i in chunk]).to(device)
            with torch.no_grad():
                p = predictors[attr].predict_raw(imgs).cpu().numpy()
            preds.append(p)
        return np.concatenate(preds)

    results = {}

    # ---- continuous attrs ----
    for attr in CONTINUOUS:
        edges = np.array(cohorts[attr]["bin_edges"])
        n_bins = len(edges) - 1
        true_vals = ds.attrs[tpr_sample_idx, names.index(attr)]
        bin_idx = np.clip(np.digitize(true_vals, edges) - 1, 0, n_bins - 1)
        half_width = (edges[bin_idx + 1] - edges[bin_idx]) / 2.0
        pred_vals = predict_batch(attr, tpr_sample_idx)
        tpr_pass = np.abs(pred_vals - true_vals) <= half_width

        target_bins_used = np.array([row["target_bin"] for row in cohorts[attr]["per_image"]])
        bin_weights = np.zeros(n_bins)
        for b in target_bins_used:
            bin_weights[b] += 1
        bin_weights = bin_weights / bin_weights.sum()

        per_bin_tpr = np.zeros(n_bins)
        per_bin_n = np.zeros(n_bins)
        for b in range(n_bins):
            m = bin_idx == b
            per_bin_n[b] = m.sum()
            per_bin_tpr[b] = tpr_pass[m].mean() if m.sum() > 0 else np.nan
        valid_bins = ~np.isnan(per_bin_tpr)
        w = bin_weights[valid_bins] / bin_weights[valid_bins].sum()
        tpr_reweighted = float(np.sum(w * per_bin_tpr[valid_bins]))
        tpr_unweighted = float(tpr_pass.mean())

        source_true = ds.attrs[fixed_indices, names.index(attr)]
        source_pred = predict_batch(attr, fixed_indices)
        target_vals = np.array([row["target_value"] for row in cohorts[attr]["per_image"]])
        target_bins = np.array([row["target_bin"] for row in cohorts[attr]["per_image"]])
        target_half_width = (edges[target_bins + 1] - edges[target_bins]) / 2.0
        fpr_pass = np.abs(source_pred - target_vals) <= target_half_width
        fpr = float(fpr_pass.mean())

        cc_obs = CC_OBS[attr]
        denom = tpr_reweighted - fpr
        c = (cc_obs - fpr) / denom if abs(denom) > 1e-6 else None

        boot_cs = []
        for _ in range(N_BOOTSTRAP):
            idx_t = rng.randint(0, len(tpr_pass), size=len(tpr_pass))
            idx_f = rng.randint(0, len(fpr_pass), size=len(fpr_pass))
            t_b = tpr_pass[idx_t].mean()
            f_b = fpr_pass[idx_f].mean()
            d_b = t_b - f_b
            if abs(d_b) > 1e-6:
                boot_cs.append((cc_obs - f_b) / d_b)
        boot_cs = np.array(boot_cs)
        c_ci = [float(np.percentile(boot_cs, 2.5)), float(np.percentile(boot_cs, 97.5))] if len(boot_cs) else None

        # multiplier needed to reach TPR=0.95: scan multiples of the tolerance
        mult_scan = {}
        for mult in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
            hw = half_width * mult
            tpr_m = float((np.abs(pred_vals - true_vals) <= hw).mean())
            mult_scan[str(mult)] = tpr_m
        needed_mult = next((m for m, v in mult_scan.items() if v >= 0.95), ">6.0")

        results[attr] = {
            "tpr_unweighted": tpr_unweighted, "tpr_reweighted_by_eval_target_dist": tpr_reweighted,
            "fpr": fpr, "cc_obs": cc_obs, "true_compliance_c": c, "c_ci95": c_ci,
            "identifiable": bool(denom > 0.1),
            "tpr_minus_fpr": float(denom),
            "tolerance_mult_scan_tpr": mult_scan, "mult_needed_for_tpr_0.95": needed_mult,
            "n_tpr_sample": len(tpr_sample_idx), "n_fpr_sample": len(fixed_indices),
        }
        print(f"{attr}: TPR(reweighted)={tpr_reweighted:.4f} FPR={fpr:.4f} CC_obs={cc_obs:.4f} "
             f"-> c={c} CI={c_ci} identifiable={results[attr]['identifiable']}")

    # ---- categorical attrs ----
    for attr in CATEGORICAL:
        n_classes = scm.specs[attr].num_classes
        true_raw = ds.attrs[tpr_sample_idx, names.index(attr)]
        true_class = scm.categorical_class_index(attr, torch.from_numpy(true_raw).float()).cpu().numpy()
        pred_raw = predict_batch(attr, tpr_sample_idx)
        pred_class = scm.categorical_class_index(attr, torch.from_numpy(pred_raw).float()).cpu().numpy()
        tpr = float((pred_class == true_class).mean())

        source_raw = ds.attrs[fixed_indices, names.index(attr)]
        source_class = scm.categorical_class_index(attr, torch.from_numpy(source_raw).float()).cpu().numpy()
        source_pred_raw = predict_batch(attr, fixed_indices)
        source_pred_class = scm.categorical_class_index(attr, torch.from_numpy(source_pred_raw).float()).cpu().numpy()
        target_class = (source_class + n_classes // 2) % n_classes
        fpr = float((source_pred_class == target_class).mean())

        cc_obs = CC_OBS[attr]
        denom = tpr - fpr
        c = (cc_obs - fpr) / denom if abs(denom) > 1e-6 else None

        boot_cs = []
        pc, tc = pred_class == true_class, source_pred_class == target_class
        for _ in range(N_BOOTSTRAP):
            idx_t = rng.randint(0, len(pc), size=len(pc))
            idx_f = rng.randint(0, len(tc), size=len(tc))
            t_b, f_b = pc[idx_t].mean(), tc[idx_f].mean()
            d_b = t_b - f_b
            if abs(d_b) > 1e-6:
                boot_cs.append((cc_obs - f_b) / d_b)
        boot_cs = np.array(boot_cs)
        c_ci = [float(np.percentile(boot_cs, 2.5)), float(np.percentile(boot_cs, 97.5))] if len(boot_cs) else None

        results[attr] = {
            "tpr": tpr, "fpr": fpr, "cc_obs": cc_obs, "true_compliance_c": c, "c_ci95": c_ci,
            "identifiable": bool(denom > 0.1), "tpr_minus_fpr": float(denom),
            "n_tpr_sample": len(tpr_sample_idx), "n_fpr_sample": len(fixed_indices),
        }
        print(f"{attr}: TPR={tpr:.4f} FPR={fpr:.4f} CC_obs={cc_obs:.4f} -> c={c} CI={c_ci}")

    Path(OUT_PATH).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
