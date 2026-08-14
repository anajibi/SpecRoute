#!/usr/bin/env python
"""Lightweight counterfactual sanity check for MorphoMNIST HDAE variants.

Not a full CF1 (see run_cf1_eval.py) -- that script's CC/FC machinery assumes CelebA's 40 binary
attributes (a >0 threshold, a sigmoid multi-label classifier, a "pos_idx"/"neg_idx" cohort file).
None of that translates to MorphoMNIST's 4 conditioning attributes without a substantial redesign
(digit is categorical, thickness/intensity/hue are continuous with no natural "positive" side) --
deferred as out of proportion to the time available; see the decisions list in PROGRESS-SUMMARY.

What this script does instead, reusing the same production SCM+adapter intervention path a full
CF1 script would use:
1. abduct+intervene+propagate through the fitted SCM (causal/scm.py) for each of the 4 attributes
   in turn -- for thickness, this also moves intensity via the declared causal edge, exactly as
   real CF1 would score.
2. adapter.encode/intervene/render the counterfactual image.
3. Measure the intended attribute's movement AND the other attributes' drift using the
   independently-trained per-attribute CNN predictors (data/train_morpho_attr_predictors.py) as
   the measurement instrument -- these were trained on ground truth, not on this model's own
   conditioning signal, so they're a real (if noisier than a purpose-built CF1 harness) check that
   interventions actually do something and don't leak into attributes they shouldn't.
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

from experiments.hdae.counterfactuals import hdae_adapter  # noqa: F401 -- registers "hdae"
from experiments.hdae.counterfactuals.cf_contract import load_adapter
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.attr_predictor import load_attr_predictor
from experiments.hdae.data.morphomnist import MorphoMNISTPacked
from experiments.hdae.hdae.grid_utils import save_labeled_grid

CONTINUOUS_TARGETS = {"thickness": {"low": 2.513, "high": 3.210},
                      "intensity": {"low": 153.155, "high": 204.982},
                      "hue": {"low": 0.151, "high": 0.850}}


def load_predictors(predictors_dir, attrs, ds):
    summary = json.loads((Path(predictors_dir) / "training_summary.json").read_text())
    out = {}
    for name in attrs:
        info = summary[name]
        out[name] = load_attr_predictor(info["checkpoint"], attr_col=ds.attribute_names.index(name))
    return out


def predict_all(predictors, img_m11, device):
    out = {}
    x = img_m11.to(device)
    for name, model in predictors.items():
        out[name] = model.predict_raw(x).numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph_morpho.yaml")
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist.h5")
    p.add_argument("--predictors-dir", default="experiments/hdae/outputs/attr_predictors")
    p.add_argument("--n-images", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--edit-strength", type=float, default=None)
    p.add_argument("--grid-images", type=int, default=8)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.causal_graph) as f:
        causal_raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(causal_raw)
    scm = SCM.load(causal_raw["scm_checkpoint"], device=device)

    adapter = load_adapter("hdae", args.config, args.ckpt, device, edit_strength=args.edit_strength, T=args.T)
    if set(adapter.modeled_attrs) != set(graph.attributes):
        raise ValueError(f"adapter.modeled_attrs={adapter.modeled_attrs} != graph.attributes={graph.attributes}")

    ds = MorphoMNISTPacked(args.packed)
    predictors = load_predictors(args.predictors_dir, graph.attributes, ds)

    test_idx = np.nonzero(ds.partitions == 1)[0]
    rng = np.random.RandomState(args.seed)
    sel = rng.choice(test_idx, size=min(args.n_images, len(test_idx)), replace=False)
    imgs = torch.stack([ds[int(i)]["img"] for i in sel])
    attrs_raw_all = torch.stack([torch.as_tensor(ds[int(i)]["attr"]) for i in sel])
    scm_cols = [ds.attribute_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}

    grid_rows, grid_labels = [], []
    rows = []
    for start in range(0, len(sel), args.batch_size):
        x = imgs[start:start + args.batch_size].to(device)
        attrs_raw = attrs_raw_all[start:start + args.batch_size].to(device)
        state = adapter.encode(x, attrs_raw, ds.attribute_names)
        recon0 = adapter.render(state)
        pred0 = predict_all(predictors, recon0 * 2 - 1, device)

        for attr in graph.attributes:
            descendants = sorted(graph.descendants(attr))
            observed = [a for a in graph.attributes if a != attr and a not in descendants]
            if attr == "digit":
                cur = attrs_raw[:, scm_cols[scm_attr_index["digit"]]]
                target = (cur.long() + 5) % 10
                directions = {"shift": target.float().view(-1, 1)}
            else:
                lo, hi = CONTINUOUS_TARGETS[attr]["low"], CONTINUOUS_TARGETS[attr]["high"]
                directions = {"low": torch.full((x.shape[0], 1), lo, device=device),
                             "high": torch.full((x.shape[0], 1), hi, device=device)}

            for direction, target_tensor in directions.items():
                cf_attrs = scm.counterfactual(attrs_raw[:, scm_cols].float(), scm_attr_index,
                                              interventions={attr: target_tensor})
                # adapter.intervene() writes every attribute in cf_attrs, not just the intervened
                # one -- correct for the full-CF1 contract, but for non-root non-descendant
                # attributes the SCM's propagated value is a noisy reconstruction of the observed
                # value, not the observed value itself (abduct/propagate round-trip, not identity
                # except for roots). Left as-is, that round-trip noise would land in the
                # "unintended drift" columns below and be indistinguishable from real model
                # leakage. Pin every non-intervened, non-descendant attribute back to its observed
                # raw value so drift only reflects the model, not SCM reconstruction error.
                held_fixed = [a for a in graph.attributes if a != attr and a not in descendants]
                for a in held_fixed:
                    cf_attrs[a] = attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()
                cf_state = adapter.intervene(state, attr, direction, cf_attrs)
                cf = adapter.render(cf_state)
                pred_cf = predict_all(predictors, cf * 2 - 1, device)

                target_np = target_tensor.detach().cpu().numpy().reshape(-1)
                if attr == "digit":
                    pred_cf_class = np.round(pred_cf["digit"]).clip(0, 9)
                    intended_success = float((pred_cf_class == target_np).mean())
                    intended_metric = {"metric": "class_match_rate", "value": intended_success}
                else:
                    err_before = np.abs(pred0[attr] - target_np)
                    err_after = np.abs(pred_cf[attr] - target_np)
                    intended_metric = {"metric": "mae_to_target", "value": float(err_after.mean()),
                                       "mae_before_intervention": float(err_before.mean()),
                                       "moved_closer_frac": float((err_after < err_before).mean())}

                drift = {}
                for other in observed:
                    drift[other] = float(np.abs(pred_cf[other] - pred0[other]).mean())
                descendant_drift = {}
                for d in descendants:
                    cf_target_d = cf_attrs[d].detach().cpu().numpy().reshape(-1)
                    descendant_drift[d] = float(np.abs(pred_cf[d] - cf_target_d).mean())

                rows.append({"attribute": attr, "direction": direction, "n": int(x.shape[0]),
                            **intended_metric,
                            "unintended_drift_mean": float(np.mean(list(drift.values()))) if drift else None,
                            "unintended_drift_per_attr": json.dumps(drift),
                            "descendant_scm_agreement_mae": float(np.mean(list(descendant_drift.values())))
                            if descendant_drift else None,
                            "descendants": ";".join(descendants)})

                if args.grid_images and start == 0:
                    take = min(args.grid_images, x.shape[0])
                    grid_rows.extend([x[:take].add(1).div(2).clamp(0, 1).detach().cpu(),
                                      recon0[:take].clamp(0, 1).detach().cpu(),
                                      cf[:take].clamp(0, 1).detach().cpu()])
                    grid_labels.extend([f"{attr} {direction} original", f"{attr} {direction} recon0",
                                        f"{attr} {direction} cf"])

    if grid_rows:
        save_labeled_grid(grid_rows, grid_labels, out / "morpho_cf_smoketest_grid.png", label_width=260)

    with open(out / "morpho_cf_smoketest.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"\nwrote {out / 'morpho_cf_smoketest.csv'} and grid")


if __name__ == "__main__":
    main()
