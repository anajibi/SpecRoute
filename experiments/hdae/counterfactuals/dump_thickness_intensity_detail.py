#!/usr/bin/env python
"""Per-image detail dump for thickness/intensity flip interventions: original (ground-truth raw
value), target (per-image cohort target), measured (CNN predictor's read of the rendered
counterfactual image), and whether it landed within tolerance -- same target/measurement/
tolerance logic as morpho_cf1_eval_binned.py's evaluate_one_intervention_binned, just written out
per-image instead of aggregated into CC.
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
from experiments.hdae.counterfactuals.morpho_cf1_eval import compute_train_stats, load_all_predictors, predict_all
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.morphomnist import MorphoMNISTPacked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph_morpho.yaml")
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist_70k.h5")
    p.add_argument("--predictors-dir", default="experiments/hdae/outputs/attr_predictors_70k")
    p.add_argument("--cohorts", default="experiments/hdae/outputs/intervention_cohorts.json")
    p.add_argument("--n-images", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--edit-strength", type=float, default=8.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.causal_graph) as f:
        causal_raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(causal_raw)
    scm = SCM.load(causal_raw["scm_checkpoint"], device=device)
    adapter = load_adapter("hdae", args.config, args.ckpt, device, edit_strength=args.edit_strength, T=args.T)

    ds = MorphoMNISTPacked(args.packed)
    predictors = load_all_predictors(args.predictors_dir, ds)
    comp = json.loads((Path(args.predictors_dir) / "comparison_results.json").read_text())["per_attribute"]
    continuous_attrs = ["thickness", "intensity"]
    train_stats = compute_train_stats(ds, continuous_attrs)
    tolerances = {}
    for attr in continuous_attrs:
        gap = abs(train_stats[attr]["p85"] - train_stats[attr]["p15"])
        tolerances[attr] = max(comp[attr]["cnn_mae"], 0.05 * gap)

    cohorts = json.loads(Path(args.cohorts).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"][: args.n_images]
    target_by_index = {attr: {row["index"]: row["target_value"] for row in cohorts[attr]["per_image"]}
                       for attr in continuous_attrs}

    scm_cols = [ds.attribute_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}
    attr_col = {a: ds.attribute_names.index(a) for a in continuous_attrs}

    rows = []
    loader = DataLoader(Subset(ds, fixed_indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    for batch in loader:
        x = batch["img"].to(device)
        attrs_raw = batch["attr"].to(device)
        batch_idx = batch["index"].tolist()
        state = adapter.encode(x, attrs_raw, ds.attribute_names)

        for attr in continuous_attrs:
            target_vals = [target_by_index[attr][i] for i in batch_idx]
            target_tensor = torch.tensor(target_vals, device=device).float().view(-1, 1)
            cf_attrs = scm.counterfactual(attrs_raw[:, scm_cols].float(), scm_attr_index,
                                          interventions={attr: target_tensor})
            observed = [a for a in graph.attributes if a != attr and a not in graph.descendants(attr)]
            for a in observed:
                cf_attrs[a] = attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()
            cf_state = adapter.intervene(state, attr, "flip_binned", cf_attrs)
            cf = adapter.render(cf_state)
            pred_cf = predict_all(predictors, cf * 2 - 1, device)

            original = attrs_raw[:, attr_col[attr]].detach().cpu().numpy()
            measured = pred_cf[attr]
            err = np.abs(measured - np.array(target_vals))
            within_tol = err <= tolerances[attr]

            for i, idx in enumerate(batch_idx):
                rows.append({
                    "index": idx, "attribute": attr,
                    "original": round(float(original[i]), 4),
                    "target": round(float(target_vals[i]), 4),
                    "measured": round(float(measured[i]), 4),
                    "tolerance": round(float(tolerances[attr]), 4),
                    "abs_error": round(float(err[i]), 4),
                    "within_tolerance": bool(within_tol[i]),
                })

    rows.sort(key=lambda r: (r["index"], r["attribute"]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_thick = sum(1 for r in rows if r["attribute"] == "thickness")
    n_thick_ok = sum(1 for r in rows if r["attribute"] == "thickness" and r["within_tolerance"])
    n_int = sum(1 for r in rows if r["attribute"] == "intensity")
    n_int_ok = sum(1 for r in rows if r["attribute"] == "intensity" and r["within_tolerance"])
    print(f"wrote {args.output} ({len(rows)} rows, {len(fixed_indices)} images)")
    print(f"thickness: {n_thick_ok}/{n_thick} within tolerance ({tolerances['thickness']:.4f})")
    print(f"intensity: {n_int_ok}/{n_int} within tolerance ({tolerances['intensity']:.4f})")


if __name__ == "__main__":
    main()
