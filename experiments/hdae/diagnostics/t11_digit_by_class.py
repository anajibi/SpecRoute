#!/usr/bin/env python
"""T11: digit CC broken down by target class. Requires actual counterfactual generation (no
saved per-image records exist from prior eval runs, per RECON.md) -- this is the lightweight,
digit-intervention-only generation needed to answer it, run on the k11 75k checkpoint (the only
k=11 checkpoint currently available; also serves as T5's recommended 75k re-check for the digit
intervention specifically)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from experiments.hdae.counterfactuals import hdae_adapter  # noqa: F401
from experiments.hdae.counterfactuals.cf_contract import load_adapter
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.attr_predictor import load_attr_predictor
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

CONFIG = "experiments/hdae/configs/morpho_hier_k11_v3.yaml"
CKPT = "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last_step75000.ckpt"
CAUSAL_GRAPH = "experiments/hdae/configs/causal_graph_morpho.yaml"
PACKED = "experiments/hdae/data/packed/morphomnist_70k.h5"
PREDICTORS_DIR = "experiments/hdae/outputs/attr_predictors_70k"
COHORTS = "experiments/hdae/outputs/intervention_cohorts.json"
OUT_PATH = "experiments/hdae/outputs/diagnostics_t11_digit_by_class.json"
EDIT_STRENGTH = 8.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(CAUSAL_GRAPH) as f:
        causal_raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(causal_raw)
    scm = SCM.load(causal_raw["scm_checkpoint"], device=device)
    ds = MorphoMNISTPacked(PACKED)
    summary = json.loads((Path(PREDICTORS_DIR) / "training_summary.json").read_text())
    digit_predictor = load_attr_predictor(summary["digit"]["checkpoint"], attr_col=ds.attribute_names.index("digit")).to(device)

    cohorts = json.loads(Path(COHORTS).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"]

    imgs = torch.stack([ds[int(i)]["img"] for i in fixed_indices]).to(device)
    attrs_raw = torch.stack([torch.as_tensor(ds[int(i)]["attr"]) for i in fixed_indices]).to(device)
    scm_cols = [ds.attribute_names.index(a) for a in graph.attributes]
    scm_attr_index = {n: i for i, n in enumerate(graph.attributes)}
    n_classes = scm.specs["digit"].num_classes
    descendants = sorted(graph.descendants("digit"))
    observed = [a for a in graph.attributes if a != "digit" and a not in descendants]

    print("loading k11 75k adapter...")
    adapter = load_adapter("hdae", CONFIG, CKPT, device, edit_strength=EDIT_STRENGTH, T=100, compile_model=False)
    state = adapter.encode(imgs, attrs_raw, ds.attribute_names)

    cur_class = scm.categorical_class_index("digit", attrs_raw[:, scm_cols[scm_attr_index["digit"]]])
    target_class = (cur_class + n_classes // 2) % n_classes
    target_tensor = scm.class_index_to_raw("digit", target_class).view(-1, 1)
    cf_attrs = scm.counterfactual(attrs_raw[:, scm_cols].float(), scm_attr_index, interventions={"digit": target_tensor})
    for a in observed:
        cf_attrs[a] = attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()

    cf_state = adapter.intervene(state, "digit", "shift", cf_attrs)
    cf = adapter.render(cf_state)
    with torch.no_grad():
        pred_raw = digit_predictor.predict_raw(cf * 2 - 1).cpu().numpy()
    pred_class = np.round(pred_raw).clip(0, 9).astype(int)
    target_np = target_class.detach().cpu().numpy()
    source_np = cur_class.detach().cpu().numpy()
    success = pred_class == target_np

    per_class = {}
    for c in range(10):
        m = target_np == c
        n = int(m.sum())
        per_class[str(c)] = {"n": n, "cc": float(success[m].mean()) if n else None}
        print(f"target_digit={c} n={n} CC={per_class[str(c)]['cc']}")

    overall_cc = float(success.mean())
    print(f"\noverall digit CC (k11 75k, gs={EDIT_STRENGTH}, n={len(fixed_indices)}): {overall_cc:.4f}")

    records = [{"index": int(i), "source_digit": int(source_np[j]), "target_digit": int(target_np[j]),
               "predicted_digit": int(pred_class[j]), "success": bool(success[j])}
              for j, i in enumerate(fixed_indices)]

    out = {"model": "k11_75k", "edit_strength": EDIT_STRENGTH, "overall_cc": overall_cc,
          "per_target_class": per_class, "records": records}
    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
