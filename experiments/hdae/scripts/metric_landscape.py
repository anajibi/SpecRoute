"""Compute EVERY variant of CC / FC / CF1 on the saved per-sample errors.

Nothing here re-renders. cfg_sweep_c3di.py saves the per-sample raw error for every
(model, intervention, strength, attribute) cell, so the entire design space -- every
choice of denominator, floor convention, and composition rule -- is post-hoc arithmetic
over those files. That is the point of saving them.

The space has four independent axes:

  A  CC denominator     what does "failed to move" mean
  B  FC denominator     what does "destroyed" mean
  C  floor convention   is instrument noise subtracted, and from which metric
  D  composition        how CC and FC become one number

This enumerates all of them and reports whether each choice changes the CONCLUSION
(k=11 > k=1) or only the numbers.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/hdae/causal"))

from experiments.hdae.data.causal3dident import CLASS_NAMES, Causal3DIdentPacked  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from train_scm_causal3dident import SCM, CausalGraph  # noqa: E402

MOD = ["class", "pos_spl", "pos_obj", "rot_obj"]
UNOBS = ["hue_obj", "hue_spl", "hue_bg"]
ALL7 = MOD + UNOBS
COLS = {"class": [0], "pos_spl": [1], "pos_obj": [2, 3, 4], "rot_obj": [5, 6, 7],
        "hue_obj": [8], "hue_spl": [9], "hue_bg": [10]}
EDGES = [("class", "rot_obj"), ("class", "pos_obj"), ("pos_spl", "pos_obj")]
MODELS = {"k1": ("k1_n256", "experiments/hdae/configs/c3di_hier_k1_final.yaml"),
          "k11": ("k11_n256", "experiments/hdae/configs/c3di_hier_k11_final.yaml")}
PRED = os.path.join(REPO, "experiments/hdae/outputs/attr_predictors_c3di")


def descendants(a):
    out, fr = set(), [a]
    while fr:
        n = fr.pop()
        for p, c in EDGES:
            if p == n and c not in out:
                out.add(c); fr.append(c)
    return out


def dist(attr, u, v):
    if attr == "class":
        return (u.reshape(-1) != v.reshape(-1)).astype(np.float64)
    return np.abs(u - v).mean(axis=1)


def main():
    ds = Causal3DIdentPacked(os.path.join(
        REPO, "experiments/hdae/data/causal3dident/causal3dident_testset_128.h5"))
    A_all = np.asarray(ds.attr, dtype=np.float64)

    # ---- the reference quantities, all exact over the full 25,200-row split ----
    const_base, samp_base, spread_max = {}, {}, {}
    for a in ALL7:
        v = A_all[:, COLS[a]]
        if a == "class":
            p = np.bincount(v.reshape(-1).astype(int), minlength=7) / len(v)
            const_base[a] = float(1.0 - p.max())        # always answer the mode
            samp_base[a] = float(1.0 - (p ** 2).sum())  # blind draw from the marginal
            spread_max[a] = 1.0                         # max 0/1 distance
        else:
            h = len(v) // 2
            const_base[a] = float(np.abs(v - v.mean(0)).mean())
            samp_base[a] = float(np.abs(v[:h] - v[h:2 * h]).mean())
            spread_max[a] = float(v.max() - v.min())    # full attribute range
    pred_mae = {}
    for a in ALL7:
        m = json.load(open(os.path.join(PRED, f"{a}_metrics.json")))
        # the MAE lives at m["test"]["mae"]; a flat m["mae"] does not exist
        pred_mae[a] = float((m.get("test") or {}).get("mae") or 0.0)

    blob = torch.load(os.path.join(REPO, "experiments/hdae/outputs/scm/causal3dident_scm_spline.pt"),
                      map_location="cpu")
    c = blob["config"]
    scm = SCM(CausalGraph(c["attributes"], c["edges"]), c["nodes"],
              mechanism=blob["mechanism"], bins=blob["bins"])
    scm.load_state_dict(blob["state_dict"]); scm.eval()
    bestg = json.load(open(os.path.join(REPO, "experiments/hdae/outputs/cfg_sweep/best_g.json")))

    out = {"references": {"constant_predictor": const_base, "blind_sampler": samp_base,
                          "attribute_range": spread_max, "predictor_test_mae": pred_mae},
           "models": {}}

    for name, (label, cfgp) in MODELS.items():
        js = json.load(open(os.path.join(REPO, f"experiments/hdae/outputs/cfg_sweep/sweep_{label}.json")))
        ps = np.load(os.path.join(REPO, f"experiments/hdae/outputs/cfg_sweep/persample_{label}.npz"))
        idx = [r["i"] for r in js["cohort"]]
        A = np.asarray(ds.attr[idx], dtype=np.float64)
        specs = load_hdae_config(os.path.join(REPO, cfgp), require_data=False).hdae_conf.encoder.cond_specs
        y = to_cond_values(torch.from_numpy(A[:, :8]), specs).numpy()

        tgt = {"class": np.array([[CLASS_NAMES.index(r["class_tgt"])] for r in js["cohort"]], dtype=np.float64)}
        for a in ["pos_spl", "pos_obj", "rot_obj"]:
            tgt[a] = np.array([r[a]["target"] for r in js["cohort"]], dtype=np.float64)
        obs = {k: torch.from_numpy(y[:, COLS[k]]).float().contiguous() for k in MOD}
        cf = {}
        for a in MOD:
            with torch.no_grad():
                cf[a] = {k: v.numpy().astype(np.float64) for k, v in
                         scm.propagate(scm.abduct(obs), obs, {a: torch.from_numpy(tgt[a]).float()}).items()}

        floor, floor_ps = {}, {}
        for a in ALL7:
            e = ps[f"ema|recon|{a}"].astype(np.float64)
            if a == "class":
                e = 1.0 - e
            floor_ps[a] = e; floor[a] = float(e.mean())

        rec = {"floor": floor, "best_g": {a: bestg[label]["attrs"][a]["best_g"] for a in MOD},
               "moves": {}, "cells": {}}
        for a in MOD:
            rec["moves"][a] = float(dist(a, A[:, COLS[a]], tgt[a]).mean())

        for a in MOD:
            g = rec["best_g"][a]
            desc = descendants(a)
            cell = {"g": g, "raw": {}, "role": {}, "cc": {}, "fc": {}}
            for k in ALL7:
                e = ps[f"ema|do({a})|g{g:g}|{k}"].astype(np.float64)
                raw = float(e.mean())
                err = (1.0 - raw) if k == "class" else raw
                cell["raw"][k] = {"metric": "accuracy" if k == "class" else "mae",
                                  "value": round(raw, 5)}
                per = (1.0 - e) if k == "class" else e          # per-sample DISTANCE
                fl = floor[k]
                if k == a or k in desc:
                    cell["role"][k] = "target" if k == a else "descendant"
                    ref = tgt[k] if k == a else cf[a][k]
                    move = float(dist(k, A[:, COLS[k]], ref).mean())
                    cell["cc"][k] = {
                        "A1_move":        1.0 - err / move,
                        "A1f_move_floor": 1.0 - max(0.0, err - fl) / (move - fl),
                        "A2_range":       1.0 - err / spread_max[k],
                        "A3_spread":      1.0 - err / const_base[k],
                        **{f"A4_hit_x{m}": float((per <= (m * pred_mae[k] if k != "class" else 0.5)).mean())
                           for m in (1, 2, 3)},
                    }
                else:
                    cell["role"][k] = "hold"
                    cell["fc"][k] = {
                        "B1_const":       1.0 - max(0.0, err - fl) / (const_base[k] - fl),
                        "B1n_const_nofl": 1.0 - err / const_base[k],
                        "B2_sampler":     1.0 - max(0.0, err - fl) / (samp_base[k] - fl),
                        "B3_pred_ratio":  err / pred_mae[k] if pred_mae[k] else float("nan"),
                        **{f"B4_hit_x{m}": float((per <= (m * pred_mae[k] if k != "class" else 0.5)).mean())
                           for m in (1, 2, 3)},
                    }
            rec["cells"][a] = cell
        out["models"][name] = rec

    p = os.path.join(REPO, "experiments/hdae/outputs/cfg_sweep/metric_landscape.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"wrote {p}")

    # ---- does the choice change the CONCLUSION? ----
    print("\nEach CC variant, both models, at each model's own best g:\n")
    variants = ["A1_move", "A1f_move_floor", "A2_range", "A3_spread", "A4_hit_x1", "A4_hit_x2", "A4_hit_x3"]
    print(f"{'variant':16s} " + " ".join(f"{a[:9]:>19s}" for a in MOD))
    print(f"{'':16s} " + " ".join(f"{'k=1':>9s}{'k=11':>10s}" for _ in MOD))
    for v in variants:
        row = []
        for a in MOD:
            x = out["models"]["k1"]["cells"][a]["cc"][a][v]
            z = out["models"]["k11"]["cells"][a]["cc"][a][v]
            row.append(f"{x:9.4f}{z:10.4f}")
        print(f"{v:16s} " + " ".join(row))
    print("\nEach FC variant, averaged over the three unmodelled hues:\n")
    fv = ["B1_const", "B1n_const_nofl", "B2_sampler", "B3_pred_ratio", "B4_hit_x1", "B4_hit_x2", "B4_hit_x3"]
    print(f"{'variant':16s} " + " ".join(f"{a[:9]:>19s}" for a in MOD))
    for v in fv:
        row = []
        for a in MOD:
            x = np.mean([out["models"]["k1"]["cells"][a]["fc"][h][v] for h in UNOBS])
            z = np.mean([out["models"]["k11"]["cells"][a]["fc"][h][v] for h in UNOBS])
            row.append(f"{x:9.4f}{z:10.4f}")
        print(f"{v:16s} " + " ".join(row))


if __name__ == "__main__":
    main()
