"""Put categorical and continuous attributes on one scale, then pool into CC / FC / CF1.

THE PROBLEM. `class` yields a 0/1 correctness; pos_spl/pos_obj/rot_obj/hues yield real
errors in their own units. Averaging an accuracy with an MAE is meaningless, so a pooled
CC needs a common scale first.

WHY NOT THE TOLERANCE SCHEME. eval_cf1_c3di.py binarises continuous attributes with a
window of `mult * predictor_test_MAE`. That works, but the window is a free parameter the
answer moves with, it discards magnitude (missing by 1.01 windows scores like missing by
10), and -- worst -- tying the window to PREDICTOR error means the attribute we measure
worst gets the widest window and the easiest pass. hue_obj has the weakest predictor here,
so it is the easiest attribute to score a hit on. That is backwards.

THE SCHEME HERE. Normalise every attribute by the quantity the metric is actually asking
about, so the categorical case falls out as an exact special case and no tolerance is
needed anywhere.

  Things that should MOVE (the intervened attribute, and its SCM descendants):

      CC_a = 1 - E|yhat - target| / E|source - target|

  the fraction of the requested move that was realised. 1 = landed on target, 0 = did not
  move at all, < 0 = moved the wrong way or overshot past the target. For `class` the
  distance is 0/1 and source != target holds by cohort construction, so the denominator is
  1 and CC_class reduces to EXACTLY the classification accuracy. Same formula, no branch.

  Things that should HOLD (non-descendant modelled attrs; the three hues):

      FC_a = 1 - max(0, E|yhat - source| - floor_a) / (MAD_a - floor_a)

  1 = held as well as an untouched reconstruction, 0 = as bad as the best TRIVIAL
  predictor. base_a is that trivial predictor's error: for a continuous attribute the
  constant-mean predictor, E|y - ybar|; for `class` the always-answer-the-mode predictor,
  1 - max_c p(c). This is the STRICTER of the two candidate baselines -- a blind sampler
  drawing from the marginal makes error E|y - y'|, which is ~33% larger and would flatter
  every FC. floor_a is this model's own no-intervention round-trip error, so the model is
  not charged for reconstruction cost that guidance did not cause.

  The two baselines coincide exactly for `class` here: the test split is balanced at 3600
  images per class, so mode error = 1 - 1/7 = 6/7 and Gini = 1 - sum p^2 = 6/7 as well.
  Only the six continuous attributes move when the convention changes.

RATIO OF MEANS, NOT MEAN OF RATIOS. Both are per-attribute aggregates of the form
1 - E[num]/E[den], never E[num/den]. A per-sample ratio has a denominator that can be
near zero and produces heavy tails; the ratio of means is stable and is just a normalised
MAE. (The cohort builder already enforces |source - target| >= 0.5, so the denominator is
bounded away from zero regardless.)

POOLING IS PER-ATTRIBUTE, NOT PER-DIMENSION. pos_obj and rot_obj are 3-vectors; their
per-dimension errors are averaged first, so each ATTRIBUTE gets one vote. Otherwise the
two vector attributes would carry 6 of the 8 modelled votes.

CF1 = harmonic mean of pooled CC and pooled FC, computed on values clipped to [0,1]
because a negative CC makes the harmonic mean meaningless. The unclipped per-attribute
numbers are always reported next to it.

    python experiments/hdae/scripts/aggregate_metrics.py --labels k1_n256 k11_n256
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/hdae/causal"))

from experiments.hdae.data.causal3dident import Causal3DIdentPacked  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from train_scm_causal3dident import SCM, CausalGraph  # noqa: E402

MODELLED = ["class", "pos_spl", "pos_obj", "rot_obj"]
UNOBS = ["hue_obj", "hue_spl", "hue_bg"]
ALL7 = MODELLED + UNOBS
COLS = {"class": [0], "pos_spl": [1], "pos_obj": [2, 3, 4], "rot_obj": [5, 6, 7],
        "hue_obj": [8], "hue_spl": [9], "hue_bg": [10]}
EDGES = [("class", "rot_obj"), ("class", "pos_obj"), ("pos_spl", "pos_obj")]
N_CLASS = 7


def descendants(a):
    out, fr = set(), [a]
    while fr:
        n = fr.pop()
        for p, c in EDGES:
            if p == n and c not in out:
                out.add(c); fr.append(c)
    return out


def dist(attr, u, v):
    """Per-sample distance: 0/1 for the categorical attribute, mean |.| for the rest."""
    if attr == "class":
        return (u.reshape(-1) != v.reshape(-1)).astype(np.float64)
    return np.abs(u - v).mean(axis=1)


def trivial_baseline(ds, attr):
    """Error of the best CONSTANT predictor -- the strict reading of 'as bad as trivial'.

    Continuous: always answer the mean, E|y - ybar|.  Categorical: always answer the mode,
    1 - max_c p(c).  Returned alongside the looser blind-SAMPLER baseline E|y - y'| (Gini
    for the categorical case) so the choice of convention stays visible in the output.
    """
    # The WHOLE test split, not a subsample. This baseline is a definitional constant of
    # the dataset -- estimating it from 20k of 25.2k rows injected sampling noise into a
    # number that has an exact value (it made the balanced-class mode error read 0.8556
    # instead of 6/7 = 0.857143).
    #
    # ds.attr is the full attribute matrix, built once in the dataset constructor. Do NOT
    # go through ds[i]: __getitem__ decodes a 128x128 image per call, so pulling rows that
    # way costs one image decode and one random HDF5 read per row, all discarded.
    v = np.asarray(ds.attr[:, COLS[attr]], dtype=np.float64)
    if attr == "class":
        p = np.bincount(v.reshape(-1).astype(int), minlength=N_CLASS) / len(v)
        return float(1.0 - p.max()), float(1.0 - (p ** 2).sum())
    half = len(v) // 2
    return float(np.abs(v - v.mean(axis=0)).mean()), float(np.abs(v[:half] - v[half:2 * half]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments/hdae/outputs/cfg_sweep")
    ap.add_argument("--labels", nargs="+", default=["k1_n256", "k11_n256"])
    ap.add_argument("--configs", nargs="+",
                    default=["experiments/hdae/configs/c3di_hier_k1_final.yaml",
                             "experiments/hdae/configs/c3di_hier_k11_final.yaml"])
    args = ap.parse_args()

    ds = Causal3DIdentPacked(os.path.join(
        REPO, "experiments/hdae/data/causal3dident/causal3dident_testset_128.h5"))
    both = {a: trivial_baseline(ds, a) for a in ALL7}   # exact, over all 25,200 rows
    mad = {a: both[a][0] for a in ALL7}          # constant-predictor baseline: the one in use
    sampler_base = {a: both[a][1] for a in ALL7}  # blind-sampler baseline: recorded, not used
    print("FC baseline in use (best constant predictor):",
          "  ".join(f"{a}={mad[a]:.4f}" for a in ALL7), flush=True)
    print("  (looser blind-sampler baseline, for reference:",
          "  ".join(f"{a}={sampler_base[a]:.4f}" for a in ALL7) + ")", flush=True)

    blob = torch.load(os.path.join(REPO, "experiments/hdae/outputs/scm/causal3dident_scm_spline.pt"),
                      map_location="cpu")
    c = blob["config"]
    scm = SCM(CausalGraph(c["attributes"], c["edges"]), c["nodes"],
              mechanism=blob["mechanism"], bins=blob["bins"])
    scm.load_state_dict(blob["state_dict"]); scm.eval()

    out = {"fc_baseline_constant": mad, "fc_baseline_sampler": sampler_base,
           "baseline_convention": "constant-mean / mode", "models": {}}
    for label, cfgp in zip(args.labels, args.configs):
        js = json.load(open(os.path.join(args.outdir, f"sweep_{label}.json")))
        ps = np.load(os.path.join(args.outdir, f"persample_{label}.npz"))
        idx = [r["i"] for r in js["cohort"]]
        A = np.asarray(ds.attr[idx], dtype=np.float64)              # [n, 11] source
        specs = load_hdae_config(os.path.join(REPO, cfgp), require_data=False).hdae_conf.encoder.cond_specs
        y = to_cond_values(torch.from_numpy(A[:, :8]), specs).numpy()

        # targets: exactly what cfg_sweep_c3di.py rendered (recorded in the cohort block)
        tgt = {}
        from experiments.hdae.data.causal3dident import CLASS_NAMES
        tgt["class"] = np.array([[CLASS_NAMES.index(r["class_tgt"])] for r in js["cohort"]], dtype=np.float64)
        for a in ["pos_spl", "pos_obj", "rot_obj"]:
            tgt[a] = np.array([r[a]["target"] for r in js["cohort"]], dtype=np.float64)

        # SCM counterfactual values for descendants, recomputed the same way
        obs = {k: torch.from_numpy(y[:, COLS[k]]).float().contiguous() for k in MODELLED}
        cf = {}
        for a in MODELLED:
            with torch.no_grad():
                cf[a] = {k: v.numpy().astype(np.float64) for k, v in
                         scm.propagate(scm.abduct(obs), obs,
                                       {a: torch.from_numpy(tgt[a]).float()}).items()}

        floor = {a: js["results"]["ema|recon"][a]["value"] for a in ALL7}
        floor = {a: (1.0 - floor[a]) if a == "class" else floor[a] for a in ALL7}  # class -> error

        raw_recon = {a: js["results"]["ema|recon"][a] for a in ALL7}
        model = {"floor": floor, "raw_recon": raw_recon, "per_g": {}}
        for g in js["strengths"]:
            cells = {}
            for a in MODELLED:
                desc = descendants(a)
                cc, fc, raw = {}, {}, {}
                for k in ALL7:
                    rawv = ps[f"ema|do({a})|g{g:g}|{k}"].astype(np.float64).mean()
                    # cfg_sweep stores CORRECTNESS for the categorical attribute and ERROR
                    # for the continuous ones. Everything below is a distance, so flip it.
                    num = (1.0 - rawv) if k == "class" else rawv
                    # ALWAYS keep the raw instrument reading beside the derived score:
                    # accuracy for `class`, MAE for everything else, unnormalised.
                    raw[k] = {"metric": "accuracy" if k == "class" else "mae",
                              "value": round(float(rawv), 5)}
                    if k == a or k in desc:
                        ref = tgt[k] if k == a else cf[a][k]
                        src = A[:, COLS[k]].astype(np.float64)
                        den = dist(k, src, ref).mean()
                        cc[k] = float(1.0 - num / den) if den > 0 else float("nan")
                    else:
                        den = mad[k] - floor[k]
                        fc[k] = float(1.0 - max(0.0, num - floor[k]) / den) if den > 0 else float("nan")
                obs_fc = {k: v for k, v in fc.items() if k in MODELLED}
                un_fc = {k: v for k, v in fc.items() if k in UNOBS}
                cl = lambda d: [min(1.0, max(0.0, v)) for v in d.values()]
                CC = float(np.mean(cl(cc))); FCo = float(np.mean(cl(obs_fc))); FCu = float(np.mean(cl(un_fc)))
                hm = lambda x, z: 0.0 if (x + z) == 0 else 2 * x * z / (x + z)
                cells[a] = {"raw_per_attr": raw, "cc_per_attr": cc, "fc_per_attr": fc,
                            "CC": CC, "FC_obs": FCo, "FC_unobs": FCu,
                            "CF1_obs": hm(CC, FCo), "CF1_unobs": hm(CC, FCu)}
            allc = [cells[a] for a in MODELLED]
            model["per_g"][f"{g:g}"] = {
                "per_intervention": cells,
                "CC": float(np.mean([x["CC"] for x in allc])),
                "FC_obs": float(np.mean([x["FC_obs"] for x in allc])),
                "FC_unobs": float(np.mean([x["FC_unobs"] for x in allc])),
                "CF1_obs": float(np.mean([x["CF1_obs"] for x in allc])),
                "CF1_unobs": float(np.mean([x["CF1_unobs"] for x in allc]))}
        out["models"][label] = model
        print(f"\n=== {label}")
        print(f"{'g':>5s} {'CC':>7s} {'FC_obs':>7s} {'FC_un':>7s} {'CF1_obs':>8s} {'CF1_un':>8s}")
        for g, v in model["per_g"].items():
            print(f"{g:>5s} {v['CC']:7.4f} {v['FC_obs']:7.4f} {v['FC_unobs']:7.4f} "
                  f"{v['CF1_obs']:8.4f} {v['CF1_unobs']:8.4f}")

    p = os.path.join(args.outdir, "aggregate_metrics.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
