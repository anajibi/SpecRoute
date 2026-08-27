"""Pick the best guidance strength per attribute, with a paired bootstrap over the cohort.

The argmin over eleven noisy cell means overfits: with 256 samples the gap between
neighbouring strengths is often smaller than the sampling error. Because every strength is
evaluated on the SAME 256 cohort members, the comparison is paired -- resampling cohort
indices (not cells) and recomputing every strength on the resampled indices gives a CI on
the DIFFERENCE from the leader, which is far tighter than two independent CIs would be.

Reported per attribute: the leader, its 95% CI, and the "tied set" -- every strength whose
paired difference from the leader has a 95% CI containing zero. The honest answer to
"what is the best g" is the tied set; the leader alone is the point estimate.
"""
import argparse
import json
import os

import numpy as np

MOD = ["class", "pos_spl", "pos_obj", "rot_obj"]
ALL7 = MOD + ["hue_obj", "hue_spl", "hue_bg"]
B = 10000


def analyse(outdir, label, seed=0):
    js = json.load(open(os.path.join(outdir, f"sweep_{label}.json")))
    ps = np.load(os.path.join(outdir, f"persample_{label}.npz"))
    S = js["strengths"]
    n = len(js["cohort"])
    rng = np.random.RandomState(seed)
    boot = rng.randint(0, n, size=(B, n))          # shared across every cell -> paired
    out = {"label": label, "n": n, "strengths": S, "attrs": {}}
    out["recon"] = {a: js["results"]["ema|recon"][a]["value"] for a in ALL7}

    for a in MOD:
        hi_better = a == "class"
        cols = np.stack([ps[f"ema|do({a})|g{g:g}|{a}"] for g in S])       # [G, n]
        means = cols.mean(1)
        lead = int(means.argmax() if hi_better else means.argmin())
        bm = cols[:, boot].mean(2)                                        # [G, B]
        ci = np.percentile(bm, [2.5, 97.5], axis=1)
        d = bm - bm[lead]                                                 # paired difference
        dci = np.percentile(d, [2.5, 97.5], axis=1)
        tied = [S[i] for i in range(len(S)) if dci[0, i] <= 0 <= dci[1, i]]
        # collateral: the three unmodelled hues, which the SCM never touches
        coll = np.stack([np.stack([ps[f"ema|do({a})|g{g:g}|{h}"]
                                   for h in ALL7[4:]]).mean(0) for g in S]).mean(1)
        out["attrs"][a] = {
            "metric": "accuracy" if hi_better else "mae",
            "best_g": S[lead], "best": float(means[lead]),
            "best_ci": [float(ci[0, lead]), float(ci[1, lead])],
            "tied_g": tied,
            "per_g": {f"{g:g}": {"mean": float(means[i]),
                                 "ci": [float(ci[0, i]), float(ci[1, i])],
                                 "delta_ci": [float(dci[0, i]), float(dci[1, i])],
                                 "collateral": float(coll[i])} for i, g in enumerate(S)},
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments/hdae/outputs/cfg_sweep")
    ap.add_argument("--labels", nargs="+", default=["k1_n256", "k11_n256"])
    a = ap.parse_args()
    res = {l: analyse(a.outdir, l) for l in a.labels}
    p = os.path.join(a.outdir, "best_g.json")
    json.dump(res, open(p, "w"), indent=2)
    for l, r in res.items():
        print(f"\n=== {l}  (n={r['n']})")
        for k, v in r["attrs"].items():
            f = (lambda x: f"{x*100:.1f}%") if v["metric"] == "accuracy" else (lambda x: f"{x:.4f}")
            print(f"  do({k:8s}) best g={v['best_g']:<5g} {f(v['best'])}  "
                  f"CI[{f(v['best_ci'][0])},{f(v['best_ci'][1])}]  tied={v['tied_g']}")
    print(f"\nwrote {p}")
