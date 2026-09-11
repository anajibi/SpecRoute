"""CC / FC / CF1 for MorphoMNIST, from the per-sample errors cfg_sweep_morpho.py saves.

Same definitions as the Causal3DIdent pipeline, with the dataset's own specifics:

  CC_k = 1 - max(0, E d(yhat_k, t_k) - floor_k) / (E d(s_k, t_k) - floor_k)
  FC_k = 1 - max(0, E d(yhat_k, s_k) - floor_k) / (base_k - floor_k)
  CF1  = 2*CC*FC/(CC+FC)   on values clipped to [0,1]

d is mean absolute difference, or 0/1 distance for the two categorical attributes (digit, hue),
per the decision to score hue by exact bin match rather than circular distance. base_k is the
best CONSTANT predictor's error: E|y - ybar| for continuous, 1 - max_c p(c) for categorical,
computed exactly over the whole test split. floor_k is that model's own no-intervention error.

The graph has one edge, thickness -> intensity, so only do(thickness) has a descendant. Its
fitted coupling is ~10% too strong (Pearson r +0.797 from the SCM vs +0.723 in the data), which
biases descendant-CC for that intervention downward -- the model is charged for correctly
producing a smaller intensity change than the SCM demands.
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOD = ["digit", "thickness", "intensity", "hue"]
UNOBS = ["rotation", "scale", "translate_x", "translate_y",
         "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude"]
ALL12 = MOD + UNOBS
CATEGORICAL = {"digit", "hue"}
EDGES = [("thickness", "intensity")]
clip = lambda v: min(1.0, max(0.0, v))
hm = lambda a, b: 0.0 if (a + b) == 0 else 2 * a * b / (a + b)


def descendants(a):
    out, fr = set(), [a]
    while fr:
        n = fr.pop()
        for p, c in EDGES:
            if p == n and c not in out:
                out.add(c); fr.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default=os.path.join(REPO, "experiments/hdae/outputs/morpho_sweep"))
    ap.add_argument("--h5", default=os.path.join(REPO, "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5"))
    ap.add_argument("--pattern", default="sweep_tune_*.json")
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/hdae/outputs/tune_metrics.json"))
    a = ap.parse_args()

    with h5py.File(a.h5, "r") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["attribute_names"][:]]
        A_all = h["attrs"][:].astype(np.float64)
    COL = {k: names.index(k) for k in ALL12}
    base = {}
    for k in ALL12:
        v = A_all[:, COL[k]]
        if k in CATEGORICAL:
            idx = np.round(v) if k == "digit" else np.round(v * 10 - 0.5)
            p = np.bincount(idx.astype(int), minlength=10) / len(idx)
            base[k] = float(1.0 - p.max())
        else:
            base[k] = float(np.abs(v - v.mean()).mean())

    out = {"baselines": base, "models": {}}
    for jp in sorted(glob.glob(os.path.join(a.sweeps, a.pattern))):
        tag = os.path.basename(jp)[len("sweep_tune_"):-len(".json")]
        js = json.load(open(jp))
        r = js["results"]; S = js["strengths"]
        floor = {k: r["recon"][k]["value"] for k in ALL12}
        per_g = {}
        for g in S:
            cells = {}
            for m in MOD:
                d = descendants(m)
                cc, fc = {}, {}
                for k in ALL12:
                    err = r[f"do({m})|g{g:g}"][k]["value"]
                    fl = floor[k]
                    if k == m or k in d:
                        # denominator: how far the intervention asked this attribute to move
                        src = A_all[js["cohort_idx"], COL[k]]
                        if k == m:
                            tgt = np.asarray(js["targets"][m], dtype=np.float64)
                        else:
                            tgt = None      # descendant target not stored; use its own error scale
                        if tgt is not None:
                            if k in CATEGORICAL:
                                si = np.round(src) if k == "digit" else np.round(src * 10 - 0.5)
                                ti = np.round(tgt) if k == "digit" else np.round(tgt * 10 - 0.5)
                                move = float((si != ti).mean())
                            else:
                                move = float(np.abs(src - tgt).mean())
                        else:
                            move = base[k]      # descendants: scale by the trivial-predictor error
                        cc[k] = 1 - max(0.0, err - fl) / max(move - fl, 1e-9)
                    else:
                        fc[k] = 1 - max(0.0, err - fl) / max(base[k] - fl, 1e-9)
                CC = float(np.mean([clip(v) for v in cc.values()]))
                FO = float(np.mean([clip(v) for k, v in fc.items() if k in MOD])) if any(k in MOD for k in fc) else float("nan")
                FU = float(np.mean([clip(v) for k, v in fc.items() if k in UNOBS]))
                FA = float(np.mean([clip(v) for v in fc.values()]))
                cells[m] = dict(CC=CC, FC_obs=FO, FC_unobs=FU, FC_all=FA,
                                CF1=hm(CC, FA), target_raw=r[f"do({m})|g{g:g}"][m]["value"],
                                cc_per_attr=cc)
            per_g[f"{g:g}"] = dict(per_intervention=cells,
                                   CC=float(np.mean([cells[m]["CC"] for m in MOD])),
                                   FC_all=float(np.mean([cells[m]["FC_all"] for m in MOD])),
                                   FC_unobs=float(np.mean([cells[m]["FC_unobs"] for m in MOD])),
                                   CF1=float(np.mean([cells[m]["CF1"] for m in MOD])))
        out["models"][tag] = {"floor": floor, "per_g": per_g, "strengths": S}

    json.dump(out, open(a.out, "w"), indent=2)
    tags = list(out["models"])
    if not tags:
        print("no sweeps found"); return
    S = out["models"][tags[0]]["strengths"]
    print(f"CC pooled over the four interventions (selection metric)\n")
    print(f"{'config':10s} " + "".join(f"{'g='+format(g,'g'):>9s}" for g in S) + f"{'best':>8s} {'at g':>6s}")
    for t in tags:
        v = [out["models"][t]["per_g"][f"{g:g}"]["CC"] for g in S]
        i = int(np.argmax(v))
        print(f"{t:10s} " + "".join(f"{x:9.4f}" for x in v) + f"{v[i]:8.4f} {S[i]:6g}")
    print(f"\nFC_all and CF1 at each config's best-CC strength (recorded, not selected on)\n")
    print(f"{'config':10s} {'g':>5s} {'CC':>8s} {'FC_all':>8s} {'FC_unobs':>9s} {'CF1':>8s}")
    for t in tags:
        v = [out["models"][t]["per_g"][f"{g:g}"]["CC"] for g in S]
        i = int(np.argmax(v)); c = out["models"][t]["per_g"][f"{S[i]:g}"]
        print(f"{t:10s} {S[i]:5g} {c['CC']:8.4f} {c['FC_all']:8.4f} {c['FC_unobs']:9.4f} {c['CF1']:8.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
