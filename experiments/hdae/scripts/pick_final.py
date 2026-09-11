"""Pick between ed256 and ed384 on their SEED MEAN, and print the launch decision.

The 13-config sweep ranked ed256 above ed384 on a single seed each. Replicates showed ed256's
seed-to-seed CC spread (0.0539 over seeds 42/43) is larger than the 0.0342 gap that ranking
rested on, and the ordering flipped on seed 43. So the decision is made on the mean over seeds,
with the spread printed next to it -- if the means are closer than the spreads, the choice is
arbitrary on this evidence and the tie-break is stability, not the point estimate.
"""
import json, os, sys
import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SW = os.path.join(REPO, "experiments/hdae/outputs/morpho_sweep")
MOD = ["digit", "thickness", "intensity", "hue"]
UNOB = ["rotation", "scale", "translate_x", "translate_y",
        "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude"]
CAT = {"digit", "hue"}

with h5py.File(os.path.join(REPO, "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5"), "r") as h:
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["attribute_names"][:]]
    A = h["attrs"][:].astype(np.float64)
COL = {k: names.index(k) for k in names}
BASE = {}
for k in MOD + UNOB:
    v = A[:, COL[k]]
    if k in CAT:
        idx = np.round(v) if k == "digit" else np.round(v * 10 - 0.5)
        BASE[k] = float(1 - np.bincount(idx.astype(int), minlength=10).max() / len(v))
    else:
        BASE[k] = float(np.abs(v - v.mean()).mean())


def score(tag):
    p = os.path.join(SW, f"sweep_tune_{tag}.json")
    if not os.path.exists(p):
        return None
    js = json.load(open(p)); r = js["results"]; S = js["strengths"]
    fl = {k: r["recon"][k]["value"] for k in MOD + UNOB}
    best = None
    for g in S:
        ccs, cfs = [], []
        for m in MOD:
            desc = ["intensity"] if m == "thickness" else []
            cc, fc = [], []
            for k in MOD + UNOB:
                err = r[f"do({m})|g{g:g}"][k]["value"]; f0 = fl[k]
                if k == m:
                    src = A[js["cohort_idx"], COL[k]]; tg = np.asarray(js["targets"][m], float)
                    if k in CAT:
                        si = np.round(src) if k == "digit" else np.round(src * 10 - 0.5)
                        ti = np.round(tg) if k == "digit" else np.round(tg * 10 - 0.5)
                        mv = float((si != ti).mean())
                    else:
                        mv = float(np.abs(src - tg).mean())
                    cc.append(np.clip(1 - max(0, err - f0) / max(mv - f0, 1e-9), 0, 1))
                elif k in desc:
                    cc.append(np.clip(1 - max(0, err - f0) / max(BASE[k] - f0, 1e-9), 0, 1))
                else:
                    fc.append(np.clip(1 - max(0, err - f0) / max(BASE[k] - f0, 1e-9), 0, 1))
            c = np.mean(cc); f = np.mean(fc)
            ccs.append(c); cfs.append(0.0 if c + f == 0 else 2 * c * f / (c + f))
        cc_g, cf_g = float(np.mean(ccs)), float(np.mean(cfs))
        if best is None or cc_g > best[1]:
            best = (g, cc_g, cf_g)
    return best


rows = {}
for cfg in ["ed256", "ed384"]:
    seeds = {}
    for tag, sd in [(cfg, 42), (f"{cfg}_s43", 43), (f"{cfg}_s44", 44)]:
        s = score(tag)
        if s:
            seeds[sd] = s
    rows[cfg] = seeds

print(f"{'config':8s} {'seed':>5s} {'g':>3s} {'CC':>8s} {'CF1':>8s}")
for cfg, seeds in rows.items():
    for sd, (g, cc, cf) in sorted(seeds.items()):
        print(f"{cfg:8s} {sd:5d} {g:3g} {cc:8.4f} {cf:8.4f}")
print()
summ = {}
for cfg, seeds in rows.items():
    cc = np.array([v[1] for v in seeds.values()]); cf = np.array([v[2] for v in seeds.values()])
    summ[cfg] = (cc.mean(), cc.std(ddof=1) if len(cc) > 1 else float("nan"),
                 cf.mean(), cf.std(ddof=1) if len(cf) > 1 else float("nan"), len(cc))
    print(f"{cfg:8s} n={len(cc)}  CC {cc.mean():.4f} +/- {summ[cfg][1]:.4f}   "
          f"CF1 {cf.mean():.4f} +/- {summ[cfg][3]:.4f}")

gap = summ["ed256"][0] - summ["ed384"][0]
spread = max(summ["ed256"][1], summ["ed384"][1])
win = "ed256" if gap > 0 else "ed384"
print(f"\nmean CC gap (ed256 - ed384): {gap:+.4f}   largest within-config sd: {spread:.4f}")
if abs(gap) < spread:
    alt = "ed384" if summ["ed384"][1] < summ["ed256"][1] else "ed256"
    print(f"gap is SMALLER than the seed spread -- the point estimates do not separate these.")
    print(f"tie-break on stability (smaller sd): {alt}")
    win = alt
else:
    print(f"gap exceeds the seed spread -- {win} wins on the mean.")
print(f"\nWINNER {win}")
