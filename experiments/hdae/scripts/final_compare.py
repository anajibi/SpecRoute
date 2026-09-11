"""Final six-way comparison on the 4,000-image cohort.

Guidance was chosen per intervention on the 512-image cohort (seed 0); these numbers are
measured on a disjoint 4,000-image draw (seed 1), so no model keeps a strength that only its
own selection cohort favoured.
"""
import json, os
import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SW = os.path.join(REPO, "experiments/hdae/outputs/morpho_sweep")
MOD = ["digit", "thickness", "intensity", "hue"]
UNOB = ["rotation", "scale", "translate_x", "translate_y",
        "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude"]
CAT = {"digit", "hue"}
NATURE = {"digit": "spatial", "thickness": "spatial", "intensity": "GLOBAL", "hue": "GLOBAL"}

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
    p = os.path.join(SW, f"sweep_n4000_{tag}.json")
    if not os.path.exists(p):
        return None
    js = json.load(open(p)); r = js["results"]; S = js["strengths"]
    fl = {k: r["recon"][k]["value"] for k in MOD + UNOB}
    per, gs = {}, {}
    for m in MOD:
        key = [k for k in r if k.startswith(f"do({m})|g")]
        if not key:
            return None
        k0 = key[0]; gs[m] = float(k0.split("|g")[1])
        desc = ["intensity"] if m == "thickness" else []
        cc, fa, fu, fo = [], [], [], []
        for k in MOD + UNOB:
            e = r[k0][k]["value"]; f0 = fl[k]
            if k == m:
                src = A[js["cohort_idx"], COL[k]]; tg = np.asarray(js["targets"][m], float)
                cv = (lambda v: np.round(v)) if k == "digit" else (lambda v: np.round(v * 10 - 0.5))
                mv = float((cv(src) != cv(tg)).mean()) if k in CAT else float(np.abs(src - tg).mean())
                cc.append(np.clip(1 - max(0, e - f0) / max(mv - f0, 1e-9), 0, 1))
            elif k in desc:
                cc.append(np.clip(1 - max(0, e - f0) / max(BASE[k] - f0, 1e-9), 0, 1))
            else:
                val = np.clip(1 - max(0, e - f0) / max(BASE[k] - f0, 1e-9), 0, 1)
                fa.append(val); (fu if k in UNOB else fo).append(val)
        c, f = float(np.mean(cc)), float(np.mean(fa))
        per[m] = dict(CC=c, FCall=f, FCu=float(np.mean(fu)), FCo=float(np.mean(fo)),
                      CF1=0.0 if c + f == 0 else 2 * c * f / (c + f))
    return dict(per=per, g=gs, floor=fl, n=js["n"])


ARMS = [("k=1", "base_k1", "adv_k1"), ("k=5", "base_k5", "adv_k5"), ("k=11", "base_k11", "adv_k11")]
D = {}
for K, b, a in ARMS:
    for lab, t in [(f"{K} baseline", b), (f"{K} adversarial", a)]:
        s = score(t)
        if s: D[lab] = s
order = [f"{K} {w}" for K, _, _ in ARMS for w in ("baseline", "adversarial")]
order = [o for o in order if o in D]
n = D[order[0]]["n"]

print(f"FINAL COMPARISON -- {n} images, per-intervention best g (selected on a disjoint 512 cohort)\n")
print(f"{'model':18s} " + "".join(f"{m:>11s}" for m in MOD) + "    (g used)")
for o in order:
    print(f"{o:18s} " + "".join(f"{D[o]['g'][m]:11g}" for m in MOD))

for met, nm in [("CC", "CC  (did the edit land)"), ("CF1", "CF1 (balance)"),
                ("FCu", "FC_unobs (preservation of the 8 unmodelled)")]:
    print(f"\n{nm}")
    print(f"{'model':18s} " + "".join(f"{m:>11s}" for m in MOD) + f"{'MEAN':>10s}")
    for o in order:
        v = [D[o]["per"][m][met] for m in MOD]
        print(f"{o:18s} " + "".join(f"{x:11.4f}" for x in v) + f"{np.mean(v):10.4f}")

print("\n" + "=" * 86)
print("ADVERSARIAL - BASELINE")
print("=" * 86)
print(f"{'depth':7s} {'dCC':>9s} {'dCF1':>9s} {'dFC_unobs':>11s}   {'dCC global':>11s} {'dCC spatial':>12s}")
for K, b, a in ARMS:
    lb, la = f"{K} baseline", f"{K} adversarial"
    if lb not in D or la not in D:
        continue
    d = lambda met, ms: np.mean([D[la]["per"][m][met] - D[lb]["per"][m][met] for m in ms])
    print(f"{K:7s} {d('CC',MOD):+9.4f} {d('CF1',MOD):+9.4f} {d('FCu',MOD):+11.4f}   "
          f"{d('CC',[m for m in MOD if NATURE[m]=='GLOBAL']):+11.4f} "
          f"{d('CC',[m for m in MOD if NATURE[m]=='spatial']):+12.4f}")

print("\nPer-attribute dCC")
print(f"{'depth':7s} " + "".join(f"{m:>12s}" for m in MOD))
for K, b, a in ARMS:
    lb, la = f"{K} baseline", f"{K} adversarial"
    if lb not in D or la not in D: continue
    print(f"{K:7s} " + "".join(f"{D[la]['per'][m]['CC']-D[lb]['per'][m]['CC']:+12.4f}" for m in MOD))

print("\nReconstruction floor (lower is better)")
print(f"{'model':18s} " + "".join(f"{k[:9]:>11s}" for k in MOD))
for o in order:
    print(f"{o:18s} " + "".join(f"{D[o]['floor'][k]:11.4f}" for k in MOD))
json.dump({o: {"g": D[o]["g"], "per": D[o]["per"], "floor": D[o]["floor"], "n": D[o]["n"]}
           for o in order}, open(os.path.join(REPO, "experiments/hdae/outputs/final_n4000.json"), "w"), indent=2)
print(f"\nwrote experiments/hdae/outputs/final_n4000.json")
