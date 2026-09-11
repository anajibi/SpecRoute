"""Did adversarial tap invariance work? Baseline vs adversarial, k=1 and k=11.

Two questions, answered separately because they can disagree:

  1. MECHANISM -- did the attributes actually leave the taps? Compared via the offline ridge
     probe, which is the only honest read: the TRAINING metric cannot distinguish success from
     failure, since uniform predictions give CE = log 10 and MSE = 1 whether the probes never
     learned or the taps were genuinely scrubbed. Both land at head_loss 1.6513.

  2. OUTCOME -- did counterfactual quality improve? The mechanism predicts k=11 should gain
     most, and specifically on the two GLOBAL attributes (hue, intensity) where baseline k=11
     trailed k=1 by -0.1201 and -0.0399 while the spatial pair was a wash (+0.0005).

Guidance is chosen per intervention, per model, on that model's own CC curve -- same rule the
baseline ladder used, so the comparison is like-for-like.
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


def cells(tag):
    p = os.path.join(SW, f"sweep_{tag}.json")
    if not os.path.exists(p):
        return None
    js = json.load(open(p)); r = js["results"]; S = js["strengths"]
    fl = {k: r["recon"][k]["value"] for k in MOD + UNOB}
    out = {"_floor": fl}
    for m in MOD:
        desc = ["intensity"] if m == "thickness" else []
        for g in S:
            cc, fa, fu = [], [], []
            for k in MOD + UNOB:
                e = r[f"do({m})|g{g:g}"][k]["value"]; f0 = fl[k]
                if k == m:
                    src = A[js["cohort_idx"], COL[k]]; tg = np.asarray(js["targets"][m], float)
                    cv = (lambda v: np.round(v) if k == "digit" else np.round(v * 10 - 0.5))
                    mv = float((cv(src) != cv(tg)).mean()) if k in CAT else float(np.abs(src - tg).mean())
                    cc.append(np.clip(1 - max(0, e - f0) / max(mv - f0, 1e-9), 0, 1))
                elif k in desc:
                    cc.append(np.clip(1 - max(0, e - f0) / max(BASE[k] - f0, 1e-9), 0, 1))
                else:
                    val = np.clip(1 - max(0, e - f0) / max(BASE[k] - f0, 1e-9), 0, 1)
                    fa.append(val)
                    if k in UNOB:
                        fu.append(val)
            c, f = float(np.mean(cc)), float(np.mean(fa))
            out[(m, g)] = dict(CC=c, FCall=f, FCu=float(np.mean(fu)),
                               CF1=0.0 if c + f == 0 else 2 * c * f / (c + f))
    out["_S"] = S
    return out


RUNS = [("k=1  baseline", "ladder_k1_s42"), ("k=1  adversarial", "adv_k1_s42"),
        ("k=11 baseline", "ladder_k11_s42"), ("k=11 adversarial", "adv_k11_s42")]
D = {lab: cells(t) for lab, t in RUNS}
D = {k: v for k, v in D.items() if v}

print("=" * 92)
print("2. OUTCOME -- per-intervention best g, and quality at it")
print("=" * 92)
best = {}
for lab, c in D.items():
    S = c["_S"]
    best[lab] = {m: max(S, key=lambda g: c[(m, g)]["CC"]) for m in MOD}
print(f"\n{'model':18s} " + "".join(f"{m:>12s}" for m in MOD) + "     (best g)")
for lab in D:
    print(f"{lab:18s} " + "".join(f"{best[lab][m]:12g}" for m in MOD))

for metric in ["CC", "CF1", "FCu"]:
    nm = {"CC": "CC", "CF1": "CF1", "FCu": "FC_unobs"}[metric]
    print(f"\n{nm} at each model's own best g")
    print(f"{'model':18s} " + "".join(f"{m:>12s}" for m in MOD) + f"{'MEAN':>10s}")
    for lab in D:
        vals = [D[lab][(m, best[lab][m])][metric] for m in MOD]
        print(f"{lab:18s} " + "".join(f"{v:12.4f}" for v in vals) + f"{np.mean(vals):10.4f}")

print("\n" + "=" * 92)
print("DELTA (adversarial - baseline), per attribute")
print("=" * 92)
for K in ["k=1 ", "k=11"]:
    b, a = f"{K} baseline".replace("k=1  ", "k=1  ") , f"{K} adversarial"
    b = [x for x in D if x.startswith(K) and "baseline" in x]
    a = [x for x in D if x.startswith(K) and "adversarial" in x]
    if not b or not a:
        print(f"\n{K}: missing a side, skipped"); continue
    b, a = b[0], a[0]
    print(f"\n{K}")
    print(f"  {'attribute':11s} {'nature':8s} {'dCC':>9s} {'dCF1':>9s} {'dFC_unobs':>11s}")
    for m in MOD:
        db = D[b][(m, best[b][m])]; da = D[a][(m, best[a][m])]
        print(f"  {m:11s} {NATURE[m]:8s} {da['CC']-db['CC']:+9.4f} "
              f"{da['CF1']-db['CF1']:+9.4f} {da['FCu']-db['FCu']:+11.4f}")
    mb = {k: np.mean([D[b][(m, best[b][m])][k] for m in MOD]) for k in ["CC", "CF1", "FCu"]}
    ma = {k: np.mean([D[a][(m, best[a][m])][k] for m in MOD]) for k in ["CC", "CF1", "FCu"]}
    print(f"  {'MEAN':11s} {'':8s} {ma['CC']-mb['CC']:+9.4f} {ma['CF1']-mb['CF1']:+9.4f} "
          f"{ma['FCu']-mb['FCu']:+11.4f}")
    gl = [m for m in MOD if NATURE[m] == "GLOBAL"]
    print(f"  GLOBAL attrs only: dCC {np.mean([D[a][(m,best[a][m])]['CC']-D[b][(m,best[b][m])]['CC'] for m in gl]):+.4f}"
          f"   (the mechanism predicts the gain concentrates here)")

print("\nReconstruction floor -- the cost of stripping attributes the decoder needs")
print(f"{'model':18s} " + "".join(f"{k[:9]:>11s}" for k in MOD))
for lab in D:
    fl = D[lab]["_floor"]
    print(f"{lab:18s} " + "".join(f"{fl[k]:11.4f}" for k in MOD))
