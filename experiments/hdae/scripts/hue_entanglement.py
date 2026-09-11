"""Does hue->intensity entanglement weaken as attr_embed_dim grows?

Under do(hue) the model also moves thickness and intensity. A predictor check on real images
(hue_confound_check.py) accounted for only ~7% of that, so the rest is the generator: changing
colour changes brightness. The question is whether that is a capacity limit in the attribute
embedding -- in which case a wider embedding should decouple them -- or baked into the data.

Comparing configs at matched g would be unfair: a config that lands the hue edit harder will
show more collateral damage for that reason alone. So this measures the DOSE-RESPONSE. For each
config, sweep g, and regress collateral damage on how much hue actually moved:

    x = hue edit landed  = 1 - (hue error rate)       in [0, 1]
    y = excess intensity error = (MAE under do(hue) - recon floor) / best-constant baseline

The slope dy/dx is the entanglement coefficient: how much intensity damage the model incurs per
unit of hue actually delivered. It is comparable across configs that differ in edit strength.
A do(digit) control gives the same slope for an edit with no colour component.
"""
import json
import os

import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SW = os.path.join(REPO, "experiments/hdae/outputs/morpho_sweep")
H5 = os.path.join(REPO, "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5")
# attr_embed_dim for each config. base/nonorm/etc are all 128 unless the name says otherwise.
EDIM = {"base": 128, "ed256": 256, "ed384": 384, "ed512": 512, "ednn": 256,
        "nonorm": 128, "mf100": 128, "f0": 128, "f8": 128, "f32": 128,
        "ad015": 128, "cd020": 128, "fusesum": 128}

with h5py.File(H5, "r") as h:
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["attribute_names"][:]]
    A = h["attrs"][:].astype(np.float64)
COL = {k: names.index(k) for k in names}
BASE = {k: float(np.abs(A[:, COL[k]] - A[:, COL[k]].mean()).mean()) for k in ["intensity", "thickness"]}


def damage_at(dose, dmg, targets):
    """Interpolate collateral damage at matched hue-dose, on the ASCENDING branch only.

    The raw curve is U-shaped: at low g the sample is a poor reconstruction, so damage is high
    while dose is still near zero. That left branch is generic weak-guidance degradation, not
    entanglement, and a straight-line fit through both branches measures neither. Everything
    from the minimum onward is the branch where pushing the hue edit harder costs intensity,
    which is the effect in question.
    """
    d = np.asarray(dose); y = np.asarray(dmg)
    k = int(np.argmin(y))
    d, y = d[k:], y[k:]
    order = np.argsort(d); d, y = d[order], y[order]
    out = []
    for t in targets:
        out.append(float(np.interp(t, d, y)) if d.min() <= t <= d.max() else float("nan"))
    return out


TARGETS = [0.90, 0.94, 0.97]
rows = []
for t in sorted(EDIM):
    p = os.path.join(SW, f"sweep_tune_{t}.json")
    if not os.path.exists(p):
        continue
    js = json.load(open(p)); r = js["results"]; S = js["strengths"]
    fl = {k: r["recon"][k]["value"] for k in ["intensity", "thickness", "hue"]}
    dose, dmg_i, dmg_t, ctrl = [], [], [], []
    for g in S:
        c = r[f"do(hue)|g{g:g}"]
        dose.append(1.0 - c["hue"]["value"])
        dmg_i.append((c["intensity"]["value"] - fl["intensity"]) / BASE["intensity"])
        dmg_t.append((c["thickness"]["value"] - fl["thickness"]) / BASE["thickness"])
        d = r[f"do(digit)|g{g:g}"]
        ctrl.append((d["intensity"]["value"] - fl["intensity"]) / BASE["intensity"])
    rows.append((EDIM[t], t, damage_at(dose, dmg_i, TARGETS),
                 damage_at(dose, dmg_t, TARGETS), max(dose), float(np.mean(ctrl))))

print("Excess intensity damage under do(hue), INTERPOLATED AT MATCHED HUE DOSE.")
print("Units: MAE above the reconstruction floor, divided by the best-constant baseline.")
print("Matched dose is what makes configs comparable -- a model that lands the hue edit")
print("harder would otherwise look worse for that reason alone.\n")
hdr = "".join(f"{'dose='+format(t,'.2f'):>11s}" for t in TARGETS)
print(f"{'edim':>5s} {'config':9s}{hdr}   {'max dose':>9s} {'digit ctrl':>11s}")
for e, t, di, dt, md, ct in sorted(rows):
    print(f"{e:5d} {t:9s}" + "".join(f"{x:11.4f}" for x in di) + f"   {md:9.4f} {ct:11.4f}")

print("\nSame, for THICKNESS damage:")
print(f"{'edim':>5s} {'config':9s}{hdr}")
for e, t, di, dt, md, ct in sorted(rows):
    print(f"{e:5d} {t:9s}" + "".join(f"{x:11.4f}" for x in dt))

ser = [(e, t, di) for e, t, di, *_ in sorted(rows) if t in ("base", "ed256", "ed384", "ed512")]
if len(ser) >= 2:
    print("\nattr_embed_dim series (one-factor-at-a-time from base):")
    print(f"{'edim':>5s} {'config':9s}{hdr}")
    for e, t, di in ser:
        print(f"{e:5d} {t:9s}" + "".join(f"{x:11.4f}" for x in di))
