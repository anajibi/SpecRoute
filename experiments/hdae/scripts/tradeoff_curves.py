"""CC-versus-FC trade-off curves, one point per guidance strength, plus a comparable AUC.

Guidance buys edit success at the cost of collateral damage. Sweeping g therefore traces a
parametric curve in the (FC, CC) plane -- exactly the shape of an ROC curve, with the same
reading: a model whose curve sits up and to the right dominates, regardless of where on it you
choose to operate. Comparing single operating points hides that; comparing curves does not.

THE AUC HAS TO BE MADE COMPARABLE, AND THE OBVIOUS WAY FAILS. Each model spans a different FC
range, so a raw trapezoid over each curve's own span measures different intervals. The first
attempt integrated over the range COMMON to all three -- but for several interventions that range
is EMPTY, because the deeper models preserve strictly better than k=1 at every single g. Disjoint
ranges are a real result (domination), not a bug, but they leave nothing to integrate over.

So the integral runs over the UNION of the three ranges, with each curve extended to that union by
holding its endpoints. The extension is meaningful rather than invented: beyond a model's
best-preservation end it would need g < 1, which does not exist, so its CC there is the CC it
achieves at g=1; beyond its other end it would need g > 8, so its CC is the CC at g=8. The
integral is divided by the union width, giving:

    "the average CC this model achieves across the preservation range ANY of them can reach"

on the same 0-1 scale as CC. The fraction of the interval that is extrapolated is reported
alongside, since a number carried mostly by held endpoints deserves less weight.

    python experiments/hdae/scripts/tradeoff_curves.py
"""
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOD = ["class", "pos_spl", "pos_obj", "rot_obj"]
UNOBS = ["hue_obj", "hue_spl", "hue_bg"]
MODELS = [("k=1", "k1_n1024"), ("k=5", "k5_n1024"), ("k=11", "k11_n1024")]
S = [1, 1.5, 2, 2.5, 3, 5, 8]
CLIP = lambda v: min(1.0, max(0.0, v))


def curves():
    ag = json.load(open(f"{REPO}/experiments/hdae/outputs/cfg_sweep/aggregate_n1024.json"))
    out = {}
    for nm, lab in MODELS:
        for a in MOD:
            cc, fo, fu, fa = [], [], [], []
            for g in S:
                cell = ag["models"][lab]["per_g"][f"{g:g}"]["per_intervention"][a]
                cc.append(CLIP(np.mean([CLIP(v) for v in cell["cc_per_attr"].values()])))
                obs = [CLIP(v) for k, v in cell["fc_per_attr"].items() if k in MOD]
                un = [CLIP(v) for k, v in cell["fc_per_attr"].items() if k in UNOBS]
                fo.append(float(np.mean(obs)) if obs else float("nan"))
                fu.append(float(np.mean(un)))
                # FC_all: every attribute that should have held still, modelled or not
                fa.append(float(np.mean(obs + un)))
            out[(nm, a)] = dict(cc=np.array(cc), fc_all=np.array(fa),
                                fc_obs=np.array(fo), fc_unobs=np.array(fu))
    return out


def auc_common(cur, a, key):
    """Mean CC over the FC interval ANY model reaches; curves extended by held endpoints."""
    lo = min(cur[(nm, a)][key].min() for nm, _ in MODELS)
    hi = max(cur[(nm, a)][key].max() for nm, _ in MODELS)
    grid = np.linspace(lo, hi, 400)
    res, ext = {}, {}
    for nm, _ in MODELS:
        d = cur[(nm, a)]
        o = np.argsort(d[key])
        x, y = d[key][o], d["cc"][o]
        # np.interp holds the endpoint values outside the sampled range, which is exactly the
        # extension described above
        res[nm] = float(np.trapz(np.interp(grid, x, y), grid) / (hi - lo))
        ext[nm] = float(((grid < x.min()) | (grid > x.max())).mean())
    return res, lo, hi, ext


def main():
    cur = curves()
    tables = {}
    for key, name in [("fc_all", "FC (all held attributes)"),
                      ("fc_obs", "FC observed (modelled, non-descendant)"),
                      ("fc_unobs", "FC unobserved (the three hues)")]:
        print(f"\n=== normalised AUC vs {name}")
        print(f"{'intervention':13s} " + "".join(f"{nm:>9s}" for nm, _ in MODELS)
              + f"{'union FC range':>22s} {'width':>7s}")
        rows = []
        for a in MOD:
            r, lo, hi, ext = auc_common(cur, a, key)
            print(f"{a:13s} " + "".join(f"{r[nm]:9.4f}" for nm, _ in MODELS)
                  + f"   [{lo:.4f}, {hi:.4f}] {hi-lo:7.4f}   extrapolated "
                  + "/".join(f"{ext[nm]*100:.0f}%" for nm, _ in MODELS))
            rows.append((a, r, lo, hi, ext))
        tables[key] = rows
    json.dump({"auc": {k: [(a, r, lo, hi, e) for a, r, lo, hi, e in v] for k, v in tables.items()},
               "curves": {f"{nm}|{a}": {kk: list(map(float, vv)) for kk, vv in cur[(nm, a)].items()}
                          for nm, _ in MODELS for a in MOD},
               "strengths": S},
              open(f"{REPO}/experiments/hdae/outputs/tradeoff.json", "w"), indent=2)
    print(f"\nwrote {REPO}/experiments/hdae/outputs/tradeoff.json")


if __name__ == "__main__":
    main()
