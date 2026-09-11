"""Paired bootstrap over the shared 512-image cohort for the tuning comparison.

morpho_metrics.py reports point estimates. On 512 images a 0.01-0.02 CC gap between
two configs is not necessarily a real ordering, and the six tuning runs were evaluated
on the SAME cohort with the SAME seed and the SAME intervention targets -- so the
comparison is paired and the paired bootstrap is far tighter than treating the configs
as independent samples.

Every config is resampled with the SAME index draws on each of the B replicates, which
is what makes the difference distribution valid: the cohort-composition noise that
dominates a 512-image estimate cancels in the difference.

Bootstrapped quantities, all recomputed from per-sample errors rather than from the
point-estimate JSON:
  floor_k  = mean recon error on the resampled indices   (resampled)
  move_k   = mean |src - tgt| on the resampled indices   (resampled)
  base_k   = best constant predictor over all 25,200 test rows (FIXED -- a definitional
             constant of the dataset, not an estimate from the cohort)
"""
import argparse, glob, json, os
import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOD = ["digit", "thickness", "intensity", "hue"]
UNOBS = ["rotation", "scale", "translate_x", "translate_y",
         "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude"]
ALL12 = MOD + UNOBS
CATEGORICAL = {"digit", "hue"}
EDGES = [("thickness", "intensity")]


def descendants(a):
    out, fr = set(), [a]
    while fr:
        n = fr.pop()
        for p, c in EDGES:
            if p == n and c not in out:
                out.add(c); fr.append(c)
    return out


def cls_idx(k, v):
    return np.round(v) if k == "digit" else np.round(np.asarray(v) * 10 - 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default=os.path.join(REPO, "experiments/hdae/outputs/morpho_sweep"))
    ap.add_argument("--h5", default=os.path.join(REPO, "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5"))
    ap.add_argument("--ref", default="base", help="config every other one is differenced against")
    ap.add_argument("-B", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/hdae/outputs/tune_bootstrap.json"))
    a = ap.parse_args()

    with h5py.File(a.h5, "r") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["attribute_names"][:]]
        A_all = h["attrs"][:].astype(np.float64)
    COL = {k: names.index(k) for k in ALL12}
    base = {}
    for k in ALL12:
        v = A_all[:, COL[k]]
        if k in CATEGORICAL:
            p = np.bincount(cls_idx(k, v).astype(int), minlength=10) / len(v)
            base[k] = float(1.0 - p.max())
        else:
            base[k] = float(np.abs(v - v.mean()).mean())

    tags, data = [], {}
    for jp in sorted(glob.glob(os.path.join(a.sweeps, "sweep_tune_*.json"))):
        t = os.path.basename(jp)[len("sweep_tune_"):-len(".json")]
        js = json.load(open(jp))
        npz = np.load(os.path.join(a.sweeps, f"persample_tune_{t}.npz"))
        tags.append(t)
        data[t] = (js, {k: npz[k] for k in npz.files})
    if a.ref not in tags:
        raise SystemExit(f"ref {a.ref} not among {tags}")

    S = data[tags[0]][0]["strengths"]
    n = data[tags[0]][0]["n"]
    # Shared draws: the SAME B x n index matrix scores every config.
    rng = np.random.default_rng(0)
    IDX = rng.integers(0, n, size=(a.B, n))

    def curves(t):
        """(B+1, len(S)) CC and CF1; row 0 is the point estimate on the full cohort."""
        js, P = data[t]
        cidx = np.asarray(js["cohort_idx"])
        # sel[0] = identity (point estimate), sel[1:] = the B resamples
        sel = np.vstack([np.arange(n)[None, :], IDX])
        floor = {k: P[f"recon|{k}"][sel].mean(1) for k in ALL12}
        move = {}
        for m in MOD:
            src = A_all[cidx, COL[m]]
            tgt = np.asarray(js["targets"][m], dtype=np.float64)
            if m in CATEGORICAL:
                d = (cls_idx(m, src) != cls_idx(m, tgt)).astype(np.float64)
            else:
                d = np.abs(src - tgt)
            move[m] = d[sel].mean(1)
        CC = np.zeros((a.B + 1, len(S))); CF = np.zeros_like(CC)
        for gi, g in enumerate(S):
            ccm, fam = [], []
            for m in MOD:
                dsc = descendants(m)
                cc_parts, fc_parts = [], []
                for k in ALL12:
                    err = P[f"do({m})|g{g:g}|{k}"][sel].mean(1)
                    fl = floor[k]
                    num = np.maximum(0.0, err - fl)
                    if k == m:
                        den = np.maximum(move[m] - fl, 1e-9)
                        cc_parts.append(np.clip(1 - num / den, 0, 1))
                    elif k in dsc:
                        cc_parts.append(np.clip(1 - num / np.maximum(base[k] - fl, 1e-9), 0, 1))
                    else:
                        fc_parts.append(np.clip(1 - num / np.maximum(base[k] - fl, 1e-9), 0, 1))
                cc = np.mean(cc_parts, 0); fa = np.mean(fc_parts, 0)
                ccm.append(cc); fam.append(fa)
            # CF1 is the harmonic mean PER INTERVENTION, then averaged -- matching
            # morpho_metrics.py. Pooling CC and FC first and taking one harmonic mean
            # is a different statistic (it hides an intervention that is bad at both).
            cf_per = [np.where(c + f > 0, 2 * c * f / np.maximum(c + f, 1e-12), 0.0)
                      for c, f in zip(ccm, fam)]
            CC[:, gi] = np.mean(ccm, 0)
            CF[:, gi] = np.mean(cf_per, 0)
        return CC, CF

    res = {t: curves(t) for t in tags}
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out = {"B": a.B, "n": n, "ref": a.ref, "strengths": S, "configs": {}}

    # best-g is chosen ONCE on the point estimate, then held fixed across resamples --
    # re-arg-maxing inside every replicate would bias each config's score upward by
    # letting it pick whichever strength the noise happened to favour.
    best = {t: int(np.argmax(res[t][0][0])) for t in tags}

    print(f"Paired bootstrap, B={a.B}, n={n}, shared resamples, best-g fixed at point estimate\n")
    print(f"{'config':8s} {'g':>4s} {'CC':>8s} {'CC 95% CI':>18s} {'CF1':>8s} {'CF1 95% CI':>18s}")
    for t in tags:
        CC, CF = res[t]; b = best[t]
        cc, cf = CC[1:, b], CF[1:, b]
        lo, hi = q(cc); flo, fhi = q(cf)
        print(f"{t:8s} {S[b]:4g} {CC[0,b]:8.4f} [{lo:7.4f},{hi:7.4f}] {CF[0,b]:8.4f} [{flo:7.4f},{fhi:7.4f}]")
        out["configs"][t] = dict(best_g=S[b], CC=float(CC[0, b]), CC_ci=[lo, hi],
                                 CF1=float(CF[0, b]), CF1_ci=[flo, fhi])

    rb = best[a.ref]
    print(f"\nPaired difference vs {a.ref} (each config at its own best g, {a.ref} at g={S[rb]:g})\n")
    print(f"{'config':8s} {'dCC':>9s} {'dCC 95% CI':>20s} {'P(>ref)':>9s}   {'dCF1':>9s} {'dCF1 95% CI':>20s} {'P(>ref)':>9s}")
    for t in tags:
        if t == a.ref:
            continue
        b = best[t]
        dcc = res[t][0][1:, b] - res[a.ref][0][1:, rb]
        dcf = res[t][1][1:, b] - res[a.ref][1][1:, rb]
        lo, hi = q(dcc); flo, fhi = q(dcf)
        p1 = float((dcc > 0).mean()); p2 = float((dcf > 0).mean())
        star = "  *" if (lo > 0 or hi < 0) else "   "
        print(f"{t:8s} {res[t][0][0,b]-res[a.ref][0][0,rb]:9.4f} [{lo:8.4f},{hi:8.4f}] {p1:9.4f}{star} "
              f"{res[t][1][0,b]-res[a.ref][1][0,rb]:9.4f} [{flo:8.4f},{fhi:8.4f}] {p2:9.4f}")
        out["configs"][t].update(dCC=float(res[t][0][0, b] - res[a.ref][0][0, rb]), dCC_ci=[lo, hi],
                                 P_CC_gt_ref=p1, dCF1=float(res[t][1][0, b] - res[a.ref][1][0, rb]),
                                 dCF1_ci=[flo, fhi], P_CF1_gt_ref=p2)
    print("\n  * = 95% CI on the paired difference excludes zero")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
