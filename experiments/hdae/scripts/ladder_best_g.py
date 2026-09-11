"""Best guidance strength per depth k, chosen ONCE per k by pooling its seeds.

Choosing g per seed would let each run pick whichever strength its own noise happened to
favour, which inflates every model and makes the depths incomparable -- the same trap as
re-arg-maxing inside a bootstrap replicate. So CC is averaged across the seeds of a given k
at each g, and the argmax of that averaged curve is the strength k is then scored at.

Two selections are reported because they can disagree:
  per-intervention   the strength that maximises CC for that intervention alone
  pooled             one strength for all four, maximising mean CC

Writes attr_g/<k>.txt as `digit=..  thickness=..  intensity=..  hue=..` for cfg_sweep_morpho.py.
"""
import argparse, glob, json, os, re
import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SW = os.path.join(REPO, "experiments/hdae/outputs/morpho_sweep")
MOD = ["digit", "thickness", "intensity", "hue"]
UNOB = ["rotation", "scale", "translate_x", "translate_y",
        "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude"]
CAT = {"digit", "hue"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO, "experiments/hdae/outputs/ladder_attr_g"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

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

    runs = {}
    for p in sorted(glob.glob(os.path.join(SW, "sweep_ladder_k*_s*.json"))):
        m = re.search(r"sweep_ladder_k(\d+)_s(\d+)\.json$", p)
        runs.setdefault(int(m.group(1)), {})[int(m.group(2))] = p
    if not runs:
        raise SystemExit("no ladder sweeps found")

    def cc_grid(path):
        """CC per (intervention, g) for one run."""
        js = json.load(open(path)); r = js["results"]; S = js["strengths"]
        fl = {k: r["recon"][k]["value"] for k in MOD + UNOB}
        out = {}
        for m in MOD:
            desc = ["intensity"] if m == "thickness" else []
            row = []
            for g in S:
                cc = []
                for k in [m] + desc:
                    err = r[f"do({m})|g{g:g}"][k]["value"]; f0 = fl[k]
                    if k == m:
                        src = A[js["cohort_idx"], COL[k]]; tg = np.asarray(js["targets"][m], float)
                        if k in CAT:
                            si = np.round(src) if k == "digit" else np.round(src * 10 - 0.5)
                            ti = np.round(tg) if k == "digit" else np.round(tg * 10 - 0.5)
                            mv = float((si != ti).mean())
                        else:
                            mv = float(np.abs(src - tg).mean())
                        den = max(mv - f0, 1e-9)
                    else:
                        den = max(BASE[k] - f0, 1e-9)
                    cc.append(np.clip(1 - max(0.0, err - f0) / den, 0, 1))
                row.append(float(np.mean(cc)))
            out[m] = np.array(row)
        return S, out

    report = {}
    for K in sorted(runs):
        seeds = sorted(runs[K])
        S, first = cc_grid(runs[K][seeds[0]])
        acc = {m: [first[m]] for m in MOD}
        for sd in seeds[1:]:
            _, gr = cc_grid(runs[K][sd])
            for m in MOD:
                acc[m].append(gr[m])
        # POOLED ACROSS SEEDS before the argmax -- this is the whole point.
        mean = {m: np.mean(acc[m], 0) for m in MOD}
        per_int = {m: S[int(np.argmax(mean[m]))] for m in MOD}
        pooled_curve = np.mean([mean[m] for m in MOD], 0)
        pooled_g = S[int(np.argmax(pooled_curve))]
        report[K] = dict(seeds=seeds, strengths=S, per_intervention_g=per_int,
                         pooled_g=pooled_g,
                         cc_curve={m: mean[m].tolist() for m in MOD},
                         cc_at_per_int={m: float(mean[m].max()) for m in MOD},
                         cc_at_pooled={m: float(mean[m][S.index(pooled_g)]) for m in MOD})
        with open(os.path.join(a.outdir, f"k{K}.txt"), "w") as f:
            f.write(" ".join(f"{m}={per_int[m]:g}" for m in MOD))

    print("CC averaged over seeds, per intervention (the curve g is chosen on)\n")
    for K in sorted(report):
        R = report[K]
        print(f"k={K}  seeds {R['seeds']}")
        print(f"  {'intervention':13s} " + "".join(f"{'g='+format(g,'g'):>9s}" for g in R["strengths"])
              + f"{'best g':>8s}{'CC':>9s}")
        for m in MOD:
            c = R["cc_curve"][m]
            print(f"  {m:13s} " + "".join(f"{x:9.4f}" for x in c)
                  + f"{R['per_intervention_g'][m]:8g}{R['cc_at_per_int'][m]:9.4f}")
        print(f"  pooled g = {R['pooled_g']:g}   mean CC at pooled g = "
              f"{np.mean(list(R['cc_at_pooled'].values())):.4f}   "
              f"at per-intervention g = {np.mean(list(R['cc_at_per_int'].values())):.4f}\n")
    json.dump(report, open(os.path.join(a.outdir, "best_g.json"), "w"), indent=2, default=str)
    print(f"wrote {a.outdir}/best_g.json and k*.txt")


if __name__ == "__main__":
    main()
