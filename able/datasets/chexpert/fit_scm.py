"""CheXpert protocol v0: fit the SCM's logits on the TRAIN SPLIT only.

Saturated, so the fit is exact: one logit vector per parent configuration
reproduces the training conditionals to floating-point. Nothing is estimated
on the held-out split, because the evaluation cohort is drawn from there and
the SCM is the thing the cohort is used to measure.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd

# python -I drops the script directory from sys.path; put it back so the
# sibling module resolves however this is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scm_exact as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    out = args.out or f"{args.data}/scm_params.json"

    meta = pd.read_csv(f"{args.data}/chexpert_meta.csv")
    split = np.load(f"{args.data}/split.npy")
    tr = meta.loc[split == 0]
    vals = {v: tr[v].to_numpy(np.int64) for v in S.ORDER}

    p = S.fit(vals)
    p["fit_on"] = "train split only (split.npy == 0)"
    p["meta_sha256"] = hashlib.sha256(open(f"{args.data}/chexpert_meta.csv", "rb").read()).hexdigest()
    p["seed"] = args.seed
    S.save(p, out)

    print("exact consistency on the fitting data:",
          S.check_consistency(vals, p, seed=args.seed, reps=8))
    for v in S.ORDER:
        tab = np.exp(np.asarray(p["logits"][v]))
        if not S.PARENTS[v]:
            print(f"  P({v}) = {np.round(tab[0], 4)}")
        else:
            for a in range(tab.shape[0]):
                print(f"  P({v}=1 | age_bin={a}) = {tab[a, 1]:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
