"""CheXpert protocol v2: fit the SCM's logits on the TRAIN SPLIT only.

Saturated, so the fit is exact: one logit vector per parent configuration
reproduces the training conditionals to floating-point. Nothing is estimated
on the held-out split, because the evaluation cohort is drawn from there and
the SCM is the thing the cohort is used to measure.

v2: edema is a modeled finding, and modeled findings have to be certain. The
frozen row set was filtered when edema was still eval-only, so its 10,105
uncertain rows carry a folded 0 in chexpert_meta.csv and cannot be removed
(images_224.npy is aligned to that row set and the archive it was built from
is gone). Those rows are therefore excluded from edema's FIT via a mask read
off label_uncertainty.csv. The alternative -- fitting P(edema=1|age) on a
column where 5.88% of the rows say "absent" when the report said "uncertain"
-- would bias every conditional downward and push every do(age) propagation
with it.
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

    # modeled-finding certainty, applied as a fit mask (see the module docstring)
    upath = f"{args.data}/label_uncertainty.csv"
    masks, mask_note = {}, "none"
    if S.CERTAIN_REQUIRED:
        if not os.path.exists(upath):
            raise SystemExit(f"{upath} is required to fit {S.CERTAIN_REQUIRED}: "
                             f"rebuild it with build_meta.py")
        unc = pd.read_csv(upath)
        if not (unc["path"].values == meta["path"].values).all():
            raise SystemExit("label_uncertainty.csv is not row-aligned with chexpert_meta.csv")
        keep_tr = (unc.loc[split == 0]).reset_index(drop=True)
        for v in S.CERTAIN_REQUIRED:
            masks[v] = (keep_tr[f"uncertain_{v}"].to_numpy() == 0)
        mask_note = {v: f"excluded {int((~masks[v]).sum())} train rows with an uncertain label"
                     for v in masks}

    p = S.fit(vals, masks=masks)
    p["fit_on"] = "train split only (split.npy == 0)"
    p["certainty_masks"] = mask_note
    p["uncertainty_sha256"] = (hashlib.sha256(open(upath, "rb").read()).hexdigest()
                               if os.path.exists(upath) else None)
    p["meta_sha256"] = hashlib.sha256(open(f"{args.data}/chexpert_meta.csv", "rb").read()).hexdigest()
    p["seed"] = args.seed
    S.save(p, out)

    print("exact consistency on the fitting data:",
          S.check_consistency(vals, p, seed=args.seed, reps=8))
    for v, note in (mask_note.items() if isinstance(mask_note, dict) else []):
        print(f"  certainty mask on {v}: {note}")
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
