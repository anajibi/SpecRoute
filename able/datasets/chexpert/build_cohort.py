"""CheXpert protocol v2, section 6: the frozen evaluation cohort.

512 held-out sources x 4 interventions = 2,048 counterfactuals.

Sources are drawn balanced over the 24 joint cells of
(sex, age_bin, effusion, edema) rather than in proportion to their natural
frequency. The rare cells are the informative ones -- a 25-year-old with both
findings is where a model that has merely learned the marginals gives itself
away -- and proportional sampling would put about three of them in the
cohort. After the certainty filter below every cell still holds at least 127
held-out images, so 21 or 22 per cell is drawn without any cell running dry.

v2: edema is modeled, so no source may carry an uncertain edema label. The
frozen row set cannot be refiltered (images_224.npy is aligned to it), so the
1,107 held-out rows whose edema label was uncertain are excluded from the
SOURCE POOL instead. This is a restriction of eligibility, not of the data,
and it is not attribute selection -- which attributes are modeled was decided
on radiologist_val, never here.

One target per intervention type: sex flips, age goes to one of the other two
bins drawn uniformly, each finding flips. do(age) is the only one with
descendants, and its propagated findings come from the SCM re-run on the
SOURCE'S OWN abducted noise -- so for a given source the findings move only
where the new age logits cross that unit's noise draw, which is what makes
the counterfactual a counterfactual rather than a fresh sample.

Every source is drawn from the held-out split. The cohort is never used to
choose which attributes are modeled; it is the thing being measured.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scm_exact as S

CELL = ["sex", "age_bin", "effusion", "edema"]
INTERVENTIONS = ["do_sex", "do_age", "do_effusion", "do_edema"]
DESCENDANTS_OF_AGE = ["effusion", "edema"]


def allocate(counts, total, rng):
    """Spread `total` draws as evenly as possible over cells, capped by supply."""
    keys = list(counts)
    take = {k: 0 for k in keys}
    left = total
    while left > 0:
        hungry = [k for k in keys if take[k] < counts[k]]
        if not hungry:
            raise SystemExit(f"held-out pool exhausted: {total - left} of {total} drawn")
        per = max(1, left // len(hungry))
        for k in rng.permutation(len(hungry)):
            k = hungry[k]
            n = min(per, counts[k] - take[k], left)
            take[k] += n
            left -= n
            if left == 0:
                break
    return take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-sources", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    out = args.out or f"{args.data}/eval_cohort.json"

    meta = pd.read_csv(f"{args.data}/chexpert_meta.csv")
    split = np.load(f"{args.data}/split.npy")
    params = S.load(f"{args.data}/scm_params.json")
    held = np.flatnonzero(split == 1)

    # modeled findings must be certain: drop ineligible sources before stratifying
    upath = f"{args.data}/label_uncertainty.csv"
    n_excluded = 0
    if S.CERTAIN_REQUIRED:
        if not os.path.exists(upath):
            raise SystemExit(f"{upath} is required when {S.CERTAIN_REQUIRED} is modeled")
        unc = pd.read_csv(upath)
        if not (unc["path"].values == meta["path"].values).all():
            raise SystemExit("label_uncertainty.csv is not row-aligned with chexpert_meta.csv")
        ok = np.ones(len(meta), bool)
        for v in S.CERTAIN_REQUIRED:
            ok &= (unc[f"uncertain_{v}"].to_numpy() == 0)
        n_excluded = int((~ok[held]).sum())
        held = held[ok[held]]
    hm = meta.iloc[held]

    rng = np.random.default_rng(args.seed)
    groups = {k: g.index.to_numpy() for k, g in hm.groupby(CELL)}
    take = allocate({k: len(v) for k, v in groups.items()}, args.n_sources, rng)
    rows = np.sort(np.concatenate([rng.choice(groups[k], take[k], replace=False)
                                   for k in groups if take[k]]))
    assert len(rows) == args.n_sources == len(set(rows.tolist()))
    assert (split[rows] == 1).all(), "a source escaped the held-out split"

    src = meta.iloc[rows]
    vals = {v: src[v].to_numpy(np.int64) for v in S.ORDER}

    # one noise draw per source, shared by all four of its counterfactuals
    g = S.abduct(vals, params, args.seed + 99)
    base = S.forward(vals, g, params)
    for v in S.ORDER:
        if not np.array_equal(base[v], vals[v]):
            raise AssertionError(f"abduction inconsistent on {v}")

    tgt_age = np.array([rng.choice([b for b in (0, 1, 2) if b != a]) for a in vals["age_bin"]])
    fixed = {
        "do_sex": ("sex", 1 - vals["sex"]),
        "do_age": ("age_bin", tgt_age),
        "do_effusion": ("effusion", 1 - vals["effusion"]),
        "do_edema": ("edema", 1 - vals["edema"]),
    }
    cf = {k: S.forward(vals, g, params, fixed={var: val}) for k, (var, val) in fixed.items()}

    # sanity: only the target and its descendants may move
    for k, (var, _) in fixed.items():
        allowed = {var} | (set(DESCENDANTS_OF_AGE) if var == "age_bin" else set())
        for v in S.ORDER:
            moved = cf[k][v] != vals[v]
            if v not in allowed and moved.any():
                raise AssertionError(f"{k} moved {v} on {int(moved.sum())} sources")

    items = []
    for i in range(len(rows)):
        r = int(rows[i])
        fac = {v: int(vals[v][i]) for v in S.ORDER}
        for k in INTERVENTIONS:
            var = fixed[k][0]
            c = {v: int(cf[k][v][i]) for v in S.ORDER}
            items.append({
                "source_row": r,
                "path": meta.at[r, "path"],
                "patient_id": int(meta.at[r, "patient_id"]),
                "intervention": k,
                "target_variable": var,
                "target_value": c[var],
                "factual": fac,
                "counterfactual": c,
                "changed": [v for v in S.ORDER if c[v] != fac[v]],
                "descendants_moved": [v for v in DESCENDANTS_OF_AGE
                                      if var == "age_bin" and c[v] != fac[v]],
            })

    rec = {
        "protocol": "chexpert v2 section 6",
        "seed": args.seed,
        "n_sources": int(len(rows)),
        "n_counterfactuals": len(items),
        "source_rows": rows.tolist(),
        "stratified_over": CELL,
        "heldout_sources_excluded_uncertain": n_excluded,
        "certainty_required_on": S.CERTAIN_REQUIRED,
        "cells_filled": len(take),
        "per_cell": {str(k): int(v) for k, v in take.items()},
        "meta_sha256": hashlib.sha256(open(f"{args.data}/chexpert_meta.csv", "rb").read()).hexdigest(),
        "scm_params_sha256": hashlib.sha256(open(f"{args.data}/scm_params.json", "rb").read()).hexdigest(),
        "items": items,
    }
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)

    if n_excluded:
        print(f"source pool: {len(held)} held-out rows "
              f"({n_excluded} excluded for an uncertain {'/'.join(S.CERTAIN_REQUIRED)} label)")
    print(f"{len(rows)} sources over {len(take)} cells "
          f"({min(take.values())}-{max(take.values())} each), {len(items)} counterfactuals")
    for k in INTERVENTIONS:
        n_desc = sum(1 for it in items if it["intervention"] == k and it["descendants_moved"])
        print(f"  {k:16s} sources whose descendants moved: {n_desc}")
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
