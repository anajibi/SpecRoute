"""Build fixed intervention cohorts for Causal3DIdent, one per modelled attribute.

A cohort is a fixed list of test-set images plus, for each image, the value we intend to
intervene that attribute to. Fixing them up front means every model and every guidance
scale is later scored on identical images with identical targets, so numbers are
comparable across runs rather than across separately-sampled draws.

Targets are chosen to make the edit *measurable*:
  class    a different class, deterministic (source + 3) mod 7 -- always a real change
  scalar   a value drawn from the attribute's own range, rejecting draws closer than
           `min_delta` to the source, so we never "intervene" to where it already was.
           For vector attributes every component is re-drawn.

Saved as a single npz per attribute: indices, source attribute rows, and targets.

    python experiments/hdae/scripts/build_cohorts_c3di.py --n 256
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from experiments.hdae.data.causal3dident import (ATTRIBUTE_NAMES, CLASS_NAMES,  # noqa: E402
                                                 Causal3DIdentPacked)

# attribute -> (dataset columns, kind, n_classes or None)
ATTRS = {
    "class":   ([ATTRIBUTE_NAMES.index("class")], "categorical", 7),
    "pos_spl": ([ATTRIBUTE_NAMES.index("pos_spl")], "scalar", None),
    "pos_obj": ([ATTRIBUTE_NAMES.index(f"pos_obj_{j}") for j in range(3)], "scalar", None),
    "rot_obj": ([ATTRIBUTE_NAMES.index(f"rot_obj_{j}") for j in range(3)], "scalar", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="images per cohort")
    ap.add_argument("--min-delta", type=float, default=0.5,
                    help="minimum |target - source| for scalar attributes, in [-1,1] units")
    ap.add_argument("--lo", type=float, default=-0.9)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h5", default="experiments/hdae/data/causal3dident/causal3dident_testset_128.h5")
    ap.add_argument("--outdir", default=os.path.join(REPO, "experiments/hdae/outputs/cohorts_c3di"))
    args = ap.parse_args()

    ds = Causal3DIdentPacked(os.path.join(REPO, args.h5))
    attr = ds.attr                      # (N, 11) float32, already in raw units
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    summary = {}

    for name, (cols, kind, k) in ATTRS.items():
        idx = np.sort(rng.choice(len(ds), args.n, replace=False))
        src = attr[idx][:, cols].astype(np.float32)

        if kind == "categorical":
            tgt = ((src[:, 0].astype(int) + 3) % k).astype(np.float32)[:, None]
            changed = float((tgt[:, 0] != src[:, 0]).mean())
            detail = {"targets_differ_from_source": changed,
                      "target_class_hist": {CLASS_NAMES[c]: int((tgt[:, 0] == c).sum()) for c in range(k)}}
        else:
            tgt = np.empty_like(src)
            for j in range(src.shape[1]):
                t = rng.uniform(args.lo, args.hi, size=len(idx)).astype(np.float32)
                # re-draw anything too close to the source: an "intervention" that lands
                # where the attribute already was measures nothing.
                for _ in range(200):
                    bad = np.abs(t - src[:, j]) < args.min_delta
                    if not bad.any():
                        break
                    t[bad] = rng.uniform(args.lo, args.hi, size=int(bad.sum()))
                tgt[:, j] = t
            d = np.abs(tgt - src)
            detail = {"min_abs_delta": float(d.min()), "mean_abs_delta": float(d.mean()),
                      "target_range": [float(tgt.min()), float(tgt.max())],
                      "source_range": [float(src.min()), float(src.max())]}

        p = os.path.join(args.outdir, f"{name}.npz")
        np.savez(p, indices=idx, source=src, target=tgt,
                 cols=np.array(cols), kind=kind, attribute=name)
        summary[name] = {"n": int(len(idx)), "kind": kind, "dims": int(src.shape[1]),
                         "columns": cols, "file": os.path.relpath(p, REPO), **detail}
        print(f"{name:9s} n={len(idx)} dims={src.shape[1]} -> {os.path.relpath(p, REPO)}")
        for kk, vv in detail.items():
            print(f"          {kk}: {vv}")

    with open(os.path.join(args.outdir, "cohorts.json"), "w") as f:
        json.dump({"n_per_cohort": args.n, "min_delta": args.min_delta,
                   "target_range": [args.lo, args.hi], "seed": args.seed,
                   "source_h5": args.h5, "cohorts": summary}, f, indent=2)
    print(f"\nwrote {os.path.join(args.outdir, 'cohorts.json')}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
