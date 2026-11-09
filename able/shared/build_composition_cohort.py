"""Composition cohort: do(X=flip) then do(X=factual), repeated.

A model can score well on CC and FC and still have edits that do not compose -- apply the flip
and flip back, and you should land on the image you started from. Nothing else in the protocol
tests that. Cycle 1 catches an edit that is not invertible at all; cycle 10 catches slow drift
that a single round trip hides.

DELIBERATELY TINY, AND SEPARATE. Ten cycles is twenty generations per (source, attribute), so
this cannot run on the main cohort: at CelebA graph A's 8,242 interventions it would be ~160
hours. It gets its own small stratified subset instead, and its own file, so it can never be
confused with the CC/FC cohort or silently averaged into it.

Only interventions ADMITTED by the main cohort are eligible -- a cycle through an
out-of-support cell would measure the support rule, not composition.

The cohort stores the two endpoints, not twenty targets: cycle k alternates between `target`
(odd k) and `factual` (even k), deterministically, so the generator derives the schedule.
"""
import argparse, json, os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)


def celebahq(coh_dir, n, seed):
    out = {}
    rng = np.random.default_rng(seed)
    for f in sorted(os.listdir(coh_dir)):
        if not f.startswith("graph") or not f.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(coh_dir, f)))
        items = rec["items"]
        if not items:
            continue
        # stratify over the factual joint cell so rare cells are not lost in a random draw
        keys = [tuple(it["factual01"]) for it in items]
        uniq = sorted(set(keys))
        per = max(1, n // len(uniq))
        pick = []
        for u in uniq:
            idx = [i for i, k in enumerate(keys) if k == u]
            pick += list(rng.choice(idx, min(per, len(idx)), replace=False))
        pick = sorted(set(pick))[:n]
        out[f"{rec['graph']}_{rec['attribute']}"] = {
            "graph": rec["graph"], "attribute": rec["attribute"],
            "conditioning_order": rec["conditioning_order"],
            "n_sources": len(pick), "cells_represented": len(uniq),
            "items": [{"row": items[i]["row"], "image_id": items[i]["image_id"],
                       "factual_model": items[i]["factual_model"],
                       "target_model": items[i]["target_model"]} for i in pick]}
    return out


def c3di(coh_dir, n, seed):
    rng = np.random.default_rng(seed)
    hdr = json.load(open(os.path.join(coh_dir, "cohort_summary.json")))
    out = {}
    for attr in hdr["conditioned"]:
        z = np.load(os.path.join(coh_dir, f"do_{attr}.npz"))
        cls = z["factual_class"]
        per = max(1, n // 7)
        pick = []
        for c in range(7):
            idx = np.flatnonzero(cls == c)
            if len(idx): pick += list(rng.choice(idx, min(per, len(idx)), replace=False))
        pick = sorted(pick)[:n]
        out[f"c3di_{attr}"] = {
            "dataset": "causal3dident128", "attribute": attr,
            "n_sources": len(pick), "classes_represented": 7,
            "rows": z["row"][pick].tolist(),
            "factual_class": cls[pick].tolist(),
            "factual": z["factual"][pick].tolist(),
            "target": {k.replace("target_", ""): z[k][pick].tolist()
                       for k in z.files if k.startswith("target_") and k != "target_class_idx"}}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--celeba", default="outputs/cohorts/celebahq")
    ap.add_argument("--c3di", default="outputs/cohorts/causal3dident")
    ap.add_argument("--out", default="outputs/cohorts/composition")
    ap.add_argument("--n", type=int, default=64, help="sources per (graph, attribute)")
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261109)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    blocks = {}
    blocks.update(celebahq(a.celeba, a.n, a.seed))
    blocks.update(c3di(a.c3di, a.n, a.seed))
    hdr = {"what": "composition / cycle-consistency cohort",
           "cycles": a.cycles, "report_at": [1, a.cycles],
           "n_per_attribute": a.n, "seed": a.seed,
           "schedule": "cycle k applies `target` on odd k and `factual` on even k; after any "
                       "EVEN number of cycles the image should equal the reconstruction",
           "eligibility": "only interventions admitted by the main cohort",
           "blocks": {k: {kk: vv for kk, vv in v.items() if kk not in ("items", "rows", "factual", "target")}
                      for k, v in blocks.items()}}
    json.dump(hdr, open(os.path.join(a.out, "composition_summary.json"), "w"), indent=1)
    for k, v in blocks.items():
        json.dump(v, open(os.path.join(a.out, f"{k}.json"), "w"), indent=1)
    tot = sum(v["n_sources"] for v in blocks.values())
    gens = tot * 2 * a.cycles
    print(f"blocks {len(blocks)}   sources total {tot}   generations for {a.cycles} cycles: {gens:,}")
    for k, v in blocks.items():
        print(f"   {k:22s} {v['n_sources']:4d} sources")
    print(f"\n  ~{gens*100*2/1e6:.2f}M UNet evals  (~{gens*100*2/80/3600:.1f} h at 80/s for the 256 blocks)")
