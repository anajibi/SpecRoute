"""Build the evaluation cohort for CelebA-HQ, one file per (graph, attribute).

A cohort file tells an evaluator everything it needs to perform one intervention and nothing
it has to infer: which source image, what the factual attribute vector is, what the
POST-INTERVENTION vector is after the SCM has propagated, which attributes therefore moved,
and in both codings (logical 0/1 and the model's -1/+1).

Three things it settles that were previously implicit:

PROPAGATION. do(Male) is not "set Male and leave the rest". Male is a parent of Beard and Bald,
so the fitted SCM abducts the exogenous noise from the factual vector, applies the intervention
and re-predicts: descendants take their post-intervention values and the model is conditioned on
the whole vector. Graph B has no edges, so there the flip IS the counterfactual -- that
asymmetry is why the two graphs are built by the same code but produce different `moved` sets.

SUPPORT. An intervention is emitted only if the post-intervention joint cell is attested --
count >= MIN_SUPPORT in the full 30,000. Checked AFTER propagation, which is why it is
interventions on CHILD attributes that get dropped: do(Beard=1) leaves Male fixed and lands on
a bearded female, 10 images in 30,000 and 0 in the test partition. Parent interventions
propagate and stay in support. Out-of-support interventions are therefore removed naturally by
the rule rather than by a hand-maintained exclusion list.

CODING. CelebA stores attributes as {-1, +1}; the SCM and all counting here work in {0, 1}.
Both are written out so an evaluator can never pick the wrong one silently.
"""
import argparse, json, os, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from experiments.hdae.causal.scm import SCM                      # noqa: E402

GRAPHS = {
    "A": {"attrs": ["Male", "Young", "Beard", "Bald"],
          "scm": "experiments/hdae/outputs/scm/celebahq_scm.pt"},
    "B": {"attrs": ["Eyeglasses", "Smiling"], "scm": None},      # no edges: flip is the CF
}
MIN_SUPPORT = 10
TEST_PARTITION = 2


def cell_counts(cols):
    """count of every joint cell over the FULL dataset, keyed by tuple of 0/1"""
    n_attr = len(cols)
    out = {}
    for code in range(1 << n_attr):
        key = tuple((code >> j) & 1 for j in range(n_attr))
        m = np.ones(len(cols[0]), bool)
        for j, v in enumerate(key):
            m &= (cols[j] == v)
        out[key] = int(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", default="experiments/hdae/data/packed/celebahq_256_attrs.npz")
    ap.add_argument("--out", default="outputs/cohorts/celebahq")
    ap.add_argument("--graphs", default="A,B")
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    ap.add_argument("--seed", type=int, default=20261109)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.attrs, allow_pickle=True)
    A, names, part = d["attrs"], list(d["attribute_names"]), d["partitions"]
    ids = d["image_ids"]
    idx = {n: i for i, n in enumerate(names)}
    test_rows = np.flatnonzero(part == TEST_PARTITION)
    print(f"test rows: {len(test_rows)}")

    summary = {}
    for g in a.graphs.split(","):
        spec = GRAPHS[g]
        attrs = spec["attrs"]
        cols_all = [(A[:, idx[n]] > 0).astype(np.int64) for n in attrs]
        counts = cell_counts(cols_all)
        fact = np.stack([c[test_rows] for c in cols_all], axis=1)      # (n_test, n_attr) 0/1

        scm = None
        if spec["scm"]:
            scm = SCM.load(os.path.join(REPO, spec["scm"]))
            torch.manual_seed(a.seed)

        g_summary = {}
        for ai, attr in enumerate(attrs):
            tgt = fact.copy()
            tgt[:, ai] = 1 - tgt[:, ai]                                 # the flip
            if scm is not None:
                t = torch.from_numpy(fact).float()
                ai_map = {n: i for i, n in enumerate(attrs)}
                inter = {attr: torch.from_numpy(tgt[:, ai]).float().unsqueeze(1)}
                cf = scm.counterfactual(t, ai_map, inter)
                for j, n in enumerate(attrs):
                    v = cf[n].detach().cpu().numpy().reshape(-1)
                    tgt[:, j] = (v > 0.5).astype(np.int64)
                tgt[:, ai] = 1 - fact[:, ai]                            # intervention is exact
            keep = np.array([counts[tuple(r)] >= a.min_support for r in tgt])
            items = []
            for k in np.flatnonzero(keep):
                f, t_ = fact[k].tolist(), tgt[k].tolist()
                moved = [attrs[j] for j in range(len(attrs)) if f[j] != t_[j]]
                items.append({
                    "row": int(test_rows[k]), "image_id": str(ids[test_rows[k]]),
                    "factual01": f, "target01": t_,
                    "factual_model": [2 * x - 1 for x in f],
                    "target_model": [2 * x - 1 for x in t_],
                    "flip": {"attr": attr, "from": f[ai], "to": t_[ai]},
                    "moved": moved,
                    "support_post": counts[tuple(t_)],
                })
            rec = {"dataset": "celebahq256", "graph": g, "attribute": attr,
                   "conditioning_order": attrs,
                   "value_coding": {"factual01/target01": "0 = negative, 1 = positive",
                                    "factual_model/target_model": "-1/+1, the coding "
                                    "causal_graph_celebahq*.yaml declares and the model consumes"},
                   "intervention": "flip: v -> 1 - v, then SCM abduct/act/predict"
                                   if scm else "flip: v -> 1 - v (graph has no edges)",
                   "support_rule": f"post-intervention joint cell count >= {a.min_support} "
                                   f"in the full 30,000",
                   "n_test": int(len(test_rows)), "n_admitted": int(keep.sum()),
                   "n_excluded_out_of_support": int((~keep).sum()),
                   "seed": a.seed, "items": items}
            p = os.path.join(a.out, f"graph{g}_{attr}.json")
            json.dump(rec, open(p, "w"), indent=1)
            n_prop = sum(1 for it in items if len(it["moved"]) > 1)
            g_summary[attr] = {"admitted": rec["n_admitted"], "excluded": rec["n_excluded_out_of_support"],
                               "with_propagation": n_prop}
            print(f"  graph {g}  do({attr}):  admitted {rec['n_admitted']:5d}/{len(test_rows)}   "
                  f"excluded {rec['n_excluded_out_of_support']:5d}   propagated {n_prop:5d}  -> {p}")
        summary[g] = g_summary
    json.dump({"min_support": a.min_support, "test_partition": TEST_PARTITION,
               "n_test": int(len(test_rows)), "graphs": summary},
              open(os.path.join(a.out, "cohort_summary.json"), "w"), indent=1)
    tot = sum(v["admitted"] for gg in summary.values() for v in gg.values())
    print(f"\ntotal admitted interventions across both graphs: {tot}")


if __name__ == "__main__":
    main()
