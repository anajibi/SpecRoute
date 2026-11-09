"""Build the evaluation cohort for Causal3DIdent.

Same contract as the CelebA builder: for every test image and every conditioned attribute, the
factual vector, the POST-INTERVENTION vector after the fitted SCM has propagated, which
attributes moved, and whether the result is in support.

THREE DIFFERENCES FROM CELEBA, all forced by the data:

 1. The conditioned attributes are not binary. `class` is a 7-way categorical; `pos_spl`,
    `pos_obj` and `rot_obj` are continuous and two of them are vector-valued (3 dims each).
    "Flip" is therefore defined per kind (protocol section 7.2):
        categorical            v -> (v + 1 + U{0, K-2}) mod K, seeded
        bounded symmetric      v -> -v          (reflection through the range centre)
        angular                v -> v + pi      (which in raw-latent units, where [-1,1]
                                                 maps to [-pi,pi], is v -> wrap(v + 1))
    `rot_obj` is the angular one. Raw latents are already bounded to [-1,1], so reflection
    keeps every value in range by construction.

 2. There is effectively NO out-of-support region. Every class spans the full [-1,1] on every
    conditioned dimension, so reflection lands outside a class's observed range for 0.032% of
    values (boundary rounding only). CelebA by contrast has 6 of 16 joint cells empty. The
    support rule is still applied -- it is the protocol's rule, not CelebA's -- it simply
    excludes almost nothing here. That asymmetry is a property of a rendered dataset with
    full-support marginals, and is worth stating rather than discovering later.

 3. The cohort is written as npz + a JSON header, not one JSON per attribute: 25,200 test rows
    x 4 attributes is 100,800 interventions, which as JSON would be ~40 MB of mostly repeated
    floats.

Three further factors -- hue_obj, hue_spl, hue_bg -- are rendered but conditioned on by
nothing. They are the FC_unobserved pool: anything that moves there is collateral damage by
construction. They are carried in the cohort so FC can be scored without re-reading the data.
"""
import argparse, json, os, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from able.shared.scm_registry import load_c3di                   # noqa: E402

COLS = {"pos_obj": [0, 1, 2], "rot_obj": [3, 4, 5], "pos_spl": [6],
        "hue_obj": [7], "hue_spl": [8], "hue_bg": [9]}
CONDITIONED = ["class", "pos_spl", "pos_obj", "rot_obj"]
UNMODELED = ["hue_obj", "hue_spl", "hue_bg"]
ANGULAR = {"rot_obj"}                      # +pi flip rather than reflection
N_CLASS = 7


def wrap(v):
    """wrap into [-1, 1), the raw-latent image of [-pi, pi)"""
    return (v + 1.0) % 2.0 - 1.0


def load_split(root, split):
    xs, cs = [], []
    for c in range(N_CLASS):
        a = np.load(f"{root}/{split}/raw_latents_{c}.npy")
        xs.append(a); cs.append(np.full(len(a), c, np.int64))
    return np.concatenate(xs).astype(np.float64), np.concatenate(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/hdae/data/causal3dident")
    ap.add_argument("--scm", default="experiments/hdae/outputs/scm/causal3dident_scm.pt")
    ap.add_argument("--out", default="outputs/cohorts/causal3dident")
    ap.add_argument("--seed", type=int, default=20261109)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    Xte, Cte = load_split(a.root, "testset")
    Xtr, Ctr = load_split(a.root, "trainset")
    Xall, Call = np.concatenate([Xtr, Xte]), np.concatenate([Ctr, Cte])
    n = len(Xte)
    print(f"test rows {n:,}   support reference {len(Xall):,} (train+test)")

    # class-conditional support, per conditioned continuous column
    sup = {c: {j: (Xall[Call == c, j].min(), Xall[Call == c, j].max())
               for name in ("pos_obj", "rot_obj", "pos_spl") for j in COLS[name]}
           for c in range(N_CLASS)}

    scm = load_c3di(os.path.join(REPO, a.scm))
    print(f"SCM loaded {scm._loaded_with}")

    def to_attrs(X, C):
        d = {"class": torch.from_numpy(C.astype(np.float32)).unsqueeze(1)}
        for nm in ("pos_spl", "pos_obj", "rot_obj"):
            d[nm] = torch.from_numpy(X[:, COLS[nm]].astype(np.float32))
        return d

    fact = to_attrs(Xte, Cte)
    summary = {}
    for attr in CONDITIONED:
        if attr == "class":
            newc = (Cte + 1 + rng.integers(0, N_CLASS - 1, n)) % N_CLASS
            inter = {"class": torch.from_numpy(newc.astype(np.float32)).unsqueeze(1)}
        else:
            v = Xte[:, COLS[attr]]
            nv = wrap(v + 1.0) if attr in ANGULAR else -v
            inter = {attr: torch.from_numpy(nv.astype(np.float32))}
        with torch.no_grad():
            cf = scm.counterfactual(fact, inter)
        cf = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in cf.items()}
        cf[attr] = inter[attr].numpy().astype(np.float64)        # intervention is exact
        cls_post = cf["class"].reshape(-1).round().astype(np.int64) if attr == "class" else Cte

        keep = np.ones(n, bool)
        for nm in ("pos_obj", "rot_obj", "pos_spl"):
            vals = cf[nm]
            for k, j in enumerate(COLS[nm]):
                lo = np.array([sup[c][j][0] for c in cls_post])
                hi = np.array([sup[c][j][1] for c in cls_post])
                keep &= (vals[:, k] >= lo - 1e-6) & (vals[:, k] <= hi + 1e-6)

        moved = {nm: int((np.abs(cf[nm] - (Cte.reshape(-1,1) if nm=="class" else Xte[:, COLS[nm]])).max(1) > 1e-6).sum())
                 for nm in CONDITIONED}
        np.savez_compressed(
            os.path.join(a.out, f"do_{attr}.npz"),
            row=np.arange(n)[keep], factual_class=Cte[keep],
            factual=Xte[keep], target_class_idx=cls_post[keep],
            **{f"target_{nm}": cf[nm][keep] for nm in CONDITIONED},
            unmodeled=Xte[:, [COLS[u][0] for u in UNMODELED]][keep], admitted=keep[keep])
        summary[attr] = {"admitted": int(keep.sum()), "excluded": int((~keep).sum()),
                         "moved_counts": moved}
        print(f"  do({attr:8s}) admitted {int(keep.sum()):6d}/{n}  excluded {int((~keep).sum()):5d}   moved {moved}")

    hdr = {"dataset": "causal3dident128", "n_test": n, "seed": a.seed,
           "conditioned": CONDITIONED, "unmodeled_fc_pool": UNMODELED,
           "latent_columns": COLS, "class_names": ["teapot","hare","dragon","cow","armadillo","horse","head"],
           "value_space": "raw latents, every column bounded to [-1,1]; [-1,1] maps to [-pi,pi] for angular factors",
           "intervention": {"class": "cyclic (v+1+U{0,K-2}) mod K, seeded",
                            "pos_spl": "v -> -v", "pos_obj": "v -> -v",
                            "rot_obj": "v -> wrap(v+1), i.e. +pi"},
           "support_rule": "post-intervention value inside the class-conditional [min,max] of "
                           "the full train+test set, per dimension",
           "scm": {"path": a.scm, "loaded_with": scm._loaded_with,
                   "note": "must be the same SCM used at evaluation time"},
           "per_attribute": summary}
    json.dump(hdr, open(os.path.join(a.out, "cohort_summary.json"), "w"), indent=1)
    tot = sum(v["admitted"] for v in summary.values())
    print(f"\ntotal admitted interventions: {tot:,}")


if __name__ == "__main__":
    main()
