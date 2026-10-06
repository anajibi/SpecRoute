"""Freeze the evaluation cohort so every baseline is scored on identical subjects.

The cohort is 2048 held-out males drawn from partitions 1 and 2 (the generative models here
train on partition 0 only). Bald is flipped in BOTH directions: 121 of the 2048 are already
bald and are asked to become haired, 1927 the reverse.

Writing the indices to disk rather than re-deriving them matters: the draw depends on a seed
AND on the partition/attribute arrays, so a baseline on another cluster that reconstructs the
cohort itself could silently get a different 2048 subjects and produce numbers that look
comparable but are not.
"""
import json, numpy as np, sys
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.data.celeba_hq import CelebAHQPacked

OBS = ["Male", "Young", "Beard", "Bald"]
ds = CelebAHQPacked("experiments/hdae/data/packed/celebahq_256.lmdb",
                    "experiments/hdae/data/packed/celebahq_256_attrs.npz")
names = list(ds.attribute_names)
cols = [names.index(a) for a in OBS]
pool = np.where((ds.partitions > 0) & (ds.attrs[:, names.index("Male")] == 1))[0]
idx = np.sort(np.random.default_rng(7).choice(pool, 2048, replace=False))

out = {
    "n": 2048,
    "seed": 7,
    "pool_rule": "partitions 1,2 (held out from generative training) AND Male == +1",
    "pool_size": int(len(pool)),
    "indices": [int(i) for i in idx],
    "observed_attrs": OBS,
    "observed_cols": cols,
    "factual": {a: [int(v) for v in ds.attrs[idx, c]] for a, c in zip(OBS, cols)},
    "note": "factual values are the raw CelebA +1/-1 labels. A counterfactual for attribute A "
            "flips ONLY that column; all others keep their factual value.",
}
json.dump(out, open("experiments/hdae/baselines/cohort_2048.json", "w"))
for a, c in zip(OBS, cols):
    pos = int((ds.attrs[idx, c] == 1).sum())
    print(f"  {a:6} {pos:5d} positive / {2048-pos:5d} negative")
print(f"\nwrote cohort_2048.json  ({len(idx)} indices from a pool of {len(pool)})")
