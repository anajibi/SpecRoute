"""CheXpert protocol v0, section 4: the patient-level 90/10 split.

Splitting on PATIENTS, not on images. A patient contributes a median of two
studies and up to dozens, and two frontals of the same chest a week apart are
near-duplicates; an image-level split would put those on both sides and the
held-out AUC would be reporting memorisation. 61,407 patients -> 171,996
images, so the leak would be large.

Writes split.npy (int8, aligned row-for-row with chexpert_meta.csv: 0 = train,
1 = held out) plus chexpert_split.json carrying the seed, the patient id lists,
the realised image counts and the attribute balance on both sides. The
alignment is by position, so the meta CSV's sha256 is recorded alongside and
every consumer is expected to check it.
"""
import argparse, hashlib, json
import numpy as np
import pandas as pd

ATTRS = ["sex", "age_bin", "effusion", "cardiomegaly"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-npy", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--heldout-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta)
    pids = np.sort(meta["patient_id"].unique())
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(pids)
    n_held = int(round(len(pids) * args.heldout_frac))
    held = np.sort(perm[:n_held])
    train = np.sort(perm[n_held:])

    assert len(np.intersect1d(held, train)) == 0
    assert len(held) + len(train) == len(pids)

    split = meta["patient_id"].isin(set(held.tolist())).to_numpy().astype(np.int8)
    np.save(args.out_npy, split)

    rec = {
        "seed": args.seed,
        "heldout_frac": args.heldout_frac,
        "meta": args.meta,
        "meta_sha256": sha256(args.meta),
        "n_images": int(len(meta)),
        "n_patients": int(len(pids)),
        "patients_train": int(len(train)),
        "patients_heldout": int(len(held)),
        "images_train": int((split == 0).sum()),
        "images_heldout": int((split == 1).sum()),
        "patient_overlap": 0,
        "balance": {
            s: {a: ({str(k): round(float(v), 6)
                     for k, v in meta.loc[split == i, a].value_counts(normalize=True).sort_index().items()}
                    if a == "age_bin" else round(float(meta.loc[split == i, a].mean()), 6))
                for a in ATTRS}
            for i, s in ((0, "train"), (1, "heldout"))
        },
        "heldout_patient_ids": held.tolist(),
    }
    with open(args.out_json, "w") as f:
        json.dump(rec, f, indent=2)
    rec["split_npy_sha256"] = sha256(args.out_npy)
    with open(args.out_json, "w") as f:
        json.dump(rec, f, indent=2)

    print(f"patients  train {len(train):6d}   held out {len(held):6d}   overlap 0")
    print(f"images    train {rec['images_train']:6d}   held out {rec['images_heldout']:6d}")
    for a in ATTRS:
        print(f"  {a:13s} train {rec['balance']['train'][a]}   heldout {rec['balance']['heldout'][a]}")
    print(f"wrote {args.out_npy} and {args.out_json}")


if __name__ == "__main__":
    main()
