#!/usr/bin/env python
"""Build model-agnostic fixed CelebA attribute cohorts for CF consistency eval."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

import numpy as np

DEFAULT_ATTRIBUTES = ["Smiling", "Eyeglasses", "Male", "Young"]


def parse_csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def sample_attr_indices(attrs, attr_names, attr, num_images, seed=0):
    idx = attr_names.index(attr)
    rng = np.random.default_rng(seed)
    pos = np.where(attrs[:, idx] > 0)[0]
    neg = np.where(attrs[:, idx] <= 0)[0]
    if len(pos) < num_images or len(neg) < num_images:
        raise ValueError(f"attribute {attr} has only pos={len(pos)} neg={len(neg)} examples; need {num_images}")
    return {"pos_idx": rng.choice(pos, size=num_images, replace=False).astype(int).tolist(),
            "neg_idx": rng.choice(neg, size=num_images, replace=False).astype(int).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attr-npz", required=True)
    p.add_argument("--attributes", default=",".join(DEFAULT_ATTRIBUTES))
    p.add_argument("--num-images", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    arrays = np.load(args.attr_npz, allow_pickle=True)
    attrs = arrays["attrs"]
    attr_names = [str(x) for x in arrays["attribute_names"]]
    out = {"attr_npz": args.attr_npz, "num_images_per_side": args.num_images, "seed": args.seed,
           "attributes": {}, "intervention_weights": {}}
    for offset, attr in enumerate(parse_csv_list(args.attributes)):
        idx = attr_names.index(attr)
        pos_count = int((attrs[:, idx] > 0).sum())
        neg_count = int((attrs[:, idx] <= 0).sum())
        out["attributes"][attr] = sample_attr_indices(attrs, attr_names, attr, args.num_images, seed=args.seed + offset)
        # Direction weights are proportional to the prevalence of the target side
        # relative to the source side over the whole dataset. Example: if
        # positive:negative is 1:10, positive->negative gets weight 10 and
        # negative->positive gets weight 1.
        out["intervention_weights"][attr] = {
            "positive": float(pos_count / neg_count) if neg_count else 0.0,
            "negative": float(neg_count / pos_count) if pos_count else 0.0,
            "positive_count": pos_count,
            "negative_count": neg_count,
        }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    weights_csv = path.with_name(path.stem + "_intervention_weights.csv")
    with weights_csv.open("w") as f:
        f.write("attribute,direction,weight,positive_count,negative_count\n")
        for attr, info in out["intervention_weights"].items():
            for direction in ("positive", "negative"):
                f.write(f"{attr},{direction},{info[direction]},{info['positive_count']},{info['negative_count']}\n")


if __name__ == "__main__":
    main()
