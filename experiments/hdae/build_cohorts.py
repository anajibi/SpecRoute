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
           "attributes": {}}
    for offset, attr in enumerate(parse_csv_list(args.attributes)):
        out["attributes"][attr] = sample_attr_indices(attrs, attr_names, attr, args.num_images, seed=args.seed + offset)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
