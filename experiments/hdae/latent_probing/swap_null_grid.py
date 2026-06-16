#!/usr/bin/env python
"""Generate a diagnostic grid for HDAE latent swaps and learned null tokens.

Rows are: originals, donors, swap Z1, swap Z2, swap Z3, swap Z1+Z2,
swap Z2+Z3, null Z1, null Z2, null Z3. Level indices are zero-based in code
but labeled as Z1/Z2/Z3 for readability.
"""
import argparse, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from experiments.hdae.hdae.grid_utils import save_labeled_grid

from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

SWAP_ROWS = [
    ("swap_Z1", [0]),
    ("swap_Z2", [1]),
    ("swap_Z3", [2]),
    ("swap_Z1_Z2", [0, 1]),
    ("swap_Z2_Z3", [1, 2]),
]
NULL_ROWS = [
    ("null_Z1", [0]),
    ("null_Z2", [1]),
    ("null_Z3", [2]),
]


def validate_three_levels(model):
    num_levels = len(model.hdae_conf.encoder.level_dims)
    if num_levels < 3:
        raise ValueError(f"swap/null grid needs at least 3 latent levels, got {num_levels}")


def merge_from_levels(model, zs):
    return model.merge(zs)


def swapped_zs(source_zs, donor_zs, levels):
    out = [z.clone() for z in source_zs]
    for level in levels:
        out[level] = donor_zs[level]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--T", type=int, default=None)
    args = p.parse_args()
    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    loader = DataLoader(ds, batch_size=args.num_images * 2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    if len(batch["img"]) < args.num_images * 2:
        raise ValueError("dataset must provide at least 2 * num_images examples for source/donor swaps")
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    model = module.ema_model
    validate_three_levels(model)
    x_src = batch["img"][:args.num_images].to(device)
    x_donor = batch["img"][args.num_images:args.num_images * 2].to(device)
    T = args.T or cfg.raw["train"]["T_eval"]
    logging.info("encoding %d source and donor images", args.num_images)
    rows, labels = [x_src.add(1).div(2), x_donor.add(1).div(2)], ["source", "donor"]
    with torch.no_grad():
        src = model.encode(x_src)
        donor = model.encode(x_donor)
        x_t = module.encode_stochastic(x_src, src["cond"], T=T)
        for label, levels in SWAP_ROWS:
            logging.info("rendering %s using donor levels %s", label, levels)
            cond = merge_from_levels(model, swapped_zs(src["zs"], donor["zs"], levels))
            rows.append(module.render(x_t, cond, T=T).clamp(0, 1)); labels.append(label)
        for label, levels in NULL_ROWS:
            logging.info("rendering %s using learned null token levels %s", label, levels)
            cond = model.encode_with_nulls(x_src, levels)["cond"]
            rows.append(module.render(x_t, cond, T=T).clamp(0, 1)); labels.append(label)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    save_labeled_grid([row.detach().cpu() for row in rows], labels, output)
    meta = {"output": str(output), "row_labels": labels,
            "source_indices": batch["index"][:args.num_images].tolist(),
            "donor_indices": batch["index"][args.num_images:args.num_images * 2].tolist()}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logging.info("wrote grid to %s with rows: %s", output, labels)


if __name__ == "__main__":
    main()
