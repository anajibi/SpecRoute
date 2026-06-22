#!/usr/bin/env python
"""Generate a diagnostic grid for HDAE latent swaps and learned null tokens.

The row set is generated from the configured number of semantic latent levels:
source, donor, every single-level swap, every adjacent-pair swap, and every
single-level null-token ablation. Level indices are zero-based in code but
labeled as Z1..ZK for readability.
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

def z_label(level):
    return f"Z{level + 1}"


def dynamic_swap_rows(num_levels):
    """Return single-level and adjacent-pair swap rows for an arbitrary K."""
    if num_levels < 1:
        raise ValueError("swap/null grid needs at least one latent level")
    rows = [(f"swap_{z_label(i)}", [i]) for i in range(num_levels)]
    rows.extend((f"swap_{z_label(i)}_{z_label(i + 1)}", [i, i + 1])
                for i in range(num_levels - 1))
    return rows


def dynamic_null_rows(num_levels):
    """Return one learned-null-token ablation row per latent level."""
    if num_levels < 1:
        raise ValueError("swap/null grid needs at least one latent level")
    return [(f"null_{z_label(i)}", [i]) for i in range(num_levels)]


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
    num_levels = len(model.hdae_conf.encoder.level_dims)
    swap_rows = dynamic_swap_rows(num_levels)
    null_rows = dynamic_null_rows(num_levels)
    x_src = batch["img"][:args.num_images].to(device)
    x_donor = batch["img"][args.num_images:args.num_images * 2].to(device)
    T = args.T or cfg.raw["train"]["T_eval"]
    logging.info("encoding %d source and donor images", args.num_images)
    rows, labels = [x_src.add(1).div(2), x_donor.add(1).div(2)], ["source", "donor"]
    with torch.no_grad():
        src = model.encode(x_src)
        donor = model.encode(x_donor)
        x_t = module.encode_stochastic(x_src, src["cond"], T=T)
        for label, levels in swap_rows:
            logging.info("rendering %s using donor levels %s", label, levels)
            cond = merge_from_levels(model, swapped_zs(src["zs"], donor["zs"], levels))
            rows.append(module.render(x_t, cond, T=T).clamp(0, 1)); labels.append(label)
        for label, levels in null_rows:
            logging.info("rendering %s using learned null token levels %s", label, levels)
            cond = model.merge(src["zs"], null_levels=levels)
            rows.append(module.render(x_t, cond, T=T).clamp(0, 1)); labels.append(label)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    save_labeled_grid([row.detach().cpu() for row in rows], labels, output)
    meta = {"output": str(output), "num_levels": num_levels, "row_labels": labels,
            "source_indices": batch["index"][:args.num_images].tolist(),
            "donor_indices": batch["index"][args.num_images:args.num_images * 2].tolist()}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logging.info("wrote grid to %s with rows: %s", output, labels)


if __name__ == "__main__":
    main()
