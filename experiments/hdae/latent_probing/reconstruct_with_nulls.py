#!/usr/bin/env python
"""Save reconstruction grids with selected HDAE latent levels replaced by null tokens."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from experiments.hdae.hdae.grid_utils import save_labeled_grid

from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.hdae.null_tokens import parse_null_levels, reconstruct_batch_with_null_levels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--null-levels", default="", help="Comma-separated level indices, e.g. '0' or '1,2'.")
    p.add_argument("--output", required=True)
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--T", type=int, default=None)
    args = p.parse_args()
    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    loader = DataLoader(ds, batch_size=args.num_images, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu")
    module.to(device).eval()
    x = batch["img"].to(device)
    levels = parse_null_levels(args.null_levels)
    with torch.no_grad():
        recon, encoded = reconstruct_batch_with_null_levels(module, x, levels, T=args.T or cfg.raw["train"]["T_eval"])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    # module.render returns [0, 1]; dataset x is [-1, 1]. Save originals followed by reconstructions.
    save_labeled_grid([x.add(1).div(2).detach().cpu(), recon.clamp(0, 1).detach().cpu()],
                      ["original", f"null_levels_{levels}"], output)
    mask = encoded["null_mask"].int().cpu().tolist()
    print({"output": str(output), "null_levels": levels, "null_mask_first_batch": mask})


if __name__ == "__main__":
    main()
