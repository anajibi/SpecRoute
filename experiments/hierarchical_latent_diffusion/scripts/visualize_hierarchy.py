#!/usr/bin/env python3
"""Generate image grids from existing stage-one and stage-two checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.batches import batch_image_ids, batch_images
from src.datasets import image_loader
from src.pipeline import load_models, resample_tail
from src.utils import get_device, load_config, output_dir, seed_everything
from src.visualization import save_image_grid, save_labeled_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-images", type=int, default=8, help="Number of test images shown in intervention grids")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of unconditional prior samples")
    parser.add_argument("--steps", type=int, default=50, help="DDIM inference steps per sampled latent")
    parser.add_argument("--seed", type=int, help="Override the config seed")
    parser.add_argument("--output-dir", type=Path, help="Override the default <experiment output>/visualizations directory")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"] if args.seed is None else args.seed)
    device = get_device(cfg["device"])
    destination = args.output_dir or output_dir(cfg) / "visualizations"
    destination.mkdir(parents=True, exist_ok=True)

    dino, vae, encoder, decoder, priors = load_models(cfg, device)
    batch = next(iter(image_loader(cfg, split="test", shuffle=False, max_images=args.num_images)))
    images = batch_images(batch, device)
    image_ids = batch_image_ids(batch)
    zs = encoder(dino(images))
    reconstruction = vae.decode(decoder(zs))

    tail_columns = [images, reconstruction]
    tail_labels = ["original", "reconstruction"]
    for levels_kept in range(cfg["K"] + 1):
        modified = resample_tail(zs, priors, levels_kept, args.steps) if levels_kept < cfg["K"] else zs
        tail_columns.append(vae.decode(decoder(modified)))
        tail_labels.append(f"keep {levels_kept}/{cfg['K']}")
    save_labeled_grid(tail_columns, tail_labels, destination / "tail_resampling.png", image_ids)

    level_columns = [images, reconstruction]
    level_labels = ["original", "reconstruction"]
    for level in range(cfg["K"]):
        modified = [z.clone() for z in zs]
        condition = None if level == 0 else torch.cat(modified[:level], dim=-1)
        modified[level] = priors.levels[level].sample(images.shape[0], condition, device, args.steps)
        level_columns.append(vae.decode(decoder(modified)))
        level_labels.append(f"replace Z{level + 1}")
    save_labeled_grid(level_columns, level_labels, destination / "single_level_interventions.png", image_ids)

    sampled_zs = priors.sample_full(args.num_samples, device, args.steps)
    generated = vae.decode(decoder(sampled_zs))
    save_image_grid(generated, destination / "unconditional_samples.png", nrow=min(4, args.num_samples))
    print(f"Saved visualization grids to {destination}")


if __name__ == "__main__":
    main()
