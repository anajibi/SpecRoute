#!/usr/bin/env python
"""Render cumulative semantic-latent reveals while reusing abducted x_T.

For a fixed batch, the script abducts both semantic latents ``Z`` and the
stochastic DDIM code ``x_T``. It then decodes rows where different subsets of
Z are kept and all other levels are replaced by their learned null tokens:

* original images
* reconstruction with all Z levels
* all Z levels nulled
* forward cumulative reveal: Z0, Z0+Z1, ..., Z0+...+Z{K-1}
* reverse cumulative reveal: Z-1, Z-1+Z-2, ..., Z-1+...+Z-K

The zero-based Z0 notation is used here to match latent tensor indices. In the
reverse rows, Z-1 means the final/fine-most latent level.
"""
import argparse, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def complement_levels(num_levels, keep_levels):
    keep = {int(level) for level in keep_levels}
    invalid = sorted(level for level in keep if level < 0 or level >= num_levels)
    if invalid:
        raise ValueError(f"invalid keep levels {invalid}; valid levels are 0..{num_levels - 1}")
    return [level for level in range(num_levels) if level not in keep]


def forward_cumulative_rows(num_levels):
    if num_levels < 1:
        raise ValueError("at least one latent level is required")
    return [(f"forward_{'_'.join(f'Z{i}' for i in range(n))}", list(range(n)))
            for n in range(1, num_levels + 1)]


def reverse_cumulative_rows(num_levels):
    if num_levels < 1:
        raise ValueError("at least one latent level is required")
    rows = []
    for n in range(1, num_levels + 1):
        levels = list(range(num_levels - n, num_levels))
        labels = [f"Z-{i}" for i in range(1, n + 1)]
        rows.append((f"reverse_{'_'.join(labels)}", levels))
    return rows


def render_with_kept_levels(module, model, x_t, zs, keep_levels, num_levels, T):
    null_levels = complement_levels(num_levels, keep_levels)
    cond = model.merge(zs, null_levels=null_levels)
    return module.render(x_t, cond, T=T).clamp(0, 1), null_levels


def main():
    import torch
    from torch.utils.data import DataLoader
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.grid_utils import save_labeled_grid
    from experiments.hdae.hdae.lit_module import HDAELitModule

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
    loader = DataLoader(ds, batch_size=args.num_images, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    if len(batch["img"]) < args.num_images:
        raise ValueError(f"dataset yielded {len(batch['img'])} images, need {args.num_images}")

    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    model = module.ema_model
    num_levels = len(model.hdae_conf.encoder.level_dims)
    T = args.T or cfg.raw["train"]["T_eval"]
    x = batch["img"][:args.num_images].to(device)

    logging.info("abducting Z and x_T for %d images with %d latent levels", args.num_images, num_levels)
    rows, labels, row_meta = [x.add(1).div(2)], ["original"], []
    with torch.no_grad():
        encoded = model.encode(x)
        x_t = module.encode_stochastic(x, encoded["cond"], T=T)
        rows.append(module.render(x_t, encoded["cond"], T=T).clamp(0, 1))
        labels.append("reconstruction_all_Z")
        row_meta.append({"label": "reconstruction_all_Z", "keep_levels": list(range(num_levels)), "null_levels": []})

        for label, keep_levels in [("all_Z_null", [])] + forward_cumulative_rows(num_levels) + reverse_cumulative_rows(num_levels):
            logging.info("rendering %s; keeping levels %s", label, keep_levels)
            rendered, null_levels = render_with_kept_levels(module, model, x_t, encoded["zs"],
                                                            keep_levels, num_levels, T)
            rows.append(rendered); labels.append(label)
            row_meta.append({"label": label, "keep_levels": keep_levels, "null_levels": null_levels})

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    save_labeled_grid([row.detach().cpu() for row in rows], labels, output)
    meta = {
        "output": str(output),
        "num_levels": num_levels,
        "T": T,
        "row_labels": labels,
        "rows": row_meta,
        "indices": batch["index"][:args.num_images].tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logging.info("wrote abductive Z/x_T grid to %s with rows: %s", output, labels)


if __name__ == "__main__":
    main()
