#!/usr/bin/env python
"""Render cumulative semantic-latent reveals while reusing abducted x_T."""
import argparse, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def forward_cumulative_rows(num_levels):
    return [(f"forward_{'_'.join(f'Z{i}' for i in range(n))}", list(range(n)))
            for n in range(1, num_levels + 1)]


def reverse_cumulative_rows(num_levels):
    return [(f"reverse_{'_'.join(f'Z-{i}' for i in range(1, n + 1))}", list(range(num_levels - n, num_levels)))
            for n in range(1, num_levels + 1)]


def keep_only(zs, keep_levels):
    keep = set(keep_levels)
    return [z.clone() if i in keep else z.clone().zero_() for i, z in enumerate(zs)]


def conditioning_indices(model, attr_names):
    return [attr_names.index(name) for name in model.hdae_conf.encoder.conditioning_attrs]


def main():
    import torch
    from torch.utils.data import DataLoader
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.attr_utils import to_index_space
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
    batch = next(iter(DataLoader(ds, batch_size=args.num_images, shuffle=False, num_workers=0)))
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    model = module.ema_model
    num_levels = len(model.hdae_conf.encoder.level_dims)
    T = args.T or cfg.raw["train"]["T_eval"]
    x = batch["img"][:args.num_images].to(device)
    cond_idx = conditioning_indices(model, ds.attribute_names)
    y_idx = to_index_space(batch["attr"][:args.num_images, cond_idx].to(device), model.hdae_conf.encoder.attr_input_range)

    rows, labels, row_meta = [x.add(1).div(2)], ["original"], []
    with torch.no_grad():
        zs = model.encode(x)
        cond = model.make_cond(zs, y_idx)
        x_t = module.encode_stochastic(x, cond, T=T)
        rows.append(module.render(x_t, cond, T=T).clamp(0, 1))
        labels.append("reconstruction_all_Z")
        row_meta.append({"label": "reconstruction_all_Z", "keep_levels": list(range(num_levels)), "zero_levels": []})
        for label, keep_levels in [("all_Z_zero", [])] + forward_cumulative_rows(num_levels) + reverse_cumulative_rows(num_levels):
            z_keep = keep_only(zs, keep_levels)
            rendered = module.render(x_t, model.make_cond(z_keep, y_idx), T=T).clamp(0, 1)
            rows.append(rendered); labels.append(label)
            row_meta.append({"label": label, "keep_levels": keep_levels,
                             "zero_levels": [i for i in range(num_levels) if i not in set(keep_levels)]})

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    save_labeled_grid([row.detach().cpu() for row in rows], labels, output)
    meta = {
        "output": str(output),
        "num_levels": num_levels,
        "T": T,
        "row_labels": labels,
            "rows": row_meta, "indices": batch["index"][:args.num_images].tolist()}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logging.info("wrote abductive Z/x_T grid to %s", output)


if __name__ == "__main__":
    main()
