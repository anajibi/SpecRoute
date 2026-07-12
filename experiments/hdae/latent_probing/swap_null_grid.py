#!/usr/bin/env python
"""Generate a diagnostic grid for conditional HDAE latent swaps and zero ablations."""
import argparse, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def z_label(level):
    return f"Z{level + 1}"


def dynamic_swap_rows(num_levels):
    rows = [(f"swap_{z_label(i)}", [i]) for i in range(num_levels)]
    rows.extend((f"swap_{z_label(i)}_{z_label(i + 1)}", [i, i + 1]) for i in range(num_levels - 1))
    return rows


def dynamic_zero_rows(num_levels):
    return [(f"zero_{z_label(i)}", [i]) for i in range(num_levels)]


def dynamic_null_rows(num_levels):
    return [(f"null_{z_label(i)}", [i]) for i in range(num_levels)]


def replace_levels(source_zs, donor_zs, levels):
    out = [z.clone() for z in source_zs]
    for level in levels:
        out[level] = donor_zs[level]
    return out


def zero_levels(zs, levels):
    out = [z.clone() for z in zs]
    for level in levels:
        out[level] = out[level].zero_()
    return out


def swapped_zs(source_zs, donor_zs, levels):
    return replace_levels(source_zs, donor_zs, levels)


def conditioning_indices(model, attr_names):
    return [attr_names.index(name) for name in model.hdae_conf.encoder.conditioning_attrs]


def main():
    import torch
    from torch.utils.data import DataLoader
    from experiments.hdae.hdae.attr_utils import to_index_space
    from experiments.hdae.hdae.grid_utils import save_labeled_grid
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.config_io import load_hdae_config
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
    loader = DataLoader(ds, batch_size=args.num_images * 2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    model = module.ema_model
    num_levels = len(model.hdae_conf.encoder.level_dims)
    x_src = batch["img"][:args.num_images].to(device)
    x_donor = batch["img"][args.num_images:args.num_images * 2].to(device)
    cond_idx = conditioning_indices(model, ds.attribute_names)
    y_src = to_index_space(batch["attr"][:args.num_images, cond_idx].to(device), model.hdae_conf.encoder.attr_input_range)
    T = args.T or cfg.raw["train"]["T_eval"]
    rows, labels = [x_src.add(1).div(2), x_donor.add(1).div(2)], ["source", "donor"]
    with torch.no_grad():
        src_zs = model.encode(x_src)
        donor_zs = model.encode(x_donor)
        src_cond = model.make_cond(src_zs, y_src)
        x_t = module.encode_stochastic(x_src, src_cond, T=T)
        rows.append(module.render(x_t, src_cond, T=T).clamp(0, 1)); labels.append("source_recon")
        for label, levels in dynamic_swap_rows(num_levels):
            cond = model.make_cond(replace_levels(src_zs, donor_zs, levels), y_src)
            rows.append(module.render(x_t, cond, T=T).clamp(0, 1)); labels.append(label)
        for label, levels in dynamic_zero_rows(num_levels):
            cond = model.make_cond(zero_levels(src_zs, levels), y_src)
            rows.append(module.render(x_t, cond, T=T).clamp(0, 1)); labels.append(label)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    save_labeled_grid([row.detach().cpu() for row in rows], labels, output)
    meta = {"output": str(output), "num_levels": num_levels, "row_labels": labels,
            "source_indices": batch["index"][:args.num_images].tolist(),
            "donor_indices": batch["index"][args.num_images:args.num_images * 2].tolist()}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logging.info("wrote grid to %s", output)


if __name__ == "__main__":
    main()
