#!/usr/bin/env python
"""Attribute-conditioning counterfactual evaluation with fixed latents."""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from hdae.counterfactuals.utils import summarize_attribute_changes

ROOT = Path(__file__).resolve().parents[3];
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.hdae.counterfactuals.attr_classifier import load_classifier
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_index_space
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.grid_utils import save_labeled_grid
from experiments.hdae.hdae.lit_module import HDAELitModule

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _probabilities(classifier, x):
    with torch.no_grad():
        return torch.sigmoid(classifier(x)).detach().cpu().numpy()


def rendered_to_classifier_input(x01):
    return x01.mul(2).sub(1).clamp(-1, 1)


def conditioning_attr_indices(model, dataset_attr_names):
    e = model.hdae_conf.encoder
    attrs = list(e.conditioning_attrs) if e.conditioning_attrs else list(dataset_attr_names[:e.n_attributes])
    missing = [a for a in attrs if a not in dataset_attr_names]
    if missing:
        raise ValueError(f"conditioning_attrs not found in dataset attributes: {missing}")
    return attrs, [dataset_attr_names.index(a) for a in attrs]


def preservation_indices(attr_names, conditioning_attrs):
    cond = set(conditioning_attrs)
    return [i for i, name in enumerate(attr_names) if name not in cond]


def _render_recon_and_cf(module, x, y_idx, target_cond_col, direction_sign, T):
    model = module.ema_model
    zs = [z.clone() for z in model.encode(x)]
    source_cond = model.make_cond(zs, y_idx)
    x_t = module.encode_stochastic(x, source_cond, T=T)
    recon0 = module.render(x_t, source_cond, T=T)
    y_cf = y_idx.clone()
    y_cf[:, target_cond_col] = 1 if direction_sign == "positive" else 0
    cf = module.render(x_t, model.make_cond(zs, y_cf), T=T)
    return recon0, cf, y_cf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--attribute", default="Smiling", help="Must be one of encoder.conditioning_attrs")
    p.add_argument("--direction", choices=["positive", "negative", "both"], default="both")
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir);
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    classifier, clf_state = load_classifier(device=device)
    attr_names = [str(x) for x in clf_state["attribute_names"]]
    target_index = attr_names.index(args.attribute)
    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    cond_attrs, cond_indices = conditioning_attr_indices(module.ema_model, ds.attribute_names)
    if args.attribute not in cond_attrs:
        raise ValueError(
            f"attribute {args.attribute!r} is not in encoder.conditioning_attrs={cond_attrs}; conditioning-only CF can only toggle conditioned attributes")
    target_cond_col = cond_attrs.index(args.attribute)
    preserve_idx = preservation_indices(attr_names, cond_attrs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    directions = ["positive", "negative"] if args.direction == "both" else [args.direction]
    all_rows, first_grid = [], None
    T = args.T or cfg.raw["train"]["T_eval"]
    seen = 0
    for batch in loader:
        if seen >= args.num_images:
            break
        x = batch["img"][:args.num_images - seen].to(device)
        y_raw = batch["attr"][:len(x), cond_indices].to(device)
        y_idx = to_index_space(y_raw, module.ema_model.hdae_conf.encoder.attr_input_range).to(device)
        real_probs = _probabilities(classifier, x)
        for direction_sign in directions:
            recon0, cf, y_cf = _render_recon_and_cf(module, x, y_idx, target_cond_col, direction_sign, T)
            recon0_probs = _probabilities(classifier, rendered_to_classifier_input(recon0))
            after = _probabilities(classifier, rendered_to_classifier_input(cf))
            delta = after - recon0_probs
            for local_i, index in enumerate(batch["index"][:len(x)].tolist()):
                rec = {"index": int(index), "direction": direction_sign,
                       "target_real": float(real_probs[local_i, target_index]),
                       "target_recon0": float(recon0_probs[local_i, target_index]),
                       "target_after": float(after[local_i, target_index]),
                       "target_delta": float(delta[local_i, target_index]),
                       "source_condition_value": int(y_idx[local_i, target_cond_col].detach().cpu()),
                       "cf_condition_value": int(y_cf[local_i, target_cond_col].detach().cpu())}
                for j, name in enumerate(attr_names):
                    rec[f"recon0_{name}"] = float(recon0_probs[local_i, j])
                    rec[f"after_{name}"] = float(after[local_i, j])
                    rec[f"delta_{name}"] = float(delta[local_i, j])
                all_rows.append(rec)
            if first_grid is None:
                first_grid = (x.add(1).div(2).detach().cpu(), recon0.clamp(0, 1).detach().cpu(),
                              cf.clamp(0, 1).detach().cpu(), direction_sign)
        seen += len(x)
        logging.info("processed %d/%d images", min(seen, args.num_images), args.num_images)

    csv_path = out / f"counterfactual_{args.attribute}_conditioning.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader();
        writer.writerows(all_rows)
    summaries = {}
    for direction_sign in directions:
        rows = [r for r in all_rows if r["direction"] == direction_sign]
        recon0 = np.asarray([[r[f"recon0_{name}"] for name in attr_names] for r in rows])
        after = np.asarray([[r[f"after_{name}"] for name in attr_names] for r in rows])
        summaries[direction_sign] = summarize_attribute_changes(recon0, after, target_index,
                                                                preservation_indices=preserve_idx)
    summary = {"attribute": args.attribute, "target_index": target_index,
               "edit_mechanism": "conditioning_signal_only_fixed_latents",
               "conditioning_attrs": cond_attrs,
               "preservation_attrs": [attr_names[i] for i in preserve_idx],
               "csv": str(csv_path), "baseline": "self_reconstruction", "summaries": summaries}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if first_grid is not None:
        orig_row, recon_row, cf_row, direction_label = first_grid
        save_labeled_grid([orig_row, recon_row, cf_row],
                          ["original", "recon0", f"cf_{args.attribute}_{direction_label}"],
                          out / "counterfactual_grid.png")
    logging.info("wrote %s and summary.json", csv_path)


if __name__ == "__main__":
    main()
