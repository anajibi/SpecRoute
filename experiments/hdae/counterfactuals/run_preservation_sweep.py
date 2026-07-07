#!/usr/bin/env python
"""Toggle attribute conditioning while measuring fixed-latent preservation drift."""
import argparse, csv, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.hdae.counterfactuals.directions import summarize_attribute_changes

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
DEFAULT_ATTRIBUTES = ["Smiling", "Eyeglasses", "Male", "Young"]


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]



def parse_strengths(value):
    """Backward-compatible parser; conditioning-only CF ignores strengths."""
    strengths = [float(item.strip()) for item in value.split(",") if item.strip()]
    return strengths if 0.0 in strengths else [0.0, *strengths]

def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def classifier_probs(classifier, x):
    import torch
    with torch.no_grad():
        return torch.sigmoid(classifier(x)).detach().cpu().numpy()


def rendered_to_classifier_input(x01):
    return x01.mul(2).sub(1).clamp(-1, 1)


def intended_target_flip_rate(before, after, target_index, direction_sign):
    before_pos = before[:, target_index] >= 0.5
    after_pos = after[:, target_index] >= 0.5
    if direction_sign == "positive":
        return float((~before_pos & after_pos).mean())
    return float((before_pos & ~after_pos).mean())


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


def _write_single_row_csv(path, row):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader(); writer.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    # Kept as ignored optional args so older wrappers fail less abruptly.
    p.add_argument("--probe-metrics", default=None, help="Ignored: CF generation changes conditioning, not latent directions.")
    p.add_argument("--probe-weights-dir", default=None, help="Ignored: CF generation changes conditioning, not latent directions.")
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--attributes", default=",".join(DEFAULT_ATTRIBUTES),
                   help="Comma-separated conditioned target attributes to toggle; Young is the age attribute.")
    p.add_argument("--direction", choices=["positive", "negative", "both"], default="both")
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-grids", action="store_true")
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from experiments.hdae.counterfactuals.attribute_classifier import load_classifier
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.attr_utils import to_index_space
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.grid_utils import save_labeled_grid
    from experiments.hdae.hdae.lit_module import HDAELitModule

    attributes = parse_csv_list(args.attributes)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    model = module.ema_model
    classifier, clf_state = load_classifier(args.attr_classifier, device=device)
    attr_names = [str(x) for x in clf_state["attribute_names"]]
    attr_to_idx = {name: i for i, name in enumerate(attr_names)}
    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    cond_attrs, cond_indices = conditioning_attr_indices(model, ds.attribute_names)
    missing_attrs = [attr for attr in attributes if attr not in attr_to_idx]
    if missing_attrs:
        raise ValueError(f"attributes not found in classifier: {missing_attrs}")
    not_conditioned = [attr for attr in attributes if attr not in cond_attrs]
    if not_conditioned:
        raise ValueError(f"conditioning-only CF can only toggle encoder.conditioning_attrs; not conditioned: {not_conditioned}")
    preserve_idx = preservation_indices(attr_names, cond_attrs)
    directions = ["positive", "negative"] if args.direction == "both" else [args.direction]
    T = args.T or cfg.raw["train"]["T_eval"]

    accum = {(attr, direction): {"base": [], "edit": []}
             for attr in attributes for direction in directions}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    first_grid = {}
    seen = 0
    for batch in loader:
        if seen >= args.num_images:
            break
        x = batch["img"][:args.num_images - seen].to(device)
        y_raw = batch["attr"][:len(x), cond_indices].to(device)
        y_idx = to_index_space(y_raw, model.hdae_conf.encoder.attr_input_range).to(device)
        with torch.no_grad():
            zs = [z.clone() for z in model.encode(x)]
            source_cond = model.make_cond(zs, y_idx)
            x_t = module.encode_stochastic(x, source_cond, T=T)
            recon0 = module.render(x_t, source_cond, T=T)
            base_probs = classifier_probs(classifier, rendered_to_classifier_input(recon0))
            for attr in attributes:
                target_cond_col = cond_attrs.index(attr)
                for direction_sign in directions:
                    y_cf = y_idx.clone()
                    y_cf[:, target_cond_col] = 1 if direction_sign == "positive" else 0
                    cf = module.render(x_t, model.make_cond(zs, y_cf), T=T)
                    edit_probs = classifier_probs(classifier, rendered_to_classifier_input(cf))
                    key = (attr, direction_sign)
                    accum[key]["base"].append(base_probs)
                    accum[key]["edit"].append(edit_probs)
                    if args.save_grids and key not in first_grid:
                        first_grid[key] = (x.add(1).div(2).detach().cpu(), recon0.clamp(0, 1).detach().cpu(), cf.clamp(0, 1).detach().cpu())
        seen += len(x)
        logging.info("processed %d/%d images", min(seen, args.num_images), args.num_images)

    long_rows = []
    for (attr, direction_sign), value in sorted(accum.items()):
        if not value["base"]:
            continue
        base = np.concatenate(value["base"], axis=0)
        edit = np.concatenate(value["edit"], axis=0)
        target_idx = attr_to_idx[attr]
        summary = summarize_attribute_changes(base, edit, target_idx, preservation_indices=preserve_idx)
        summary["target_intended_flip_rate"] = intended_target_flip_rate(base, edit, target_idx, direction_sign)
        row = {"attribute": attr, "direction": direction_sign,
               "edit_mechanism": "conditioning_signal_only_fixed_latents",
               "num_conditioning_attributes": len(cond_attrs),
               "num_preservation_attributes": len(preserve_idx), **summary}
        long_rows.append(row)

    long_path = out / "preservation_sweep.csv"
    with open(long_path, "w", newline="") as f:
        fieldnames = list(long_rows[0].keys()) if long_rows else ["attribute", "direction"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(long_rows)

    for row in long_rows:
        suffix = "_pos" if row["direction"] == "positive" else "_neg"
        _write_single_row_csv(out / f"{safe_name(row['attribute'])}{suffix}_conditioning_preservation.csv", row)

    if args.save_grids:
        for (attr, direction_sign), (orig, recon0, edit) in first_grid.items():
            save_labeled_grid([orig, recon0, edit], ["original", "recon0", f"condition_{attr}_{direction_sign}"],
                              out / f"{safe_name(attr)}_{direction_sign}_grid.png")

    summary = {"config": args.config, "ckpt": args.ckpt, "attributes": attributes,
               "attribute_notes": {"Young": "age attribute"}, "directions": directions,
               "edit_mechanism": "conditioning_signal_only_fixed_latents",
               "conditioning_attrs": cond_attrs,
               "preservation_attrs": [attr_names[i] for i in preserve_idx],
               "num_images": int(seen), "csv": str(long_path)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("wrote conditioning preservation eval to %s", long_path)


if __name__ == "__main__":
    main()
