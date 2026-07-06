#!/usr/bin/env python
"""Sweep latent-edit level and strength while measuring preservation drift."""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.hdae.counterfactuals.directions import (
    choose_probe_row,
    direction_from_probe_checkpoint,
    probe_weight_path,
    summarize_attribute_changes,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
DEFAULT_ATTRIBUTES = ["Smiling", "Eyeglasses", "Male", "Young"]
DEFAULT_STRENGTHS = [0.0, 0.5, 1.0, 2.0, 4.0]


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_strengths(value):
    strengths = [float(item.strip()) for item in value.split(",") if item.strip()]
    if 0.0 not in strengths:
        logging.warning("strength list did not include 0; injecting strength-0 drift-control column")
        strengths = [0.0, *strengths]
    return strengths


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def classifier_probs_gpu(classifier, x):
    """Returns raw GPU tensor; avoids blocking host PCIe sync."""
    return torch.sigmoid(classifier(x))


def rendered_to_classifier_input(x01):
    return x01.mul(2).sub(1).clamp(-1, 1)


def intended_target_flip_rate(before, after, target_index, direction_sign):
    before_pos = before[:, target_index] >= 0.5
    after_pos = after[:, target_index] >= 0.5
    if direction_sign == "positive":
        return float((~before_pos & after_pos).mean())
    return float((before_pos & ~after_pos).mean())


def plot_preservation_heatmap(path, levels, strengths, matrix, title=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(5, len(strengths) * 0.9), max(3, len(levels) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(strengths)))
    ax.set_xticklabels([f"{s:g}" for s in strengths])
    ax.set_yticks(np.arange(len(levels)))
    ax.set_yticklabels([f"Z{level + 1}" for level in levels])
    ax.set_xlabel("CF strength")
    ax.set_ylabel("Edited latent level")
    ax.set_title(title or "Non-target absolute delta")
    fig.colorbar(im, ax=ax, label="mean |Δ non-target| base=recon0")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_matrix_csv(path, levels, strengths, values_by_cell):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", *[f"strength_{s:g}" for s in strengths]])
        for level in levels:
            writer.writerow([level, *[values_by_cell.get((level, float(s)), float("nan")) for s in strengths]])


def _load_level_directions(attributes, num_levels, probe_metrics, probe_weights_dir):
    directions = {}
    for attr in attributes:
        for level in range(num_levels):
            try:
                row = choose_probe_row(probe_metrics, attr, level=level)
                weight_path = probe_weight_path(probe_weights_dir, row)
                direction, _state = direction_from_probe_checkpoint(weight_path)
                directions[(attr, level)] = (row, direction)
            except Exception as exc:
                logging.warning("skipping attribute=%s level=%d: %s", attr, level, exc)
    return directions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--probe-metrics", required=True)
    p.add_argument("--probe-weights-dir", required=True)
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--attributes", default=",".join(DEFAULT_ATTRIBUTES), help="Comma-separated target attributes")
    p.add_argument("--strengths", default=",".join(f"{s:g}" for s in DEFAULT_STRENGTHS))
    p.add_argument("--direction", choices=["positive", "negative", "both"], default="both")
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4, help="Subprocess workers for DataLoader")
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--normalize-strength", dest="normalize_strength", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--save-grids", action="store_true")
    p.add_argument("--per-attribute-matrix", action="store_true")
    args = p.parse_args()

    from torch.utils.data import DataLoader
    from experiments.hdae.counterfactuals.attribute_classifier import load_classifier
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.grid_utils import save_labeled_grid
    from experiments.hdae.hdae.lit_module import HDAELitModule

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    attributes = parse_csv_list(args.attributes)
    strengths = parse_strengths(args.strengths)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    model = module.ema_model
    param_dtype = next(model.parameters()).dtype

    classifier, clf_state = load_classifier(args.attr_classifier, device=device)
    attr_names = [str(x) for x in clf_state["attribute_names"]]
    attr_to_idx = {name: i for i, name in enumerate(attr_names)}

    if missing_attrs := [attr for attr in attributes if attr not in attr_to_idx]:
        raise ValueError(f"Attributes not found in classifier: {missing_attrs}")

    num_levels = len(model.hdae_conf.encoder.level_dims)
    levels = list(range(num_levels))
    directions = ["positive", "negative"] if args.direction == "both" else [args.direction]
    T = args.T or cfg.raw["train"]["T_eval"]
    raw_dirs = _load_level_directions(attributes, num_levels, args.probe_metrics, args.probe_weights_dir)

    gpu_level_dirs = {
        (attr, lvl): (row, torch.as_tensor(vec, dtype=param_dtype, device=device)[None, :])
        for (attr, lvl), (row, vec) in raw_dirs.items()
    }

    all_base_probs = []
    accum = {
        (attr, level, float(strength), direction): []
        for attr in attributes for level in levels for strength in strengths for direction in directions
        if (attr, level) in gpu_level_dirs
    }

    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    master_grids = {}
    seen = 0

    with torch.inference_mode():
        for batch in loader:
            if seen >= args.num_images:
                break

            x = batch["img"][:args.num_images - seen].to(device, non_blocking=True)
            encoded = model.encode(x)
            zs = encoded["zs"]

            x_t = module.encode_stochastic(x, encoded["cond"], T=T)
            recon0 = module.render(x_t, model.merge(zs), T=T)

            base_probs = classifier_probs_gpu(classifier, rendered_to_classifier_input(recon0))
            all_base_probs.append(base_probs)

            level_scales = [z.norm(dim=1).mean() for z in zs]

            for attr in attributes:
                for level in levels:
                    if (attr, level) not in gpu_level_dirs:
                        continue

                    _row, d = gpu_level_dirs[(attr, level)]

                    for direction_sign in directions:
                        sign = 1.0 if direction_sign == "positive" else -1.0

                        # Decoupled strictly into a 2-tuple for line 306 unpacking
                        grid_key = (attr, direction_sign)
                        if args.save_grids and grid_key not in master_grids:
                            master_grids[grid_key] = {
                                "orig": x.add(1).div(2).cpu(),
                                "recon0": recon0.clamp(0, 1).cpu(),
                                "edits": {}
                            }

                        for strength in strengths:
                            key = (attr, level, float(strength), direction_sign)

                            if strength == 0.0:
                                accum[key].append(base_probs)
                                continue

                            s_eff = strength * (level_scales[level] if args.normalize_strength else 1.0)
                            zs_edit = list(zs)
                            zs_edit[level] = zs[level] + sign * s_eff * d

                            cf = module.render(x_t, model.merge(zs_edit), T=T)
                            edit_probs = classifier_probs_gpu(classifier, rendered_to_classifier_input(cf))
                            accum[key].append(edit_probs)

                            if args.save_grids:
                                edit_sub_key = (level, float(strength))
                                if edit_sub_key not in master_grids[grid_key]["edits"]:
                                    master_grids[grid_key]["edits"][edit_sub_key] = cf.clamp(0, 1).cpu()

            seen += len(x)
            logging.info("processed %d/%d images", min(seen, args.num_images), args.num_images)

    logging.info("Draining CUDA queues and compiling metrics...")
    stacked_base = torch.cat(all_base_probs, dim=0).cpu().numpy()

    long_rows = []
    for key in sorted(accum.keys(), key=lambda kv: (kv[0], kv[3], kv[1], kv[2])):
        if not accum[key]:
            continue

        stacked_edit = torch.cat(accum[key], dim=0).cpu().numpy()
        attr, level, strength, direction_sign = key
        target_idx = attr_to_idx[attr]

        summary = summarize_attribute_changes(stacked_base, stacked_edit, target_idx)
        summary["target_intended_flip_rate"] = intended_target_flip_rate(
            stacked_base, stacked_edit, target_idx, direction_sign
        )

        row = {
            "attribute": attr,
            "level": int(level),
            "level_dim": int(model.hdae_conf.encoder.level_dims[level]),
            "strength": float(strength),
            "direction": direction_sign,
            **summary,
        }
        long_rows.append(row)

    long_path = out / "preservation_sweep.csv"
    with open(long_path, "w", newline="") as f:
        fieldnames = list(long_rows[0].keys()) if long_rows else ["attribute", "level", "strength", "direction"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)

    for attr in attributes:
        for direction_sign in directions:
            suffix = "" if args.direction != "both" else ("_pos" if direction_sign == "positive" else "_neg")
            rows = [r for r in long_rows if r["attribute"] == attr and r["direction"] == direction_sign]
            by = {(int(r["level"]), float(r["strength"])): r for r in rows}
            attr_safe = safe_name(attr)

            _write_matrix_csv(out / f"{attr_safe}{suffix}_target_delta.csv", levels, strengths,
                              {(level, s): by[(level, s)]["target_delta_mean"] for level, s in by})
            _write_matrix_csv(out / f"{attr_safe}{suffix}_nontarget_abs_delta.csv", levels, strengths,
                              {(level, s): by[(level, s)]["non_target_abs_delta_mean"] for level, s in by})
            _write_matrix_csv(out / f"{attr_safe}{suffix}_nontarget_flip_frac.csv", levels, strengths,
                              {(level, s): by[(level, s)]["non_target_flip_fraction"] for level, s in by})

            if args.per_attribute_matrix:
                matrix = np.full((len(levels), len(strengths)), np.nan, dtype=float)
                for i, level in enumerate(levels):
                    for j, strength in enumerate(strengths):
                        if (level, float(strength)) in by:
                            matrix[i, j] = by[(level, float(strength))]["non_target_abs_delta_mean"]
                plot_preservation_heatmap(out / f"{attr_safe}{suffix}_preservation_heatmap.png", levels, strengths,
                                          matrix,
                                          title=f"{attr} {direction_sign} preservation")

    if args.save_grids:
        for (attr, direction_sign), data in master_grids.items():
            grid_images = [data["orig"], data["recon0"]]
            grid_labels = ["original", "recon0"]

            for lvl, s in sorted(data["edits"].keys()):
                grid_images.append(data["edits"][(lvl, s)])
                grid_labels.append(f"Z{lvl + 1}_s{s:g}")

            save_labeled_grid(
                grid_images,
                grid_labels,
                out / f"{safe_name(attr)}_{direction_sign}_master_sweep.png"
            )

    summary_data = {
        "config": args.config,
        "ckpt": args.ckpt,
        "attributes": attributes,
        "attribute_notes": {"Young": "age attribute"},
        "strengths": strengths,
        "directions": directions,
        "normalize_strength": bool(args.normalize_strength),
        "level_dims": {str(i): int(dim) for i, dim in enumerate(model.hdae_conf.encoder.level_dims)},
        "num_images": int(seen),
        "csv": str(long_path)
    }
    (out / "summary.json").write_text(json.dumps(summary_data, indent=2))
    logging.info("wrote preservation sweep to %s", long_path)


if __name__ == "__main__":
    main()