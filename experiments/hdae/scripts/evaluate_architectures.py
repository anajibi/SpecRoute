#!/usr/bin/env python
"""Evaluate HDAE architectures on hierarchy usage and edit preservation.

No training is performed. The script consumes trained HDAE checkpoints, existing
linear-probe directions, and an attribute classifier, then writes one JSON file
with architecture-level metrics:

* Axis 1: per-level causal influence from nulling levels, plus effective number
  of used levels (entropy participation ratio).
* Axis 3: editing frontier reduced to preservation-at-fixed target efficacy.
"""
import argparse, csv, json, logging, math, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.hdae.counterfactuals.directions import (
    choose_probe_row,
    direction_from_probe_checkpoint,
    probe_weight_path,
    summarize_attribute_changes,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def parse_named_paths(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        if not name:
            raise ValueError(f"empty NAME in {item!r}")
        out[name] = Path(path)
    return out


def parse_csv_list(value, *, cast=str):
    if value is None or value == "":
        return []
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def validate_same_names(**groups):
    expected = None
    for group_name, mapping in groups.items():
        names = set(mapping)
        if expected is None:
            expected = names
            continue
        if names != expected:
            raise ValueError(f"{group_name} names {sorted(names)} do not match expected {sorted(expected)}")
    return sorted(expected or [])


def rendered_to_classifier_input(x01):
    return x01.mul(2).sub(1).clamp(-1, 1)


def classifier_probs(classifier, x):
    import torch
    with torch.no_grad():
        return torch.sigmoid(classifier(x)).detach().cpu().numpy()


def build_perceptual_distance(device):
    try:
        import lpips
        net = lpips.LPIPS(net="alex").to(device).eval()
        return net, "lpips"
    except Exception as exc:
        logging.warning("LPIPS unavailable; using RMSE fallback for level influence: %s", exc)
        return None, "rmse_fallback"


def perceptual_distance(x, y, lpips_net=None):
    """Return one distance per image for tensors in [-1, 1]."""
    import torch
    with torch.no_grad():
        if lpips_net is not None:
            return lpips_net(x, y).flatten()
        return (x - y).square().flatten(1).mean(1).sqrt()


def effective_num_levels(influences, eps=1e-12):
    """Entropy participation ratio for a nonnegative influence vector."""
    arr = np.asarray(influences, dtype=np.float64)
    total = float(arr.sum())
    if total <= eps:
        return {"normalized_influence": [0.0 for _ in arr], "entropy": 0.0,
                "n_eff": 0.0, "n_eff_norm": 0.0}
    p = arr / total
    entropy = -float(np.sum(np.where(p > 0, p * np.log(p), 0.0)))
    n_eff = float(np.exp(entropy))
    return {"normalized_influence": p.tolist(), "entropy": entropy,
            "n_eff": n_eff, "n_eff_norm": n_eff / len(arr)}


def intended_target_flip_rate(before, after, target_index, direction_sign):
    before_pos = before[:, target_index] >= 0.5
    after_pos = after[:, target_index] >= 0.5
    if direction_sign == "positive":
        return float((~before_pos & after_pos).mean())
    return float((before_pos & ~after_pos).mean())


def weighted_average_summaries(items):
    """Average batch summaries with image-count weights."""
    if not items:
        return {}
    total = sum(weight for _summary, weight in items)
    keys = sorted({key for summary, _weight in items for key in summary})
    out = {}
    for key in keys:
        vals = [(summary[key], weight) for summary, weight in items
                if key in summary and isinstance(summary[key], (int, float))]
        if vals:
            out[key] = float(sum(float(value) * weight for value, weight in vals) / total)
    return out


def preservation_at_efficacy(rows, threshold):
    """Return first point whose target flip-rate reaches the threshold."""
    if not rows:
        return {"reached_threshold": False, "selected_strength": None,
                "preservation_at_efficacy": None, "max_target_flip_rate": 0.0}
    ordered = sorted(rows, key=lambda r: float(r["strength"]))
    max_target = max(float(r.get("target_intended_flip_rate", 0.0)) for r in ordered)
    for row in ordered:
        if float(row.get("target_intended_flip_rate", 0.0)) >= threshold:
            return {"reached_threshold": True,
                    "selected_strength": float(row["strength"]),
                    "preservation_at_efficacy": float(row["non_target_abs_delta_mean"]),
                    "target_flip_rate": float(row.get("target_intended_flip_rate", 0.0)),
                    "target_delta_abs_mean": float(row.get("target_delta_abs_mean", 0.0)),
                    "non_target_flip_fraction": float(row.get("non_target_flip_fraction", 0.0)),
                    "non_target_severe_fraction": float(row.get("non_target_severe_fraction", 0.0))}
    best_available = min(ordered, key=lambda r: float(r.get("non_target_abs_delta_mean", float("inf"))))
    return {"reached_threshold": False,
            "selected_strength": None,
            "preservation_at_efficacy": None,
            "max_target_flip_rate": float(max_target),
            "best_available_non_target_abs_delta_mean": float(best_available.get("non_target_abs_delta_mean", float("nan")))}


def reduce_frontier_by_level(frontier_rows, efficacy_threshold):
    grouped = defaultdict(list)
    for row in frontier_rows:
        grouped[(row["model"], row["attribute"], row["direction"], int(row["level"]))].append(row)
    out = []
    for (model, attr, direction, level), rows in sorted(grouped.items()):
        result = preservation_at_efficacy(rows, efficacy_threshold)
        first = rows[0]
        out.append({"model": model, "attribute": attr, "direction": direction,
                    "level": level, "resolution": first.get("resolution"), "dim": first.get("dim"),
                    **result})
    return out


def pick_best_level_per_attribute(level_rows):
    grouped = defaultdict(list)
    for row in level_rows:
        grouped[(row["model"], row["attribute"], row["direction"])].append(row)
    best = []
    for (model, attr, direction), rows in sorted(grouped.items()):
        reached = [r for r in rows if r.get("reached_threshold")]
        if reached:
            chosen = min(reached, key=lambda r: float(r["preservation_at_efficacy"]))
        else:
            chosen = max(rows, key=lambda r: float(r.get("max_target_flip_rate", 0.0)))
        rec = dict(chosen)
        rec["best_level"] = rec.pop("level")
        rec["best_resolution"] = rec.pop("resolution")
        rec["best_dim"] = rec.pop("dim")
        best.append(rec)
    return best


def summarize_model_editing(best_rows, efficacy_threshold):
    reached = [r for r in best_rows if r.get("reached_threshold")]
    values = [float(r["preservation_at_efficacy"]) for r in reached]
    return {"efficacy_threshold": float(efficacy_threshold),
            "mean_preservation_at_efficacy": float(np.mean(values)) if values else None,
            "median_preservation_at_efficacy": float(np.median(values)) if values else None,
            "coverage": float(len(reached) / len(best_rows)) if best_rows else 0.0,
            "num_targets": int(len(best_rows)), "num_reached": int(len(reached))}


def _load_level_directions(attributes, levels, probe_metrics, probe_weights_dir):
    directions = {}
    for attr in attributes:
        for level in levels:
            try:
                row = choose_probe_row(probe_metrics, attr, level=level)
                direction, _state = direction_from_probe_checkpoint(probe_weight_path(probe_weights_dir, row))
                directions[(attr, level)] = direction
            except Exception as exc:
                logging.warning("missing edit direction attr=%s level=%s from %s: %s", attr, level, probe_metrics, exc)
    return directions


def compute_level_influence(module, batch, T, lpips_net):
    import torch
    device = next(module.parameters()).device
    x = batch["img"].to(device)
    model = module.ema_model
    with torch.no_grad():
        encoded = model.encode(x)
        zs = [z.clone() for z in encoded["zs"]]
        x_t = module.encode_stochastic(x, encoded["cond"], T=T)
        recon0 = module.render(x_t, encoded["cond"], T=T)
        recon_m11 = rendered_to_classifier_input(recon0)
        values = []
        for level in range(len(zs)):
            cond_null = model.merge(zs, null_levels=[level])
            null_img = module.render(x_t, cond_null, T=T)
            null_m11 = rendered_to_classifier_input(null_img)
            values.append(float(perceptual_distance(recon_m11, null_m11, lpips_net).mean().detach().cpu()))
    return values


def compute_edit_frontier_for_model(module, classifier, attr_names, loader, *, model_name,
                                    probe_metrics, probe_weights_dir, attributes, strengths,
                                    direction_signs, num_images, T, normalize_strength=True):
    import torch
    model = module.ema_model
    device = next(module.parameters()).device
    levels = list(range(len(model.hdae_conf.encoder.level_dims)))
    level_dims = [int(x) for x in model.hdae_conf.encoder.level_dims]
    tap_resolutions = [int(x) for x in model.hdae_conf.encoder.tap_resolutions]
    attr_to_idx = {name: i for i, name in enumerate(attr_names)}
    directions = _load_level_directions(attributes, levels, probe_metrics, probe_weights_dir)
    accum = defaultdict(list)
    seen = 0
    for batch in loader:
        if seen >= num_images:
            break
        x = batch["img"][:num_images - seen].to(device)
        with torch.no_grad():
            encoded = model.encode(x)
            zs = [z.clone() for z in encoded["zs"]]
            x_t = module.encode_stochastic(x, encoded["cond"], T=T)
            recon0 = module.render(x_t, encoded["cond"], T=T)
            base_probs = classifier_probs(classifier, rendered_to_classifier_input(recon0))
            level_scales = [float(z.norm(dim=1, keepdim=True).mean().detach().cpu()) for z in zs]
            for attr in attributes:
                if attr not in attr_to_idx:
                    raise ValueError(f"attribute {attr!r} not found in classifier")
                target_idx = attr_to_idx[attr]
                for level in levels:
                    if (attr, level) not in directions:
                        continue
                    d = torch.as_tensor(directions[(attr, level)], dtype=zs[level].dtype, device=zs[level].device)[None, :]
                    for direction_sign in direction_signs:
                        sign = 1.0 if direction_sign == "positive" else -1.0
                        for strength in strengths:
                            s_eff = float(strength) * (level_scales[level] if normalize_strength else 1.0)
                            zs_edit = [z.clone() for z in zs]
                            zs_edit[level] = zs_edit[level] + sign * s_eff * d
                            edit_img = module.render(x_t, model.merge(zs_edit), T=T)
                            edit_probs = classifier_probs(classifier, rendered_to_classifier_input(edit_img))
                            summary = summarize_attribute_changes(base_probs, edit_probs, target_idx)
                            summary["target_intended_flip_rate"] = intended_target_flip_rate(
                                base_probs, edit_probs, target_idx, direction_sign)
                            key = (attr, direction_sign, level, float(strength))
                            accum[key].append((summary, len(x)))
        seen += len(x)
        logging.info("%s edit frontier processed %d/%d images", model_name, min(seen, num_images), num_images)
    rows = []
    for (attr, direction_sign, level, strength), items in sorted(accum.items()):
        summary = weighted_average_summaries(items)
        rows.append({"model": model_name, "attribute": attr, "direction": direction_sign,
                     "level": int(level), "resolution": tap_resolutions[level], "dim": level_dims[level],
                     "strength": float(strength), **summary})
    return rows


def _rank_by_hierarchy(models):
    return sorted(models, key=lambda name: models[name]["hierarchy_usage"]["n_eff_norm"], reverse=True)


def _rank_by_editing(models):
    def key(name):
        q = models[name]["editing_quality"]
        score = q["mean_preservation_at_efficacy"]
        return (-q["coverage"], float("inf") if score is None else score)
    return sorted(models, key=key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True, help="NAME=config.yaml entries")
    p.add_argument("--ckpts", nargs="+", required=True, help="NAME=checkpoint.ckpt entries")
    p.add_argument("--probe-metrics", nargs="+", required=True, help="NAME=probe_metrics.csv entries")
    p.add_argument("--probe-weights", nargs="+", required=True, help="NAME=probe weights dir entries")
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--attributes", default="Smiling,Eyeglasses,Male,Young")
    p.add_argument("--strengths", default="0,0.5,1,2,4")
    p.add_argument("--directions", default="positive,negative")
    p.add_argument("--efficacy-threshold", type=float, default=0.8)
    p.add_argument("--num-images", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--normalize-strength", dest="normalize_strength", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from experiments.hdae.counterfactuals.attribute_classifier import load_classifier
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.lit_module import HDAELitModule

    configs = parse_named_paths(args.configs)
    ckpts = parse_named_paths(args.ckpts)
    probe_metrics = parse_named_paths(args.probe_metrics)
    probe_weights = parse_named_paths(args.probe_weights)
    names = validate_same_names(configs=configs, ckpts=ckpts, probe_metrics=probe_metrics, probe_weights=probe_weights)
    attributes = parse_csv_list(args.attributes)
    strengths = parse_csv_list(args.strengths, cast=float)
    direction_signs = parse_csv_list(args.directions)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier, clf_state = load_classifier(args.attr_classifier, device=device)
    attr_names = [str(x) for x in clf_state["attribute_names"]]
    lpips_net, distance_name = build_perceptual_distance(device)

    output = {"settings": {"num_images": args.num_images, "batch_size": args.batch_size, "T": args.T,
                             "attributes": attributes, "strengths": strengths, "directions": direction_signs,
                             "efficacy_threshold": args.efficacy_threshold,
                             "normalize_strength": bool(args.normalize_strength),
                             "level_influence_distance": distance_name},
              "models": {}, "per_level": [], "editing_frontiers": [],
              "best_editing_by_attribute": [], "ranking": {}}

    for name in names:
        logging.info("evaluating architecture %s", name)
        cfg = load_hdae_config(str(configs[name]))
        raw = cfg.raw
        T = args.T or raw["train"]["T_eval"]
        module = HDAELitModule.load_from_checkpoint(str(ckpts[name]), conf=cfg.train_conf, map_location="cpu").to(device).eval()
        ds = CelebAHQPacked(raw["data"]["lmdb_path"], raw["data"]["attr_npz"], flip=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        influence_sums = None
        influence_count = 0
        influence_seen = 0
        for batch in loader:
            if influence_seen >= args.num_images:
                break
            sub = {k: (v[:args.num_images - influence_seen] if hasattr(v, "__getitem__") else v) for k, v in batch.items()}
            influences = np.asarray(compute_level_influence(module, sub, T, lpips_net), dtype=np.float64)
            if influence_sums is None:
                influence_sums = np.zeros_like(influences)
            n = len(sub["img"])
            influence_sums += influences * n
            influence_count += n
            influence_seen += n
        level_influence = (influence_sums / max(influence_count, 1)).tolist()
        eff = effective_num_levels(level_influence)
        level_dims = [int(x) for x in raw["encoder"]["level_dims"]]
        tap_resolutions = [int(x) for x in raw["encoder"].get("tap_resolutions", list(range(len(level_dims))))]
        frontier_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        frontier_rows = compute_edit_frontier_for_model(
            module, classifier, attr_names, frontier_loader, model_name=name,
            probe_metrics=str(probe_metrics[name]), probe_weights_dir=str(probe_weights[name]),
            attributes=attributes, strengths=strengths, direction_signs=direction_signs,
            num_images=args.num_images, T=T, normalize_strength=args.normalize_strength)
        level_reductions = reduce_frontier_by_level(frontier_rows, args.efficacy_threshold)
        best_rows = pick_best_level_per_attribute(level_reductions)
        editing_quality = summarize_model_editing(best_rows, args.efficacy_threshold)
        output["models"][name] = {"config": str(configs[name]), "ckpt": str(ckpts[name]),
                                   "K": len(level_dims), "level_dims": level_dims,
                                   "tap_resolutions": tap_resolutions,
                                   "hierarchy_usage": {"level_influence": level_influence,
                                                       **eff, "distance": distance_name},
                                   "editing_quality": editing_quality}
        for level, influence in enumerate(level_influence):
            output["per_level"].append({"model": name, "level": level,
                                         "resolution": tap_resolutions[level], "dim": level_dims[level],
                                         "influence": influence,
                                         "normalized_influence": eff["normalized_influence"][level]})
        output["editing_frontiers"].extend(frontier_rows)
        output["best_editing_by_attribute"].extend(best_rows)

    output["ranking"] = {"by_hierarchy_usage": _rank_by_hierarchy(output["models"]),
                          "by_editing_quality": _rank_by_editing(output["models"])}
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logging.info("wrote architecture evaluation to %s", out_path)


if __name__ == "__main__":
    main()
