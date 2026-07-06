#!/usr/bin/env python
"""Run counterfactual/factual consistency metrics from cached HDAE encodings.

Model specs are NAME=CONFIG,CKPT,PROBE_METRICS,PROBE_WEIGHTS_DIR. The script is
HDAE-native; external DiffAE baselines should be adapted to the same cache/edit
interface before being passed here.
"""
import argparse, csv, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.hdae.counterfactuals.directions import (
    choose_probe_row,
    direction_from_probe_checkpoint,
    probe_weight_path,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def parse_csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_models(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=config,ckpt,probe_metrics,weights, got {item!r}")
        name, spec = item.split("=", 1)
        parts = [Path(x) for x in spec.split(",")]
        if len(parts) != 4:
            raise ValueError(f"model {name!r} needs 4 comma-separated paths")
        out[name] = {"config": parts[0], "ckpt": parts[1], "probe_metrics": parts[2], "probe_weights": parts[3]}
    return out


def source_indices(cohorts, attr, direction):
    side = "neg_idx" if direction == "positive" else "pos_idx"
    return [int(x) for x in cohorts["attributes"][attr][side]]


def batched(seq, n):
    for start in range(0, len(seq), n):
        yield seq[start:start + n]


def rendered_to_classifier_input(x01):
    return x01.mul(2).sub(1).clamp(-1, 1)


def classifier_probs(classifier, x):
    import torch
    with torch.inference_mode():
        return torch.sigmoid(classifier(x)).detach().cpu().numpy()


def non_target_flip_fraction(base_probs, edit_probs, target_idx, mask):
    if mask.sum() == 0:
        return float("nan")
    non = [i for i in range(base_probs.shape[1]) if i != target_idx]
    flips = (base_probs[mask][:, non] >= 0.5) != (edit_probs[mask][:, non] >= 0.5)
    return float(flips.mean())


def compute_consistency(base_probs, edit_probs, target_idx, direction):
    if direction == "positive":
        source_mask = base_probs[:, target_idx] < 0.5
        success = source_mask & (edit_probs[:, target_idx] >= 0.5)
    else:
        source_mask = base_probs[:, target_idx] >= 0.5
        success = source_mask & (edit_probs[:, target_idx] < 0.5)
    fail = source_mask & ~success
    n_source = int(source_mask.sum())
    n_success = int(success.sum())
    n_fail = int(fail.sum())
    return {"counterfactual_consistency": float(n_success / n_source) if n_source else float("nan"),
            "factual_flip_success": non_target_flip_fraction(base_probs, edit_probs, target_idx, success),
            "factual_flip_fail": non_target_flip_fraction(base_probs, edit_probs, target_idx, fail),
            "n_source": n_source, "n_success": n_success, "n_fail": n_fail}


def load_directions(probe_metrics, probe_weights, attributes, levels):
    dirs = {}
    for attr in attributes:
        for level in levels:
            try:
                row = choose_probe_row(str(probe_metrics), attr, level=level)
                direction, _state = direction_from_probe_checkpoint(probe_weight_path(str(probe_weights), row))
                dirs[(attr, level)] = direction
            except Exception as exc:
                logging.warning("skip direction attr=%s level=%s: %s", attr, level, exc)
    return dirs


def cache_path(cache_dir, model_name, index):
    return Path(cache_dir) / model_name / f"{int(index):08d}.pt"


def ensure_cached(module, dataset, indices, cache_dir, model_name, T, batch_size, device):
    import torch
    model = module.ema_model
    missing = [idx for idx in indices if not cache_path(cache_dir, model_name, idx).exists()]
    if not missing:
        return
    (Path(cache_dir) / model_name).mkdir(parents=True, exist_ok=True)
    for ids in batched(missing, batch_size):
        imgs = torch.stack([dataset[i]["img"] for i in ids]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            encoded = model.encode(imgs)
            zs = [z.detach().cpu().float() for z in encoded["zs"]]
            cond = model.merge(encoded["zs"])
            x_t = module.encode_stochastic(imgs, cond, T=T).detach().cpu().float()
            recon0 = module.render(x_t.to(device), cond, T=T)
        for local, idx in enumerate(ids):
            torch.save({"index": int(idx), "zs": [z[local].clone() for z in zs],
                        "x_t": x_t[local].clone()}, cache_path(cache_dir, model_name, idx))


def load_cached_batch(cache_dir, model_name, indices, device):
    import torch
    states = [torch.load(cache_path(cache_dir, model_name, idx), map_location="cpu") for idx in indices]
    num_levels = len(states[0]["zs"])
    zs = [torch.stack([state["zs"][level] for state in states]).to(device) for level in range(num_levels)]
    x_t = torch.stack([state["x_t"] for state in states]).to(device)
    return zs, x_t


def score_recon0(module, classifier, zs, x_t, T, device):
    import torch
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        cond = module.ema_model.merge(zs)
        recon = module.render(x_t, cond, T=T)
    return classifier_probs(classifier, rendered_to_classifier_input(recon))


def run_model(model_name, spec, cohorts, attributes, directions, strength, T_eval, batch_size, cache_dir, attr_classifier, output_rows):
    import torch
    from experiments.hdae.counterfactuals.attribute_classifier import load_classifier
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.lit_module import HDAELitModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_hdae_config(str(spec["config"]))
    data = cfg.raw["data"]
    T = T_eval or cfg.raw["train"]["T_eval"]
    module = HDAELitModule.load_from_checkpoint(str(spec["ckpt"]), conf=cfg.train_conf, map_location="cpu").to(device).eval()
    dataset = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    classifier, clf_state = load_classifier(attr_classifier, device=device)
    attr_names = [str(x) for x in clf_state["attribute_names"]]
    attr_to_idx = {name: i for i, name in enumerate(attr_names)}
    levels = list(range(len(module.ema_model.hdae_conf.encoder.level_dims)))
    dirs = load_directions(spec["probe_metrics"], spec["probe_weights"], attributes, levels)

    all_indices = sorted({idx for attr in attributes for d in directions for idx in source_indices(cohorts, attr, d)})
    ensure_cached(module, dataset, all_indices, cache_dir, model_name, T, batch_size, device)

    for attr in attributes:
        target_idx = attr_to_idx[attr]
        for direction in directions:
            indices = source_indices(cohorts, attr, direction)
            for level in levels:
                if (attr, level) not in dirs:
                    continue
                base_all, edit_all = [], []
                for ids in batched(indices, batch_size):
                    zs, x_t = load_cached_batch(cache_dir, model_name, ids, device)
                    base_probs = score_recon0(module, classifier, zs, x_t, T, device)
                    import torch
                    dvec = torch.as_tensor(dirs[(attr, level)], dtype=zs[level].dtype, device=device)[None, :]
                    sign = 1.0 if direction == "positive" else -1.0
                    zs_edit = [z.clone() for z in zs]
                    zs_edit[level] = zs_edit[level] + sign * float(strength) * dvec
                    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                        cond = module.ema_model.merge(zs_edit)
                        edit = module.render(x_t, cond, T=T)
                    edit_probs = classifier_probs(classifier, rendered_to_classifier_input(edit))
                    base_all.append(base_probs); edit_all.append(edit_probs)
                base = np.concatenate(base_all, axis=0)
                edit = np.concatenate(edit_all, axis=0)
                rec = compute_consistency(base, edit, target_idx, direction)
                output_rows.append({"model": model_name, "attribute": attr, "latent_used": level,
                                    "direction": direction, **rec})


def write_rows(path, rows):
    fields = ["model", "attribute", "latent_used", "direction", "counterfactual_consistency",
              "factual_flip_success", "factual_flip_fail", "n_source", "n_success", "n_fail"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts", required=True)
    p.add_argument("--models", nargs="+", required=True,
                   help="NAME=config,ckpt,probe_metrics,probe_weights_dir entries")
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--attributes", default="Smiling,Eyeglasses,Male,Young")
    p.add_argument("--directions", default="positive,negative")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--T-eval", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cohorts = json.loads(Path(args.cohorts).read_text())
    models = parse_models(args.models)
    rows = []
    for model_name, spec in models.items():
        run_model(model_name, spec, cohorts, parse_csv_list(args.attributes), parse_csv_list(args.directions),
                  args.strength, args.T_eval, args.batch_size, args.cache_dir, args.attr_classifier, rows)
    write_rows(args.out, rows)
    Path(args.out).with_suffix(".json").write_text(json.dumps({"rows": len(rows), "strength": args.strength,
                                                                 "T_eval": args.T_eval, "cache_dir": args.cache_dir}, indent=2))


if __name__ == "__main__":
    main()
