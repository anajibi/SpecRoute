#!/usr/bin/env python
"""Run counterfactual/factual consistency metrics from cached HDAE encodings.

Model specs are NAME=CONFIG,CKPT. Legacy NAME=CONFIG,CKPT,PROBE_METRICS,PROBE_WEIGHTS_DIR entries are accepted, but probe paths are ignored because CF generation toggles conditioning only.
"""
import argparse, csv, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

import numpy as np


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def parse_csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_models(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=config,ckpt, got {item!r}")
        name, spec = item.split("=", 1)
        parts = [Path(x) for x in spec.split(",")]
        if len(parts) not in {2, 4}:
            raise ValueError(f"model {name!r} needs CONFIG,CKPT (probe paths are no longer used)")
        out[name] = {"config": parts[0], "ckpt": parts[1]}
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


def non_target_flip_fraction(base_probs, edit_probs, target_idx, mask, preservation_idx=None):
    if mask.sum() == 0:
        return float("nan")
    non = list(preservation_idx) if preservation_idx is not None else [i for i in range(base_probs.shape[1]) if i != target_idx]
    flips = (base_probs[mask][:, non] >= 0.5) != (edit_probs[mask][:, non] >= 0.5)
    return float(flips.mean())


def compute_consistency(base_probs, edit_probs, target_idx, direction, preservation_idx=None):
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
            "factual_flip_success": non_target_flip_fraction(base_probs, edit_probs, target_idx, success, preservation_idx),
            "factual_flip_fail": non_target_flip_fraction(base_probs, edit_probs, target_idx, fail, preservation_idx),
            "n_source": n_source, "n_success": n_success, "n_fail": n_fail}


def cache_path(cache_dir, model_name, index):
    return Path(cache_dir) / model_name / f"{int(index):08d}.pt"


def ensure_cached(module, dataset, indices, cache_dir, model_name, T, batch_size, device, cond_indices):
    import torch
    model = module.ema_model
    missing = [idx for idx in indices if not cache_path(cache_dir, model_name, idx).exists()]
    if not missing:
        return
    (Path(cache_dir) / model_name).mkdir(parents=True, exist_ok=True)
    for ids in batched(missing, batch_size):
        imgs = torch.stack([dataset[i]["img"] for i in ids]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            from experiments.hdae.hdae.attr_utils import to_index_space
            y_raw = torch.stack([dataset[i]["attr"][cond_indices] for i in ids]).to(device)
            y_idx = to_index_space(y_raw, model.hdae_conf.encoder.attr_input_range).to(device)
            encoded = model.encode(imgs)
            zs = [z.detach().cpu().float() for z in encoded["zs"]]
            cond = {"zs": encoded["zs"], "y_idx": y_idx}
            x_t = module.encode_stochastic(imgs, cond, T=T).detach().cpu().float()
        for local, idx in enumerate(ids):
            torch.save({"index": int(idx), "zs": [z[local].clone() for z in zs],
                        "x_t": x_t[local].clone(), "y_idx": y_idx[local].detach().cpu().clone()}, cache_path(cache_dir, model_name, idx))


def load_cached_batch(cache_dir, model_name, indices, device):
    import torch
    states = [torch.load(cache_path(cache_dir, model_name, idx), map_location="cpu") for idx in indices]
    num_levels = len(states[0]["zs"])
    zs = [torch.stack([state["zs"][level] for state in states]).to(device) for level in range(num_levels)]
    x_t = torch.stack([state["x_t"] for state in states]).to(device)
    y_idx = torch.stack([state["y_idx"] for state in states]).to(device)
    return zs, x_t, y_idx


def score_recon0(module, classifier, zs, x_t, y_idx, T, device):
    import torch
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        recon = module.render(x_t, {"zs": zs, "y_idx": y_idx}, T=T)
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
    cond_attrs = list(module.ema_model.hdae_conf.encoder.conditioning_attrs)
    cond_indices = [dataset.attribute_names.index(a) for a in cond_attrs]
    preservation_idx = [i for i, name in enumerate(attr_names) if name not in set(cond_attrs)]

    all_indices = sorted({idx for attr in attributes for d in directions for idx in source_indices(cohorts, attr, d)})
    ensure_cached(module, dataset, all_indices, cache_dir, model_name, T, batch_size, device, cond_indices)

    for attr in attributes:
        target_idx = attr_to_idx[attr]
        for direction in directions:
            indices = source_indices(cohorts, attr, direction)
            if attr not in cond_attrs:
                raise ValueError(f"attribute {attr!r} is not in conditioning_attrs={cond_attrs}")
            target_cond_col = cond_attrs.index(attr)
            base_all, edit_all = [], []
            for ids in batched(indices, batch_size):
                zs, x_t, y_idx = load_cached_batch(cache_dir, model_name, ids, device)
                base_probs = score_recon0(module, classifier, zs, x_t, y_idx, T, device)
                import torch
                y_cf = y_idx.clone()
                y_cf[:, target_cond_col] = 1 if direction == "positive" else 0
                with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                    edit = module.render(x_t, {"zs": zs, "y_idx": y_cf}, T=T)
                edit_probs = classifier_probs(classifier, rendered_to_classifier_input(edit))
                base_all.append(base_probs); edit_all.append(edit_probs)
            base = np.concatenate(base_all, axis=0)
            edit = np.concatenate(edit_all, axis=0)
            rec = compute_consistency(base, edit, target_idx, direction, preservation_idx)
            output_rows.append({"model": model_name, "attribute": attr, "latent_used": "conditioning",
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
                   help="NAME=config,ckpt entries; legacy four-field specs are accepted but probe paths are ignored")
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--attributes", default="Smiling,Eyeglasses,Male,Young")
    p.add_argument("--directions", default="positive,negative")
    p.add_argument("--strength", type=float, default=1.0, help="Ignored: conditioning-only CF has no strength.")
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
    Path(args.out).with_suffix(".json").write_text(json.dumps({"rows": len(rows), "edit_mechanism": "conditioning_signal_only_fixed_latents",
                                                                 "T_eval": args.T_eval, "cache_dir": args.cache_dir}, indent=2))


if __name__ == "__main__":
    main()
