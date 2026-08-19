#!/usr/bin/env python
"""T5: per-attribute conditioning strength, CPU-only, no diffusion sampling.

Corrected per advisor review: PerBlockStyleFiLM computes style = z_style*(1+scale(attr_emb)) +
shift(attr_emb) -- scale/shift depend only on attr_emb, but the final STYLE also depends on
z_style (i.e. zs), multiplicatively. Measuring swing in attr_emb alone (as the original plan text
said) doesn't reflect what actually reaches the decoder. This holds a batch of REAL encoded zs
fixed and measures the swing in the resulting per-block STYLE vectors instead, via
HierarchicalAutoencModel._styles(zs, y_idx) -- the exact function used at both train and eval time.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from experiments.hdae.counterfactuals import hdae_adapter  # noqa: F401 (registers hdae adapter)
from experiments.hdae.hdae.attr_utils import to_cond_values
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

PACKED = "experiments/hdae/data/packed/morphomnist_70k.h5"
N_CONTEXT = 64  # real images providing zs
N_BOOTSTRAP = 200
OUT_PATH = "experiments/hdae/outputs/diagnostics_t5_swing.json"

MODELS = [
    ("k1", "experiments/hdae/configs/morpho_hier_k1_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k1_v3/checkpoints/last.ckpt"),
    ("k5", "experiments/hdae/configs/morpho_hier_k5_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k5_v3/checkpoints/last.ckpt"),
    ("k11_75k", "experiments/hdae/configs/morpho_hier_k11_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last_step75000.ckpt"),
]
ATTRS = ["digit", "thickness", "intensity", "hue"]


def load_model(config_path, ckpt_path, device="cpu"):
    cfg = load_hdae_config(config_path)
    from experiments.hdae.hdae.lit_module import HDAELitModule
    module = HDAELitModule(cfg.train_conf)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    module.load_state_dict(ckpt["state_dict"], strict=True)
    module.eval()
    return module.ema_model.to(device), cfg


def main():
    device = "cpu"
    ds = MorphoMNISTPacked(PACKED)
    names = ds.attribute_names
    val_mask = ds.partitions == 1
    val_idx = np.where(val_mask)[0]
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(val_idx, size=N_CONTEXT, replace=False)
    imgs = torch.stack([ds[int(i)]["img"] for i in sample_idx]).to(device)

    train_mask = ds.partitions == 0
    train_attrs = ds.attrs[train_mask]
    percentiles = {}
    for a in ATTRS:
        col = train_attrs[:, names.index(a)]
        percentiles[a] = {"p5": float(np.percentile(col, 5)), "p50": float(np.percentile(col, 50)),
                          "p95": float(np.percentile(col, 95))}

    results = {}
    for model_name, config_path, ckpt_path in MODELS:
        print(f"=== {model_name} ===")
        model, cfg = load_model(config_path, ckpt_path, device)
        cond_specs = model.hdae_conf.encoder.cond_specs
        modeled_attrs = list(model.hdae_conf.encoder.conditioning_attrs)

        with torch.no_grad():
            zs = model.encode(imgs)  # list of per-level latents, real context

        base_raw = torch.tensor([[percentiles[a]["p50"] for a in modeled_attrs]] * N_CONTEXT,
                                dtype=torch.float32, device=device)

        model_results = {}
        for attr in modeled_attrs:
            i = modeled_attrs.index(attr)
            raw_lo = base_raw.clone()
            raw_hi = base_raw.clone()
            raw_lo[:, i] = percentiles[attr]["p5"]
            raw_hi[:, i] = percentiles[attr]["p95"]
            y_lo = to_cond_values(raw_lo, cond_specs).to(device)
            y_hi = to_cond_values(raw_hi, cond_specs).to(device)

            with torch.no_grad():
                styles_lo = model._styles(zs, y_lo)
                styles_hi = model._styles(zs, y_hi)

            per_block_swing = []
            per_block_style_norm = []
            for s_lo, s_hi in zip(styles_lo, styles_hi):
                diff = (s_hi - s_lo).float()
                swing_per_example = diff.norm(dim=-1)  # [N_CONTEXT]
                style_norm_per_example = s_lo.float().norm(dim=-1)
                per_block_swing.append(swing_per_example.numpy())
                per_block_style_norm.append(style_norm_per_example.numpy())
            per_block_swing = np.stack(per_block_swing, axis=0)   # [n_blocks, N_CONTEXT]
            per_block_style_norm = np.stack(per_block_style_norm, axis=0)

            total_swing_per_example = per_block_swing.sum(axis=0)  # sum over blocks, per context image
            normalized_swing = per_block_swing / (per_block_style_norm + 1e-8)

            boot_means = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.randint(0, N_CONTEXT, size=N_CONTEXT)
                boot_means.append(total_swing_per_example[idx].mean())
            boot_means = np.array(boot_means)

            model_results[attr] = {
                "mean_total_swing": float(total_swing_per_example.mean()),
                "swing_ci95": [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))],
                "mean_normalized_swing_per_block": [float(x) for x in normalized_swing.mean(axis=1)],
                "mean_normalized_swing_overall": float(normalized_swing.mean()),
            }
            print(f"  {attr:10s} mean_total_swing={model_results[attr]['mean_total_swing']:.4f} "
                 f"CI={model_results[attr]['swing_ci95']} norm_swing={model_results[attr]['mean_normalized_swing_overall']:.4f}")
        results[model_name] = model_results
        del model
    Path(OUT_PATH).write_text(json.dumps({"percentiles": percentiles, "results": results}, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
