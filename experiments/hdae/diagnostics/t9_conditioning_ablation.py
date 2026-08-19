#!/usr/bin/env python
"""T9: conditioning-ablation loss probe -- direct test of conditioning collapse / latent leakage.

Hypothesis: as training proceeds, the decoder increasingly reconstructs from the rich encoder
latent (zs) and ignores the FiLM-injected attribute conditioning, because zs is a far richer
signal. If so, the loss should become nearly IDENTICAL whether the true attributes or a null
mask are fed in -- i.e. delta_L = L(null attrs) - L(true attrs) should shrink toward 0.

Method: for each checkpoint, over a fixed grid of held-out images x timesteps, compute the
training loss twice with IDENTICAL noise/timestep/image (only the conditioning differs): once
with true attributes, once with a chosen null mask. This isolates the marginal effect of
conditioning on the loss with no other source of variance -- a paired, low-variance estimator by
construction (not because of large N).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.hdae.counterfactuals import hdae_adapter  # noqa: F401
from experiments.hdae.hdae.attr_utils import to_cond_values
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

PACKED = "experiments/hdae/data/packed/morphomnist_70k.h5"
N_IMAGES = 32
TIMESTEPS = [50, 150, 300, 450, 600, 750, 850, 950]
OUT_PATH = "experiments/hdae/outputs/diagnostics_t9_delta_l.json"

MODELS = [
    ("k1_30k", "experiments/hdae/configs/morpho_hier_k1_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k1_v3/checkpoints/last.ckpt"),
    ("k5_30k", "experiments/hdae/configs/morpho_hier_k5_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k5_v3/checkpoints/last.ckpt"),
    ("k11_66k", "experiments/hdae/configs/morpho_hier_k11_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last_step66000.ckpt"),
    ("k11_75k", "experiments/hdae/configs/morpho_hier_k11_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last_step75000.ckpt"),
]
ATTRS = ["digit", "thickness", "intensity", "hue"]


def load_module(config_path, ckpt_path, device="cpu"):
    cfg = load_hdae_config(config_path)
    module = HDAELitModule(cfg.train_conf)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    module.load_state_dict(ckpt["state_dict"], strict=True)
    module.eval()
    module.to(device)
    return module


def main():
    device = "cpu"
    ds = MorphoMNISTPacked(PACKED)
    names = ds.attribute_names
    val_mask = ds.partitions == 1
    val_idx = np.where(val_mask)[0]
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(val_idx, size=N_IMAGES, replace=False)
    imgs = torch.stack([ds[int(i)]["img"] for i in sample_idx]).to(device)
    attrs_raw = torch.stack([torch.as_tensor(ds[int(i)]["attr"]) for i in sample_idx]).to(device)

    torch.manual_seed(0)
    fixed_noise = {t: torch.randn_like(imgs) for t in TIMESTEPS}

    results = {}
    for model_name, config_path, ckpt_path in MODELS:
        print(f"=== {model_name} ===")
        module = load_module(config_path, ckpt_path, device)
        model = module.ema_model
        conditioning_attrs = list(model.hdae_conf.encoder.conditioning_attrs)
        cond_specs = model.hdae_conf.encoder.cond_specs
        col_idx = [names.index(a) for a in conditioning_attrs]

        with torch.no_grad():
            zs = model.encode(imgs)
            raw = attrs_raw[:, col_idx].float()
            y_true = to_cond_values(raw, cond_specs).to(device)

        n_attr = len(conditioning_attrs)
        masks = {"all": torch.ones(N_IMAGES, n_attr, dtype=torch.bool)}
        for i, a in enumerate(conditioning_attrs):
            m = torch.zeros(N_IMAGES, n_attr, dtype=torch.bool)
            m[:, i] = True
            masks[a] = m

        model_result = {}
        for label, mask in masks.items():
            deltas = []
            for t_val in TIMESTEPS:
                t = torch.full((N_IMAGES,), t_val, dtype=torch.long, device=device)
                noise = fixed_noise[t_val]
                with torch.no_grad():
                    cond_true = model.make_cond(zs, y_true, null_mask=None)
                    loss_true = module.sampler.training_losses(
                        model=model, x_start=imgs, t=t, model_kwargs={"cond": cond_true}, noise=noise
                    )["loss"]
                    cond_null = model.make_cond(zs, y_true, null_mask=mask)
                    loss_null = module.sampler.training_losses(
                        model=model, x_start=imgs, t=t, model_kwargs={"cond": cond_null}, noise=noise
                    )["loss"]
                delta = (loss_null - loss_true).numpy()  # [N_IMAGES], per-image paired delta
                deltas.append(delta)
            deltas = np.stack(deltas, axis=0)  # [n_timesteps, N_IMAGES]
            model_result[label] = {
                "mean_delta_l": float(deltas.mean()),
                "delta_l_ci95": [float(np.percentile(deltas.mean(axis=0), 2.5)),
                                 float(np.percentile(deltas.mean(axis=0), 97.5))] if False else
                                [float(deltas.mean() - 1.96 * deltas.std() / np.sqrt(deltas.size)),
                                 float(deltas.mean() + 1.96 * deltas.std() / np.sqrt(deltas.size))],
                "per_timestep_mean": [float(deltas[i].mean()) for i in range(len(TIMESTEPS))],
            }
            print(f"  null={label:10s} mean_delta_L={model_result[label]['mean_delta_l']:.5f} "
                 f"CI={model_result[label]['delta_l_ci95']}")
        results[model_name] = model_result
        del module, model
    Path(OUT_PATH).write_text(json.dumps({"timesteps": TIMESTEPS, "results": results}, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
