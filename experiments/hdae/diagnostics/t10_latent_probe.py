#!/usr/bin/env python
"""T10: latent linear probe -- direct test of leakage and of the assumed capacity ordering.

Encode images with each checkpoint's encoder (no diffusion sampling needed, just model.encode),
concatenate the hierarchical zs levels into one feature vector, fit a probe (logistic regression
for digit, ridge for thickness/intensity/hue) on TRAIN-partition encodings, evaluate held-out
accuracy/R^2 on VAL-partition encodings. Compares across k (capacity/depth) and, for k=11, across
training steps (66k vs 75k -- 30k/45k unavailable, see RECON.md).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from experiments.hdae.counterfactuals import hdae_adapter  # noqa: F401
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

PACKED = "experiments/hdae/data/packed/morphomnist_70k.h5"
N_TRAIN_PROBE = 2000
N_VAL_PROBE = 500
BATCH = 128
OUT_PATH = "experiments/hdae/outputs/diagnostics_t10_probe.json"

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
PROBE_ATTRS = ["digit", "thickness", "intensity", "hue"]


def load_module(config_path, ckpt_path, device="cpu"):
    cfg = load_hdae_config(config_path)
    module = HDAELitModule(cfg.train_conf)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    module.load_state_dict(ckpt["state_dict"], strict=True)
    module.eval()
    module.to(device)
    return module


def encode_all(model, ds, indices, device):
    feats = []
    for start in range(0, len(indices), BATCH):
        chunk = indices[start:start + BATCH]
        imgs = torch.stack([ds[int(i)]["img"] for i in chunk]).to(device)
        with torch.no_grad():
            zs = model.encode(imgs)
        flat = torch.cat([z.reshape(z.shape[0], -1) for z in zs], dim=1)
        feats.append(flat.numpy())
    return np.concatenate(feats, axis=0)


def main():
    device = "cpu"
    ds = MorphoMNISTPacked(PACKED)
    names = ds.attribute_names
    train_mask = ds.partitions == 0
    val_mask = ds.partitions == 1
    train_idx_all = np.where(train_mask)[0]
    val_idx_all = np.where(val_mask)[0]

    rng = np.random.RandomState(0)
    train_idx = rng.choice(train_idx_all, size=N_TRAIN_PROBE, replace=False)
    val_idx = rng.choice(val_idx_all, size=N_VAL_PROBE, replace=False)

    y_train = {a: ds.attrs[train_idx, names.index(a)] for a in PROBE_ATTRS}
    y_val = {a: ds.attrs[val_idx, names.index(a)] for a in PROBE_ATTRS}

    results = {}
    for model_name, config_path, ckpt_path in MODELS:
        print(f"=== {model_name} ===")
        module = load_module(config_path, ckpt_path, device)
        model = module.ema_model

        z_train = encode_all(model, ds, train_idx, device)
        z_val = encode_all(model, ds, val_idx, device)
        print(f"  z dim: {z_train.shape[1]}")

        model_result = {"z_dim": int(z_train.shape[1])}
        for attr in PROBE_ATTRS:
            if attr == "digit":
                clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="auto")
                clf.fit(z_train, y_train[attr].astype(int))
                acc = float(clf.score(z_val, y_val[attr].astype(int)))
                model_result[attr] = {"metric": "accuracy", "value": acc}
                print(f"  digit probe accuracy: {acc:.4f}")
            else:
                reg = Ridge(alpha=10.0)
                reg.fit(z_train, y_train[attr])
                pred = reg.predict(z_val)
                ss_res = float(np.sum((y_val[attr] - pred) ** 2))
                ss_tot = float(np.sum((y_val[attr] - y_val[attr].mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot
                model_result[attr] = {"metric": "r2", "value": r2}
                print(f"  {attr} probe R^2: {r2:.4f}")
        results[model_name] = model_result
        del module, model
    Path(OUT_PATH).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
