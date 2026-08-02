#!/usr/bin/env python
"""Fit linear-probe attribute-edit directions for a frozen DiffAE checkpoint.

DiffAEProbeAdapter (diffae_adapter.py) edits along ``z_sem + alpha * w``,
where ``w`` is a per-attribute logistic-regression direction in the frozen
model's semantic latent space. This script extracts z_sem for a sample of
the packed CelebA-HQ images used by the rest of the CF1 pipeline, fits one
direction per modeled attribute, and writes ``directions.pt`` for the
adapter to load. Run once per (checkpoint, dataset) pair, independent of
CF1 evaluation itself -- mirrors attribute_directions.py from the archived
diffae_latent_probe project, refit here against this repo's packed data
rather than imported from there (nothing under experiments/hdae imports
archive/, see AGENDA.md Sec.7).
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "diffae_upstream"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import templates
from experiment import LitModel

from experiments.hdae.data.celeba_hq import CelebAHQPacked

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def extract_z_sem(module, ds, indices, native_image_size, batch_size, device):
    loader = DataLoader(torch.utils.data.Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=8)
    zs = []
    for batch in loader:
        x = batch["img"].to(device)
        if x.shape[-1] != native_image_size:
            x = F.interpolate(x, size=(native_image_size, native_image_size), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            zs.append(module.encode(x).detach().cpu())
    return torch.cat(zs, 0)


def fit_directions(z, labels01, attrs, class_weight="balanced", normalize=True):
    z_np = z.numpy()
    directions, metrics_rows = {}, []
    for attr in attrs:
        y = labels01[attr].values.astype(int)
        x_train, x_val, y_train, y_val = train_test_split(z_np, y, test_size=0.2, random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=200, class_weight=class_weight)
        clf.fit(x_train, y_train)
        logits = clf.decision_function(x_val)
        pred = clf.predict(x_val)
        try:
            auroc = float(roc_auc_score(y_val, logits))
        except ValueError:
            auroc = float("nan")
        w = clf.coef_.reshape(-1)
        if normalize:
            w = w / (np.linalg.norm(w) + 1e-8)
        directions[attr] = {"w": torch.tensor(w, dtype=torch.float32), "b": float(clf.intercept_[0])}
        metrics_rows.append({"attribute": attr, "accuracy": float((pred == y_val).mean()),
                             "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)), "auroc": auroc,
                             "positive_prevalence": float(y.mean())})
    return directions, pd.DataFrame(metrics_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/hdae/configs/diffae_probe.yaml")
    p.add_argument("--ckpt", required=True, help="Frozen DiffAE checkpoint (e.g. ffhq256_autoenc/last.ckpt).")
    p.add_argument("--lmdb-path", required=True)
    p.add_argument("--attr-npz", required=True)
    p.add_argument("--num-images", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="Defaults to config's directions_path.")
    args = p.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    output = Path(args.output or raw["directions_path"])
    output.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("loading frozen DiffAE template=%s ckpt=%s on device=%s", raw["template"], args.ckpt, device)
    conf = getattr(templates, raw["template"])()
    conf.pretrain = None
    conf.latent_infer_path = None
    conf.eval_programs = tuple()
    module = LitModel(conf)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    module.load_state_dict(state["state_dict"], strict=False)
    module = module.to(device).eval()
    module.ema_model.eval()

    ds = CelebAHQPacked(args.lmdb_path, args.attr_npz, flip=False)
    rng = np.random.default_rng(args.seed)
    n = min(args.num_images, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False)
    logging.info("extracting z_sem for %d images at native_image_size=%d", n, raw["native_image_size"])
    z = extract_z_sem(module, ds, indices, raw["native_image_size"], args.batch_size, device)

    attrs01 = pd.DataFrame((ds.attrs[indices] > 0).astype(int), columns=ds.attribute_names)
    modeled_attrs = list(raw["modeled_attrs"])
    logging.info("fitting logistic-regression directions for %s", modeled_attrs)
    directions, metrics = fit_directions(z, attrs01, modeled_attrs)

    torch.save(directions, output)
    metrics_path = output.with_name(output.stem + "_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    logging.info("wrote %d directions to %s; metrics=%s", len(directions), output, metrics_path)
    logging.info("\n%s", metrics.to_string(index=False))


if __name__ == "__main__":
    main()
