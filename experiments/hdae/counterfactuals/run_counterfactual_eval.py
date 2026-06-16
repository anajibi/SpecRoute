#!/usr/bin/env python
"""Latent-direction pseudo-counterfactual evaluation with preservation metrics."""
import argparse, csv, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from experiments.hdae.counterfactuals.attribute_classifier import load_classifier
from experiments.hdae.counterfactuals.directions import (
    choose_probe_row, direction_from_probe_checkpoint, probe_weight_path,
    summarize_attribute_changes,
)
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def _probabilities(classifier, x):
    with torch.no_grad():
        return torch.sigmoid(classifier(x)).detach().cpu().numpy()


def _make_cf(module, x, row, direction, strength, direction_sign, T):
    model = module.ema_model
    encoded = model.encode(x)
    zs = [z.clone() for z in encoded["zs"]]
    level = int(row["level"])
    d = torch.as_tensor(direction, dtype=zs[level].dtype, device=zs[level].device)[None, :]
    sign = 1.0 if direction_sign == "positive" else -1.0
    zs[level] = zs[level] + sign * float(strength) * d
    cond_cf = model.merge(zs)
    x_t = module.encode_stochastic(x, encoded["cond"], T=T)
    recon = module.render(x_t, cond_cf, T=T)
    return recon, encoded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--probe-metrics", required=True)
    p.add_argument("--probe-weights-dir", required=True)
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--attribute", default="Smiling")
    p.add_argument("--level", default="best", help="best or integer latent level")
    p.add_argument("--direction", choices=["positive", "negative", "both"], default="both")
    p.add_argument("--strength", type=float, default=2.0)
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Loading HDAE checkpoint %s", args.ckpt)
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    logging.info("Loading attribute classifier %s", args.attr_classifier)
    classifier, clf_state = load_classifier(args.attr_classifier, device=device)
    attr_names = [str(x) for x in clf_state["attribute_names"]]
    target_index = attr_names.index(args.attribute)
    row = choose_probe_row(args.probe_metrics, args.attribute, args.level)
    weight_path = probe_weight_path(args.probe_weights_dir, row)
    direction, probe_state = direction_from_probe_checkpoint(weight_path)
    logging.info("Using direction attribute=%s level=%s probe=%s val_balanced=%s", args.attribute, row["level"], weight_path, row.get("val_balanced_accuracy"))
    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    directions = ["positive", "negative"] if args.direction == "both" else [args.direction]
    all_rows, first_grid = [], None
    T = args.T or cfg.raw["train"]["T_eval"]
    seen = 0
    for batch in loader:
        if seen >= args.num_images: break
        x = batch["img"][:args.num_images - seen].to(device)
        before = _probabilities(classifier, x)
        for direction_sign in directions:
            cf, _encoded = _make_cf(module, x, row, direction, args.strength, direction_sign, T)
            after = _probabilities(classifier, cf.mul(2).sub(1).clamp(-1, 1))
            delta = after - before
            for local_i, index in enumerate(batch["index"][:len(x)].tolist()):
                rec = {"index": int(index), "direction": direction_sign,
                       "target_before": float(before[local_i, target_index]),
                       "target_after": float(after[local_i, target_index]),
                       "target_delta": float(delta[local_i, target_index])}
                for j, name in enumerate(attr_names):
                    rec[f"delta_{name}"] = float(delta[local_i, j])
                all_rows.append(rec)
            if first_grid is None:
                first_grid = torch.cat([x.add(1).div(2), cf.clamp(0, 1)], dim=0).detach().cpu()
        seen += len(x)
        logging.info("processed %d/%d images", min(seen, args.num_images), args.num_images)
    csv_path = out / f"counterfactual_{args.attribute}_level{row['level']}_strength{args.strength}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader(); writer.writerows(all_rows)
    # Aggregate by requested direction.
    summaries = {}
    for direction_sign in directions:
        rows = [r for r in all_rows if r["direction"] == direction_sign]
        before = np.asarray([[r["target_before"] if j == target_index else 0 for j in range(len(attr_names))] for r in rows])
        # Reconstruct before/after matrices from delta columns for summary target/non-target deltas.
        delta = np.asarray([[r[f"delta_{name}"] for name in attr_names] for r in rows])
        summaries[direction_sign] = summarize_attribute_changes(np.zeros_like(delta), delta, target_index)
    summary = {"attribute": args.attribute, "target_index": target_index, "level": int(row["level"]),
               "strength": args.strength, "probe_weight_path": str(weight_path), "csv": str(csv_path),
               "summaries": summaries}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if first_grid is not None:
        save_image(first_grid, out / "counterfactual_grid.png", nrow=first_grid.shape[0] // 2)
    logging.info("wrote %s and summary.json", csv_path)


if __name__ == "__main__":
    main()
