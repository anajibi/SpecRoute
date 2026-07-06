#!/usr/bin/env python
"""Quantify single-level latent swaps as an attribute transfer matrix."""
import argparse, csv, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.hdae.latent_probing.swap_null_grid import swapped_zs

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def _probabilities(classifier, x):
    import torch
    with torch.no_grad():
        return torch.sigmoid(classifier(x)).detach().cpu().numpy()


def rendered_to_classifier_input(x01):
    """Convert rendered [0, 1] images to the classifier's [-1, 1] convention."""
    return x01.mul(2).sub(1).clamp(-1, 1)


def transfer_for_scores(source, donor, recon_source, swaps, eps=1e-6):
    """Return mean transfer, valid counts, and per-pair ratios.

    ``swaps`` has shape ``(K, N, A)``; the other arrays have shape ``(N, A)``.
    Transfer is ``(swap - recon_source) / (donor - source)`` averaged over valid
    denominators for each latent level and attribute.
    """
    source = np.asarray(source, dtype=np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    recon_source = np.asarray(recon_source, dtype=np.float64)
    swaps = np.asarray(swaps, dtype=np.float64)
    denom = donor - source
    valid = np.abs(denom) >= eps
    ratios = np.full_like(swaps, np.nan, dtype=np.float64)
    for level in range(swaps.shape[0]):
        ratios[level] = np.divide(swaps[level] - recon_source, denom,
                                  out=np.full_like(denom, np.nan, dtype=np.float64),
                                  where=valid)
    matrix = np.nanmean(ratios, axis=1)
    valid_counts = np.broadcast_to(valid, swaps.shape).sum(axis=1).astype(np.int64)
    return matrix, valid_counts, ratios


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--num-pairs", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16, help="number of source/donor pairs per batch")
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--denom-eps", type=float, default=1e-6)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    import torch
    from torch.utils.data import DataLoader
    from experiments.hdae.counterfactuals.attribute_classifier import load_classifier
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.lit_module import HDAELitModule

    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    model = module.ema_model
    classifier, clf_state = load_classifier(args.attr_classifier, device=device)
    attr_names = [str(x) for x in clf_state["attribute_names"]]
    num_levels = len(model.hdae_conf.encoder.level_dims)
    T = args.T or cfg.raw["train"]["T_eval"]

    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    loader = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    source_scores, donor_scores, recon_scores = [], [], []
    swap_scores = [[] for _ in range(num_levels)]
    raw_rows = []
    seen = 0
    for batch in loader:
        if seen >= args.num_pairs:
            break
        n = min(args.num_pairs - seen, len(batch["img"]) // 2)
        if n <= 0:
            break
        x_source = batch["img"][:n].to(device)
        x_donor = batch["img"][n:2 * n].to(device)
        source_idx = batch["index"][:n].tolist()
        donor_idx = batch["index"][n:2 * n].tolist()

        with torch.no_grad():
            src = model.encode(x_source)
            donor = model.encode(x_donor)
            x_t = module.encode_stochastic(x_source, src["cond"], T=T)
            recon_source = module.render(x_t, src["cond"], T=T)
            batch_source_scores = _probabilities(classifier, x_source)
            batch_donor_scores = _probabilities(classifier, x_donor)
            batch_recon_scores = _probabilities(classifier, rendered_to_classifier_input(recon_source))
            batch_swap_scores = []
            for level in range(num_levels):
                cond = model.merge(swapped_zs(src["zs"], donor["zs"], [level]))
                swap = module.render(x_t, cond, T=T)
                scores = _probabilities(classifier, rendered_to_classifier_input(swap))
                batch_swap_scores.append(scores)
                swap_scores[level].append(scores)

        source_scores.append(batch_source_scores)
        donor_scores.append(batch_donor_scores)
        recon_scores.append(batch_recon_scores)
        for local_i in range(n):
            for level in range(num_levels):
                row = {"pair": seen + local_i, "source_index": int(source_idx[local_i]),
                       "donor_index": int(donor_idx[local_i]), "level": level}
                for j, name in enumerate(attr_names):
                    row[f"source_{name}"] = float(batch_source_scores[local_i, j])
                    row[f"donor_{name}"] = float(batch_donor_scores[local_i, j])
                    row[f"recon_source_{name}"] = float(batch_recon_scores[local_i, j])
                    row[f"swap_{name}"] = float(batch_swap_scores[level][local_i, j])
                raw_rows.append(row)
        seen += n
        logging.info("processed %d/%d source/donor pairs", seen, args.num_pairs)

    if seen == 0:
        raise ValueError("no source/donor pairs were available")
    source = np.concatenate(source_scores, axis=0)
    donor = np.concatenate(donor_scores, axis=0)
    recon = np.concatenate(recon_scores, axis=0)
    swaps = np.stack([np.concatenate(level_scores, axis=0) for level_scores in swap_scores], axis=0)
    matrix, valid_counts, ratios = transfer_for_scores(source, donor, recon, swaps, eps=args.denom_eps)

    matrix_path = out / "swap_transfer_matrix.csv"
    with open(matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", *attr_names])
        for level in range(num_levels):
            writer.writerow([level, *[float(x) for x in matrix[level]]])

    counts_path = out / "swap_transfer_valid_counts.csv"
    with open(counts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", *attr_names])
        for level in range(num_levels):
            writer.writerow([level, *[int(x) for x in valid_counts[level]]])

    raw_path = out / "swap_transfer_raw_scores.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader(); writer.writerows(raw_rows)

    summary = {"num_pairs": int(seen), "num_levels": int(num_levels), "num_attributes": len(attr_names),
               "attribute_names": attr_names, "denom_eps": args.denom_eps,
               "baseline": "source_self_reconstruction",
               "matrix_csv": str(matrix_path), "valid_counts_csv": str(counts_path),
               "raw_scores_csv": str(raw_path),
               "mean_abs_transfer_by_level": np.nanmean(np.abs(ratios), axis=(1, 2)).tolist()}
    (out / "swap_transfer_summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("wrote swap transfer matrix to %s", matrix_path)


if __name__ == "__main__":
    main()
