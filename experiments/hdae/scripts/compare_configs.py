#!/usr/bin/env python
"""Compare HDAE configs on reconstruction, probe quality, and edit preservation."""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import numpy as np


def _fmt(value):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def read_probe_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_probe_metrics(probe_root, raw_config, config_path):
    output_dir = Path(raw_config["output_dir"])
    candidates = []
    if probe_root:
        root = Path(probe_root)
        candidates.extend([
            root / output_dir.name / "latent_probing" / "probes" / "probe_metrics.csv",
            root / output_dir.name / "probe_metrics.csv",
            root / Path(config_path).stem / "latent_probing" / "probes" / "probe_metrics.csv",
            root / Path(config_path).stem / "probe_metrics.csv",
        ])
    candidates.append(output_dir / "latent_probing" / "probes" / "probe_metrics.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def probe_summary(rows, tap_resolutions):
    if not rows:
        return {}, [], []
    metric = "test_balanced_accuracy"
    values = [float(r[metric]) for r in rows if r.get(metric) not in (None, "")]
    by_level = defaultdict(list)
    by_attr = defaultdict(list)
    for row in rows:
        if row.get(metric) in (None, ""):
            continue
        level = int(row["level"])
        value = float(row[metric])
        by_level[level].append(value)
        by_attr[row["attribute_name"]].append((value, level))
    best = []
    for attr, vals in sorted(by_attr.items()):
        value, level = max(vals, key=lambda x: x[0])
        best.append({"attribute_name": attr, "best_level": level, "best_resolution": tap_resolutions[level], metric: value})
    level_rows = [
        {"level": level, "resolution": tap_resolutions[level], f"mean_{metric}": float(np.mean(vals)), "num_attributes": len(vals)}
        for level, vals in sorted(by_level.items())
    ]
    best_res_counts = dict(Counter(row["best_resolution"] for row in best))
    summary = {"probe_mean_test_balanced_accuracy": float(np.mean(values)) if values else None,
               "best_level_by_resolution": best_res_counts}
    return summary, level_rows, best


def find_preservation_csv(raw_config):
    output_dir = Path(raw_config["output_dir"])
    candidates = [
        output_dir / "counterfactuals" / "preservation_sweep" / "preservation_sweep.csv",
        output_dir / "preservation_sweep" / "preservation_sweep.csv",
        output_dir / "counterfactuals" / "preservation_sweep.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def editing_summary(raw_config):
    path = find_preservation_csv(raw_config)
    if path is None:
        return {"preservation_sweep_csv": "", "edit_s1_target_intended_flip_rate": None,
                "edit_s1_nontarget_abs_delta": None}
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if abs(float(row["strength"]) - 1.0) < 1e-9:
                rows.append(row)
    if not rows:
        return {"preservation_sweep_csv": str(path), "edit_s1_target_intended_flip_rate": None,
                "edit_s1_nontarget_abs_delta": None}
    return {"preservation_sweep_csv": str(path),
            "edit_s1_target_intended_flip_rate": float(np.mean([float(r.get("target_intended_flip_rate", r.get("target_flip_rate", 0))) for r in rows])),
            "edit_s1_nontarget_abs_delta": float(np.mean([float(r["non_target_abs_delta_mean"]) for r in rows]))}


def config_name(path, raw):
    return Path(raw.get("output_dir", Path(path).stem)).name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--probe-root", default=None)
    p.add_argument("--num-images", type=int, default=32)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    if len(args.configs) != len(args.ckpts):
        raise ValueError("--configs and --ckpts must have the same length")

    import torch
    from torch.utils.data import DataLoader
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.lit_module import HDAELitModule
    from experiments.hdae.scripts.reconstruct import compute_recon_metrics

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows, detail = [], {"configs": []}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for config_path, ckpt in zip(args.configs, args.ckpts):
        cfg = load_hdae_config(config_path)
        raw = cfg.raw
        name = config_name(config_path, raw)
        dims = [int(x) for x in raw["encoder"]["level_dims"]]
        taps = [int(x) for x in raw["encoder"]["tap_resolutions"]]
        ds = CelebAHQPacked(raw["data"]["lmdb_path"], raw["data"]["attr_npz"], flip=False)
        batch = next(iter(DataLoader(ds, batch_size=args.num_images, shuffle=False, num_workers=0)))
        module = HDAELitModule.load_from_checkpoint(ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
        recon = compute_recon_metrics(module, batch, T=args.T or raw["train"]["T_eval"], num_images=args.num_images)

        probe_path = find_probe_metrics(args.probe_root, raw, config_path)
        probe_rows = read_probe_rows(probe_path)
        probe_sum, probe_level_rows, best_probe_rows = probe_summary(probe_rows, taps)
        edit = editing_summary(raw)
        row = {"config": name, "config_path": config_path, "ckpt": ckpt, "K": len(dims),
               "semantic_dim": sum(dims), "level_dims": "/".join(map(str, dims)),
               "tap_resolutions": "/".join(map(str, taps)),
               "lpips_mean": recon["summary"]["lpips"]["mean"], "lpips_std": recon["summary"]["lpips"]["std"],
               "mse_mean": recon["summary"]["mse"]["mean"], "mse_std": recon["summary"]["mse"]["std"],
               "ssim_mean": recon["summary"]["ssim"]["mean"], "ssim_std": recon["summary"]["ssim"]["std"],
               "probe_metrics_csv": str(probe_path), **probe_sum, **edit}
        rows.append(row)
        detail["configs"].append({"config": name, "probe_by_level": probe_level_rows,
                                  "best_probe_level_by_attribute": best_probe_rows,
                                  "best_level_by_resolution": probe_sum.get("best_level_by_resolution", {})})

    csv_path = out / "config_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    columns = ["config", "K", "semantic_dim", "level_dims", "tap_resolutions", "lpips_mean", "mse_mean", "ssim_mean",
               "probe_mean_test_balanced_accuracy", "edit_s1_target_intended_flip_rate", "edit_s1_nontarget_abs_delta"]
    md = markdown_table(rows, columns)
    md_path = out / "config_comparison.md"
    md_path.write_text(md + "\n")
    detail.update({"csv": str(csv_path), "markdown": str(md_path), "num_images": args.num_images})
    (out / "summary.json").write_text(json.dumps(detail, indent=2))
    print(md)


if __name__ == "__main__":
    main()
