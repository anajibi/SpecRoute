#!/usr/bin/env python
"""Diagnose whether DDIM-abducted x_T dominates conditional HDAE outputs.

Produces three grids plus one CSV:
  A: fixed x_T, varied z        -> is z inert?
  B: fixed z, varied x_T        -> does x_T dominate?
  C: shallower abduction sweep  -> does reducing t* revive z, and at what recon cost?
"""
import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.hdae.counterfactuals.attribute_classifier import load_classifier
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_index_space
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.grid_utils import save_labeled_grid
from experiments.hdae.hdae.lit_module import HDAELitModule

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")




class MetricComputer:
    def __init__(self, device):
        self.net = None
        try:
            import lpips
            self.net = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as exc:
            logging.warning("LPIPS unavailable; falling back to sqrt(MSE): %s", exc)

    def __call__(self, x, y):
        mse = (x - y).square().flatten(1).mean(1)
        ux = x.flatten(1).mean(1)
        uy = y.flatten(1).mean(1)
        vx = x.flatten(1).var(1)
        vy = y.flatten(1).var(1)
        cov = ((x.flatten(1) - ux[:, None]) * (y.flatten(1) - uy[:, None])).mean(1)
        ssim = ((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)) / (
            (ux.square() + uy.square() + .01 ** 2) * (vx + vy + .03 ** 2))
        if self.net is None:
            percept = mse.sqrt()
        else:
            percept = self.net(x, y).flatten()
        return percept, mse, ssim

def parse_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def classifier_probs(classifier, x01):
    with torch.inference_mode():
        x = x01.mul(2).sub(1).clamp(-1, 1)
        return torch.sigmoid(classifier(x)).detach().cpu().numpy()


def to_minus_one_one(x01):
    return x01.mul(2).sub(1).clamp(-1, 1)


def conditioning_attr_indices(model, dataset_attr_names):
    e = model.hdae_conf.encoder
    attrs = list(e.conditioning_attrs) if e.conditioning_attrs else list(dataset_attr_names[:e.n_attributes])
    missing = [name for name in attrs if name not in dataset_attr_names]
    if missing:
        raise ValueError(f"conditioning_attrs not found in dataset attributes: {missing}")
    return attrs, [dataset_attr_names.index(name) for name in attrs]


def encode_semantic(model, x):
    encoded = model.encode(x)
    return [z.clone() for z in encoded["zs"]]


def cond_dict(zs, y_idx):
    return {"zs": zs, "y_idx": y_idx}


def abduct_at_depth(module, x, zs, y_idx, t_star):
    """DDIM-invert to a requested number of evaluation steps.

    Upstream ``encode_stochastic`` takes ``T`` and constructs a sampler with that
    many timesteps, so shallower abduction is implemented by passing a smaller
    ``T`` rather than editing upstream diffusion code.
    """
    return module.encode_stochastic(x, cond_dict(zs, y_idx), T=int(t_star))


def decode_at_depth(module, x_t, zs, y_idx, t_star):
    return module.render(x_t, cond_dict(zs, y_idx), T=int(t_star)).clamp(0, 1)


def lpips_ssim(metric_fn, real_m11, pred01):
    pred_m11 = to_minus_one_one(pred01)
    lp, mse, ssim = metric_fn(real_m11, pred_m11)
    mean_abs = (real_m11 - pred_m11).abs().flatten(1).mean(1)
    return lp.detach().cpu(), mse.detach().cpu(), ssim.detach().cpu(), mean_abs.detach().cpu()


def pair_metrics(metric_fn, ref01, other01):
    ref_m11 = to_minus_one_one(ref01)
    other_m11 = to_minus_one_one(other01)
    lp, mse, ssim = metric_fn(ref_m11, other_m11)
    mean_abs = (ref01 - other01).abs().flatten(1).mean(1)
    return lp.detach().cpu(), mse.detach().cpu(), ssim.detach().cpu(), mean_abs.detach().cpu()


def append_metric_rows(rows, test, variant, image_ids, lp, mse, ssim, mean_abs, t_fraction=None, t_steps=None):
    for i, image_id in enumerate(image_ids):
        rows.append({
            "test": test,
            "variant": variant,
            "image_id": int(image_id),
            "t_fraction": "" if t_fraction is None else float(t_fraction),
            "t_steps": "" if t_steps is None else int(t_steps),
            "lpips": float(lp[i]),
            "mse": float(mse[i]),
            "ssim": float(ssim[i]),
            "mean_abs_pixel_delta": float(mean_abs[i]),
        })


def mean_from_rows(rows, test, variant, field="lpips", t_fraction=None):
    vals = [float(r[field]) for r in rows if r["test"] == test and r["variant"] == variant
            and (t_fraction is None or float(r["t_fraction"]) == float(t_fraction))]
    return float(np.mean(vals)) if vals else float("nan")


def plot_c_curve(path, c_summary):
    import matplotlib.pyplot as plt
    xs = [row["t_fraction"] for row in c_summary]
    z_inf = [row["z_influence_lpips_mean"] for row in c_summary]
    recon = [row["recon_lpips_mean"] for row in c_summary]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, z_inf, marker="o", label="z influence LPIPS: real vs zero z")
    ax.plot(xs, recon, marker="o", label="recon LPIPS: real vs recon0")
    ax.set_xlabel("abduction depth fraction of T")
    ax.set_ylabel("LPIPS")
    ax.set_title("Shallow abduction tradeoff")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def verdict_from(rows, c_summary, z_eps, injection_eps, recon_max):
    a_zero = mean_from_rows(rows, "A_z_fixed_xT", "z_zero")
    b_rand = np.nanmean([mean_from_rows(rows, "B_xT_fixed_z", f"xT_rand{i}") for i in [1, 2, 3]])
    if b_rand < injection_eps:
        return "INJECTION_DEAD"
    for row in c_summary:
        if row["z_influence_lpips_mean"] >= z_eps and row["recon_lpips_mean"] <= recon_max:
            return "SHALLOW_FIXES"
    return "RETRAIN_NEEDED"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--T", type=int, default=None, help="Base eval DDIM steps; defaults to train.T_eval")
    p.add_argument("--t-fractions", default="1.0,0.75,0.5,0.25")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--z-influence-lpips-threshold", type=float, default=0.05)
    p.add_argument("--injection-dead-lpips-threshold", type=float, default=0.01)
    p.add_argument("--max-acceptable-recon-lpips", type=float, default=0.20)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = device == "cuda"
    out = Path(args.output_dir)
    grid_dir = out / "diag_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    T_base = int(args.T or cfg.raw["train"]["T_eval"])
    t_fractions = parse_floats(args.t_fractions)
    t_steps = {frac: max(1, int(round(T_base * frac))) for frac in t_fractions}

    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    metric_fn = MetricComputer(device)
    model = module.ema_model
    classifier, clf_state = load_classifier(args.attr_classifier, device=device)
    classifier.eval()
    attr_names = [str(x) for x in clf_state["attribute_names"]]

    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    cond_attrs, cond_indices = conditioning_attr_indices(model, ds.attribute_names)
    loader = DataLoader(ds, batch_size=min(args.batch_size, args.num_images), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    x = batch["img"][:args.num_images].to(device)
    image_ids = batch["index"][:len(x)].tolist()
    y_raw = batch["attr"][:len(x), cond_indices].to(device)
    y_idx = to_index_space(y_raw, model.hdae_conf.encoder.attr_input_range).to(device)

    rows = []
    c_summary = []
    with torch.inference_mode(), torch.autocast(device_type=device, enabled=amp_enabled):
        zs = encode_semantic(model, x)
        x_t_full = abduct_at_depth(module, x, zs, y_idx, T_base)
        recon_real = decode_at_depth(module, x_t_full, zs, y_idx, T_base)

        # Test A: fixed x_T, vary z.
        zs_zero = [torch.zeros_like(z) for z in zs]
        zs_other = [z.roll(shifts=1, dims=0) for z in zs]
        zs_rand = [torch.randn_like(z) for z in zs]
        a_outputs = {
            "z_real": recon_real,
            "z_zero": decode_at_depth(module, x_t_full, zs_zero, y_idx, T_base),
            "z_other": decode_at_depth(module, x_t_full, zs_other, y_idx, T_base),
            "z_rand": decode_at_depth(module, x_t_full, zs_rand, y_idx, T_base),
        }
        save_labeled_grid([x.add(1).div(2).cpu(), a_outputs["z_real"].cpu(), a_outputs["z_zero"].cpu(),
                           a_outputs["z_other"].cpu(), a_outputs["z_rand"].cpu()],
                          ["real", "z_real", "z_zero", "z_other", "z_rand"], grid_dir / "A.png")
        for name in ["z_zero", "z_other", "z_rand"]:
            append_metric_rows(rows, "A_z_fixed_xT", name, image_ids, *pair_metrics(metric_fn, a_outputs["z_real"], a_outputs[name]))

        # Test B: fixed z, vary x_T.
        b_outputs = {"xT_abducted": recon_real}
        for i in [1, 2, 3]:
            b_outputs[f"xT_rand{i}"] = decode_at_depth(module, torch.randn_like(x_t_full), zs, y_idx, T_base)
        save_labeled_grid([x.add(1).div(2).cpu(), b_outputs["xT_abducted"].cpu(), b_outputs["xT_rand1"].cpu(),
                           b_outputs["xT_rand2"].cpu(), b_outputs["xT_rand3"].cpu()],
                          ["real", "xT_abducted", "xT_rand1", "xT_rand2", "xT_rand3"], grid_dir / "B.png")
        for name in ["xT_rand1", "xT_rand2", "xT_rand3"]:
            append_metric_rows(rows, "B_xT_fixed_z", name, image_ids, *pair_metrics(metric_fn, b_outputs["xT_abducted"], b_outputs[name]))

        # Test C: sweep abduction depth and compare z influence vs reconstruction.
        c_grid_rows = []
        c_grid_labels = []
        for frac in t_fractions:
            steps = t_steps[frac]
            x_t = abduct_at_depth(module, x, zs, y_idx, steps)
            recon = decode_at_depth(module, x_t, zs, y_idx, steps)
            z_zero = decode_at_depth(module, x_t, zs_zero, y_idx, steps)
            lp_z, mse_z, ssim_z, abs_z = pair_metrics(metric_fn, recon, z_zero)
            lp_r, mse_r, ssim_r, abs_r = lpips_ssim(metric_fn, x, recon)
            append_metric_rows(rows, "C_tstar", "z_influence_real_vs_zero", image_ids, lp_z, mse_z, ssim_z, abs_z, frac, steps)
            append_metric_rows(rows, "C_tstar", "reconstruction", image_ids, lp_r, mse_r, ssim_r, abs_r, frac, steps)
            c_summary.append({
                "t_fraction": float(frac),
                "t_steps": int(steps),
                "z_influence_lpips_mean": float(lp_z.mean()),
                "recon_lpips_mean": float(lp_r.mean()),
                "recon_ssim_mean": float(ssim_r.mean()),
            })
            c_grid_rows.extend([x.add(1).div(2).cpu(), recon.cpu(), z_zero.cpu()])
            c_grid_labels.extend([f"real_t{frac:g}", f"recon_t{frac:g}", f"z_zero_t{frac:g}"])
        save_labeled_grid(c_grid_rows, c_grid_labels, grid_dir / "C.png")

    # Attribute readout for bookkeeping: does classifier see source attributes on recon0?
    recon_probs = classifier_probs(classifier, recon_real)
    for local_i, image_id in enumerate(image_ids):
        for attr in cond_attrs:
            if attr in attr_names:
                rows.append({"test": "attr_readout", "variant": attr, "image_id": int(image_id),
                             "t_fraction": 1.0, "t_steps": T_base,
                             "lpips": "", "mse": "", "ssim": "",
                             "mean_abs_pixel_delta": float(recon_probs[local_i, attr_names.index(attr)])})

    metrics_path = out / "diag_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="") as f:
        fieldnames = ["test", "variant", "image_id", "t_fraction", "t_steps", "lpips", "mse", "ssim", "mean_abs_pixel_delta"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    plot_c_curve(out / "diag_C_curve.png", c_summary)
    verdict = verdict_from(rows, c_summary, args.z_influence_lpips_threshold,
                           args.injection_dead_lpips_threshold, args.max_acceptable_recon_lpips)
    summary = {"verdict": verdict, "config": args.config, "ckpt": args.ckpt, "T_base": T_base,
               "t_fractions": t_fractions, "t_steps": t_steps, "conditioning_attrs": cond_attrs,
               "attribute_names": attr_names, "c_summary": c_summary,
               "metrics_csv": str(metrics_path), "grid_dir": str(grid_dir)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(verdict)
    logging.info("wrote x_T dominance diagnostics to %s", out)


if __name__ == "__main__":
    main()
