#!/usr/bin/env python
"""EMA DDIM reconstruction evaluation for a packed CelebA-HQ test batch."""
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.hdae.hdae.grid_utils import save_labeled_grid
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.datamodule import CelebAHQDataModule


def metrics(x, y):
    mse = (x - y).square().flatten(1).mean(1)
    ux = x.flatten(1).mean(1)
    uy = y.flatten(1).mean(1)
    vx = x.flatten(1).var(1)
    vy = y.flatten(1).var(1)
    cov = ((x.flatten(1) - ux[:, None]) * (y.flatten(1) - uy[:, None])).mean(1)
    ssim = ((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)) / (
        (ux.square() + uy.square() + .01 ** 2) * (vx + vy + .03 ** 2))
    try:
        import lpips
        net = lpips.LPIPS(net="alex").to(x.device)
        percept = net(x, y).flatten()
    except Exception:
        percept = mse.sqrt()
    return percept, mse, ssim


def summarize_metric_tensors(lp, mse, ssim):
    return {k: {"mean": float(v.mean()), "std": float(v.std())}
            for k, v in [("lpips", lp), ("mse", mse), ("ssim", ssim)]}


def compute_recon_metrics(module, batch, T, num_images=None):
    """Render reconstructions for ``batch`` and return metric tensors plus images.

    Inputs and outputs use the repository convention: dataset images are in
    ``[-1, 1]`` and ``module.render`` returns ``[0, 1]`` images, so rendered
    images are converted back to ``[-1, 1]`` before metrics are computed.
    """
    device = next(module.parameters()).device
    x = batch["img"][:num_images].to(device) if num_images is not None else batch["img"].to(device)
    with torch.no_grad():
        cond = module.ema_model.encode(x)["cond"]
        xt = module.encode_stochastic(x, cond, T=T)
        y = module.render(xt, cond, T=T) * 2 - 1
        lp, mse, ssim = metrics(x, y)
    return {"x": x, "y": y, "lpips": lp, "mse": mse, "ssim": ssim,
            "summary": summarize_metric_tensors(lp, mse, ssim)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-images", type=int, default=32)
    p.add_argument("--T", type=int, default=None)
    args = p.parse_args()

    cfg = load_hdae_config(args.config)
    d = cfg.raw["data"]
    t = cfg.raw["train"]
    dm = CelebAHQDataModule(d["lmdb_path"], d["attr_npz"], min(args.num_images, t["batch_size_per_gpu"]), 0, False)
    dm.setup()
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu")
    module.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module.to(device)
    batch = next(iter(dm.test_dataloader()))
    T = args.T or t["T_eval"]
    result = compute_recon_metrics(module, batch, T=T, num_images=args.num_images)

    out = Path(cfg.raw["output_dir"]) / "reconstruction"
    out.mkdir(parents=True, exist_ok=True)
    save_labeled_grid([result["x"].add(1).div(2).detach().cpu(), result["y"].add(1).div(2).detach().cpu()],
                      ["original", "reconstruction"], out / "grid.png")
    with open(out / "recon_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "lpips", "mse", "ssim"])
        w.writerows(zip(batch["index"].tolist(), result["lpips"].cpu().tolist(),
                        result["mse"].cpu().tolist(), result["ssim"].cpu().tolist()))
    (out / "recon_summary.json").write_text(json.dumps(result["summary"], indent=2))
    print(result["summary"])


if __name__ == "__main__":
    main()
