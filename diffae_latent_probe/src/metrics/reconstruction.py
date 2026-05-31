from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def mse(original: torch.Tensor, recon: torch.Tensor) -> float:
    diff = original.detach().cpu().numpy() - recon.detach().cpu().numpy()
    return float(np.mean(diff ** 2))


def ssim(original: torch.Tensor, recon: torch.Tensor) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim_fn

        orig_np = original.permute(1, 2, 0).detach().cpu().numpy()
        recon_np = recon.permute(1, 2, 0).detach().cpu().numpy()
        return float(ssim_fn(orig_np, recon_np, channel_axis=2, data_range=1.0))
    except Exception:
        return float("nan")


def lpips(metric, device: torch.device, original: torch.Tensor, recon: torch.Tensor) -> float:
    if metric is None:
        return float("nan")
    with torch.inference_mode():
        orig = original.to(device, non_blocking=True) * 2.0 - 1.0
        rec = recon.to(device, non_blocking=True) * 2.0 - 1.0
        score = metric(orig.unsqueeze(0), rec.unsqueeze(0))
    return float(score.detach().cpu().view(-1)[0].item())

