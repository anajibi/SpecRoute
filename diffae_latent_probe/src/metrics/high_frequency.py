from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_blur(batch: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return batch
    radius = max(1, int(3 * sigma + 0.5))
    size = radius * 2 + 1
    device = batch.device
    dtype = batch.dtype
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, size, size)
    kernel_2d = kernel_2d.repeat(batch.shape[1], 1, 1, 1)
    batch_pad = F.pad(batch, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(batch_pad, kernel_2d, padding=0, groups=batch.shape[1])


def highpass(batch: torch.Tensor, sigma: float) -> torch.Tensor:
    return batch - gaussian_blur(batch, sigma=sigma)


def hf_mse(original: torch.Tensor, recon: torch.Tensor, sigma: float) -> float:
    diff = highpass(original.unsqueeze(0), sigma=sigma) - highpass(recon.unsqueeze(0), sigma=sigma)
    return float(torch.mean(diff ** 2).item())


def hf_l1(original: torch.Tensor, recon: torch.Tensor, sigma: float) -> float:
    diff = highpass(original.unsqueeze(0), sigma=sigma) - highpass(recon.unsqueeze(0), sigma=sigma)
    return float(torch.mean(diff.abs()).item())

