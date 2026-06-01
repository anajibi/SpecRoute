from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, DDPMPipeline


@dataclass
class DiffusionBackbone:
    unet: torch.nn.Module
    scheduler: DDIMScheduler
    inverse_scheduler: DDIMInverseScheduler


def _build_ddim_scheduler_config(pipe_scheduler_config: Any, *, clip_sample: bool) -> dict[str, Any]:
    """Return a DDIM config that is safe for deterministic inversion/reconstruction.

    Diffusers DDIM schedulers default to clipping every predicted x0 estimate.
    That is useful for some generation loops, but it is lossy for DDIM inversion:
    every inverse and reconstruction step recombines the clipped estimate into the
    next sample, which can destroy the source image even when no classifier
    guidance is applied.  We keep final image clamping at save time and disable
    intermediate scheduler clipping by default.
    """
    scheduler_config = dict(pipe_scheduler_config)
    scheduler_config["clip_sample"] = bool(clip_sample)
    return scheduler_config


def load_unconditional_celebahq_backbone(
    model_id: str = "google/ddpm-celebahq-256",
    device: torch.device | str = "cuda",
    use_fp16: bool = True,
    clip_sample: bool = False,
) -> DiffusionBackbone:
    device = torch.device(device)
    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32
    pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=dtype)
    unet = pipe.unet.to(device)
    unet.eval()
    for parameter in unet.parameters():
        parameter.requires_grad_(False)

    scheduler_config = _build_ddim_scheduler_config(pipe.scheduler.config, clip_sample=clip_sample)
    scheduler = DDIMScheduler.from_config(scheduler_config)
    inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config)
    return DiffusionBackbone(unet=unet, scheduler=scheduler, inverse_scheduler=inverse_scheduler)
