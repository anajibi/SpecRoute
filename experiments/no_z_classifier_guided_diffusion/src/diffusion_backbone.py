from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, DDPMPipeline


@dataclass
class DiffusionBackbone:
    unet: torch.nn.Module
    scheduler: DDIMScheduler
    inverse_scheduler: DDIMInverseScheduler


def load_unconditional_celebahq_backbone(
    model_id: str = "google/ddpm-celebahq-256",
    device: torch.device | str = "cuda",
    use_fp16: bool = True,
) -> DiffusionBackbone:
    device = torch.device(device)
    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32
    pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=dtype)
    unet = pipe.unet.to(device)
    unet.eval()
    for parameter in unet.parameters():
        parameter.requires_grad_(False)

    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    return DiffusionBackbone(unet=unet, scheduler=scheduler, inverse_scheduler=inverse_scheduler)
