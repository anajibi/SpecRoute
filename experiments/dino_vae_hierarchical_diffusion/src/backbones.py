"""Frozen reconstructive and semantic backbones."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class FrozenSDVAE(nn.Module):
    """Thin deterministic wrapper around a diffusers AutoencoderKL."""
    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse", model: nn.Module | None = None):
        super().__init__()
        if model is None:
            from diffusers import AutoencoderKL
            model = AutoencoderKL.from_pretrained(model_id)
        self.model = model
        _freeze(self.model)

    def train(self, mode: bool = True):
        super().train(False); self.model.eval(); return self

    @property
    def scaling_factor(self) -> float:
        return float(getattr(self.model.config, "scaling_factor", 0.18215))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x).latent_dist.mode() * self.scaling_factor

    @torch.no_grad()
    def decode(self, z_scaled: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z_scaled / self.scaling_factor).sample


class FrozenDINOv2(nn.Module):
    """DINOv2 feature wrapper accepting images in [-1,1] or [0,1]."""
    def __init__(self, variant: str = "dinov2_vitb14", model: nn.Module | None = None):
        super().__init__()
        self.model = model if model is not None else torch.hub.load("facebookresearch/dinov2", variant)
        self.register_buffer("mean", torch.tensor([.485,.456,.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("std", torch.tensor([.229,.224,.225]).view(1,3,1,1), persistent=False)
        _freeze(self.model)

    def train(self, mode: bool = True):
        super().train(False); self.model.eval(); return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.amin().item() < 0: x = (x + 1) / 2
        x = F.interpolate(x, (224, 224), mode="bicubic", align_corners=False)
        features = self.model.forward_features((x - self.mean) / self.std)
        cls, patches = features["x_norm_clstoken"], features["x_norm_patchtokens"]
        side = int(patches.shape[1] ** .5)
        feature_map = patches.reshape(x.shape[0], side, side, patches.shape[-1]).permute(0,3,1,2)
        return cls.detach(), feature_map.detach()
