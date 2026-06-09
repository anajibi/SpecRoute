"""Hard-frozen DINOv2 and Stable Diffusion VAE wrappers."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class _Frozen(nn.Module):
    def train(self, mode: bool = True):
        super().train(False)
        return self
    def _freeze(self):
        self.requires_grad_(False); super().train(False)


class DINOv2Backbone(_Frozen):
    def __init__(self, variant="vitb14", return_patches=False, model: nn.Module | None = None, config: dict | None = None):
        super().__init__(); config = config or {}; variant = config.get("variant", variant); self.return_patches = config.get("return_patches", return_patches)
        self.model = model if model is not None else torch.hub.load("facebookresearch/dinov2", f"dinov2_{variant}")
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1,3,1,1)); self.register_buffer("std", torch.tensor([.229,.224,.225]).view(1,3,1,1)); self._freeze()

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate((x + 1) / 2, (224, 224), mode="bicubic", align_corners=False)
        feats = self.model.forward_features((x - self.mean) / self.std)
        if isinstance(feats, dict): out = feats["x_norm_patchtokens"] if self.return_patches else feats["x_norm_clstoken"]
        else: out = feats
        return out.detach()


class SDVAEBackbone(_Frozen):
    def __init__(self, model_id="stabilityai/sd-vae-ft-ema", model: nn.Module | None = None, config: dict | None = None):
        super().__init__(); config = config or {}; model_id = config.get("model_id", model_id)
        if model is None:
            from diffusers import AutoencoderKL
            model = AutoencoderKL.from_pretrained(model_id)
        self.model = model; self.scaling_factor = config.get("scaling_factor", getattr(getattr(model, "config", None), "scaling_factor", 1.0)); self._freeze()

    @torch.no_grad()
    def encode(self, x): return (self.model.encode(x).latent_dist.mode() * self.scaling_factor).detach()
    @torch.no_grad()
    def decode(self, z): return self.model.decode(z / self.scaling_factor).sample.detach()
