"""Hierarchy-to-SD-VAE latent decoders."""
from __future__ import annotations
import math
import torch
from torch import nn


class HierarchicalDecoder(nn.Module):
    """Map all hierarchy levels simultaneously to an SD-VAE latent."""
    def __init__(self, level_dims: list[int], sd_latent_shape=(4, 32, 32), arch="tokens",
                 hidden_dim=512, num_transformer_layers=6, num_heads=8) -> None:
        super().__init__()
        self.level_dims, self.sd_latent_shape, self.arch = list(level_dims), tuple(sd_latent_shape), arch
        out_dim = math.prod(self.sd_latent_shape)
        if arch == "mlp":
            self.network = nn.Sequential(nn.Linear(sum(level_dims), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))
        elif arch == "tokens":
            if hidden_dim % num_heads: raise ValueError("hidden_dim must be divisible by num_heads")
            self.projections = nn.ModuleList([nn.Linear(d, hidden_dim) for d in level_dims])
            self.level_embedding = nn.Parameter(torch.randn(1, len(level_dims), hidden_dim) * .02)
            layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, batch_first=True,
                                               activation="gelu", norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_transformer_layers, enable_nested_tensor=False)
            self.network = nn.Sequential(nn.LayerNorm(hidden_dim * len(level_dims)), nn.Linear(hidden_dim * len(level_dims), out_dim))
        else: raise ValueError(f"Unsupported decoder arch: {arch}")

    def forward(self, zs: list[torch.Tensor]) -> torch.Tensor:
        if len(zs) != len(self.level_dims): raise ValueError(f"Expected {len(self.level_dims)} levels, got {len(zs)}")
        for z, d in zip(zs, self.level_dims):
            if z.ndim != 2 or z.shape[-1] != d: raise ValueError(f"Expected level shape (B, {d}), got {tuple(z.shape)}")
        if self.arch == "mlp": flat = self.network(torch.cat(zs, dim=-1))
        else:
            tokens = torch.stack([proj(z) for proj, z in zip(self.projections, zs)], dim=1) + self.level_embedding
            flat = self.network(self.transformer(tokens).flatten(1))
        return flat.view(zs[0].shape[0], *self.sd_latent_shape)
