"""Trainable chain encoders for hierarchical latents."""
from __future__ import annotations
import torch
from torch import nn


def _norm(name: str, dim: int) -> nn.Module:
    if name == "layernorm": return nn.LayerNorm(dim)
    if name in {"none", "identity"}: return nn.Identity()
    raise ValueError(f"Unsupported norm: {name}")


class ChainEncoder(nn.Module):
    """Encode F into a chain where level l only receives preceding latents."""
    def __init__(self, input_dim: int, level_dims: list[int], hidden_dim: int = 512,
                 num_layers_per_encoder: int = 3, noise_std: float = 0.0,
                 norm: str = "layernorm") -> None:
        super().__init__()
        if not level_dims or not all(a > b for a, b in zip(level_dims, level_dims[1:])):
            raise AssertionError("level_dims must be strictly decreasing (each Z more compressed than previous)")
        if num_layers_per_encoder < 1: raise ValueError("num_layers_per_encoder must be positive")
        self.K, self.level_dims, self.noise_std = len(level_dims), list(level_dims), noise_std
        self.encoders = nn.ModuleList([
            self._make_mlp(input_dim if i == 0 else sum(level_dims[:i]), hidden_dim, out_dim,
                           num_layers_per_encoder, norm)
            for i, out_dim in enumerate(level_dims)
        ])

    @staticmethod
    def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int, norm: str) -> nn.Sequential:
        if depth == 1: return nn.Sequential(nn.Linear(in_dim, out_dim))
        layers: list[nn.Module] = []
        for i in range(depth - 1):
            layers.extend([nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), _norm(norm, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        zs: list[torch.Tensor] = []
        for i, encoder in enumerate(self.encoders):
            z = encoder(features if i == 0 else torch.cat(zs, dim=-1))
            if self.training and self.noise_std > 0: z = z + self.noise_std * torch.randn_like(z)
            zs.append(z)
        return zs
