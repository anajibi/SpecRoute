"""Merge hierarchical semantic codes before the unchanged DiffAE decoder."""
from typing import Sequence
import torch
from torch import nn


class ConcatProjectionMerger(nn.Module):
    def __init__(self, level_dims: Sequence[int], target_dim: int):
        super().__init__()
        if sum(level_dims) != target_dim:
            raise ValueError("concat_proj requires sum(level_dims) == style_ch")
        # Keep matched-budget flat/hierarchical comparisons transparent while
        # retaining a learnable mixing layer.
        self.proj = nn.Linear(target_dim, target_dim)

    def forward(self, zs):
        if not zs:
            raise ValueError("at least one latent is required")
        return self.proj(torch.cat(zs, dim=1))


def build_merger(strategy: str, level_dims: Sequence[int], target_dim: int):
    if strategy == "concat_proj":
        return ConcatProjectionMerger(level_dims, target_dim)
    if strategy == "per_resolution":
        raise NotImplementedError("per_resolution is experimental and intentionally disabled in phase 1")
    raise ValueError(f"unknown conditioning strategy: {strategy}")
