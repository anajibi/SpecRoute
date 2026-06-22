"""Merge hierarchical semantic codes before the unchanged DiffAE decoder."""
from typing import Iterable, Sequence
import torch
from torch import nn


class ConcatProjectionMerger(nn.Module):
    """Concatenate per-level latents, with learned per-level null tokens.

    During training, each level is independently replaced by its learned null
    token with probability ``latent_drop_prob``. At evaluation time dropout is
    disabled unless explicit ``null_levels`` are requested. Null tokens are
    ordinary parameters, so they are saved in checkpoints and learn to occupy a
    useful point in each level's latent space.
    """

    def __init__(self, level_dims: Sequence[int], target_dim: int,
                 latent_drop_prob: float = 0.12):
        super().__init__()
        self.level_dims = list(level_dims)
        self.target_dim = target_dim
        self.latent_drop_prob = float(latent_drop_prob)
        if sum(level_dims) != target_dim:
            raise ValueError("concat_proj requires sum(level_dims) == style_ch")
        if not 0 <= self.latent_drop_prob < 1:
            raise ValueError("latent_drop_prob must be in [0, 1)")
        self.null_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for dim in self.level_dims
        ])
        # Keep matched-budget flat/hierarchical comparisons transparent while
        # retaining a learnable mixing layer.
        self.proj = nn.Linear(target_dim, target_dim)
        self.last_null_mask = None

    def _normalize_null_levels(self, null_levels: Iterable[int] = None):
        if null_levels is None:
            return set()
        levels = {int(level) for level in null_levels}
        invalid = sorted(level for level in levels if level < 0 or level >= len(self.level_dims))
        if invalid:
            raise ValueError(f"invalid null levels {invalid}; valid levels are 0..{len(self.level_dims) - 1}")
        return levels

    def _nullify_level(self, z, level: int, force_level: bool):
        token = self.null_tokens[level].to(dtype=z.dtype, device=z.device).unsqueeze(0).expand_as(z)
        if force_level:
            mask = torch.ones(z.shape[0], 1, dtype=torch.bool, device=z.device)
            return token, mask
        if self.training and self.latent_drop_prob > 0:
            mask = torch.rand(z.shape[0], 1, device=z.device) < self.latent_drop_prob
            return torch.where(mask, token, z), mask
        mask = torch.zeros(z.shape[0], 1, dtype=torch.bool, device=z.device)
        return z, mask

    def forward(self, zs, null_levels: Iterable[int] = None, return_mask: bool = False):
        if not zs:
            raise ValueError("at least one latent is required")
        if len(zs) != len(self.level_dims):
            raise ValueError(f"expected {len(self.level_dims)} latents, got {len(zs)}")
        forced = self._normalize_null_levels(null_levels)
        merged, masks = [], []
        for level, z in enumerate(zs):
            if z.shape[1] != self.level_dims[level]:
                raise ValueError(f"level {level} expected dim {self.level_dims[level]}, got {z.shape[1]}")
            z, mask = self._nullify_level(z, level, level in forced)
            merged.append(z); masks.append(mask)
        self.last_null_mask = torch.cat(masks, dim=1).detach()
        cond = self.proj(torch.cat(merged, dim=1))
        if return_mask:
            return cond, self.last_null_mask
        return cond


def build_merger(strategy: str, level_dims: Sequence[int], target_dim: int,
                 latent_drop_prob: float = 0.12):
    if strategy == "concat_proj":
        return ConcatProjectionMerger(level_dims, target_dim, latent_drop_prob)
    if strategy == "per_resolution":
        raise NotImplementedError("per_resolution is experimental and intentionally disabled in phase 1")
    raise ValueError(f"unknown conditioning strategy: {strategy}")
