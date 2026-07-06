"""Binary attribute conditioning and per-decoder-block style projections."""
from typing import Sequence

import torch
from torch import nn


class AttributeEmbedding(nn.Module):
    """Sum independent binary/null embeddings for a fixed set of attributes.

    Input must already be in index space: 0=negative, 1=positive, 2=null/CFG.
    This module never accepts raw {-1, 1} labels or floating probabilities; range
    normalization is owned by the training/evaluation glue.
    """

    def __init__(self, n_attributes: int = 5, attr_embed_dim: int = 128,
                 attr_dropout_prob: float = 0.1):
        super().__init__()
        if n_attributes <= 0:
            raise ValueError("n_attributes must be positive")
        if attr_embed_dim <= 0:
            raise ValueError("attr_embed_dim must be positive")
        if not 0 <= attr_dropout_prob < 1:
            raise ValueError("attr_dropout_prob must be in [0, 1)")
        self.n_attributes = int(n_attributes)
        self.attr_embed_dim = int(attr_embed_dim)
        self.attr_dropout_prob = float(attr_dropout_prob)
        self.embeddings = nn.ModuleList([
            nn.Embedding(3, self.attr_embed_dim) for _ in range(self.n_attributes)
        ])

    def apply_cfg_dropout(self, y_idx: torch.Tensor) -> torch.Tensor:
        if y_idx.max() > 1 or y_idx.min() < 0:
            raise ValueError("CFG dropout expects pre-dropout indices in {0, 1}")
        if not self.training or self.attr_dropout_prob == 0:
            return y_idx
        mask = torch.rand(y_idx.shape, device=y_idx.device) < self.attr_dropout_prob
        return torch.where(mask, torch.full_like(y_idx, 2), y_idx)

    def forward(self, y_idx: torch.Tensor, apply_dropout: bool = True) -> torch.Tensor:
        if y_idx.ndim != 2 or y_idx.shape[1] != self.n_attributes:
            raise ValueError(f"expected y_idx shape [N, {self.n_attributes}], got {tuple(y_idx.shape)}")
        if not torch.is_floating_point(y_idx):
            y_idx = y_idx.long()
        else:
            if not torch.equal(y_idx, y_idx.round()):
                raise ValueError("attribute indices must be integers in {0, 1, 2}")
            y_idx = y_idx.long()
        if y_idx.numel() and (int(y_idx.min()) < 0 or int(y_idx.max()) > 2):
            raise ValueError("attribute indices must be in {0, 1, 2}")
        if apply_dropout:
            y_idx = self.apply_cfg_dropout(y_idx)
        outs = [emb(y_idx[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(outs, dim=0).sum(dim=0)


class PerBlockStyle(nn.Module):
    """Project assigned latent plus attribute embedding to one style per decoder block."""

    def __init__(self, level_dims: Sequence[int], block_to_level: Sequence[int],
                 attr_embed_dim: int, embed_channels: int, proj: str = "linear"):
        super().__init__()
        if proj != "linear":
            raise ValueError("only linear per-block style projection is supported")
        self.level_dims = [int(x) for x in level_dims]
        self.block_to_level = [int(x) for x in block_to_level]
        if not self.block_to_level:
            raise ValueError("block_to_level must be non-empty")
        if min(self.block_to_level) < 0 or max(self.block_to_level) >= len(self.level_dims):
            raise ValueError("block_to_level entries must index level_dims")
        self.projections = nn.ModuleList([
            nn.Linear(self.level_dims[level] + attr_embed_dim, embed_channels)
            for level in self.block_to_level
        ])

    def forward(self, zs, attr_emb: torch.Tensor):
        if len(zs) != len(self.level_dims):
            raise ValueError(f"expected {len(self.level_dims)} latents, got {len(zs)}")
        styles = []
        for block, (level, proj) in enumerate(zip(self.block_to_level, self.projections)):
            z = zs[level]
            if z.shape[1] != self.level_dims[level]:
                raise ValueError(f"level {level} expected dim {self.level_dims[level]}, got {z.shape[1]}")
            styles.append(proj(torch.cat([z, attr_emb], dim=1)))
        return styles
