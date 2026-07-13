"""Attribute embeddings and per-decoder-block style projections."""
from typing import Sequence

import torch
from torch import nn


class AttributeEmbedding(nn.Module):
    """Sum per-attribute embeddings. Inputs are integer indices 0=neg, 1=pos."""

    def __init__(self, n_attributes: int, attr_embed_dim: int, attr_dropout_prob: float, cfg_drop_prob: float = 0.0):
        super().__init__()
        self.attr_dropout_prob = float(attr_dropout_prob)
        self.cfg_drop_prob = float(cfg_drop_prob)
        if not 0 <= self.cfg_drop_prob < 1:
            raise ValueError("cfg_drop_prob must be in [0, 1)")
        self.embeddings = nn.ModuleList([nn.Embedding(3, attr_embed_dim) for _ in range(n_attributes)])

    def forward(self, y_idx: torch.Tensor, apply_dropout: bool = True) -> torch.Tensor:
        y_idx = y_idx.long()
        if apply_dropout and self.training and self.cfg_drop_prob > 0:
            drop_sample = torch.rand(y_idx.shape[0], 1, device=y_idx.device) < self.cfg_drop_prob
            y_idx = torch.where(drop_sample, torch.full_like(y_idx, 2), y_idx)
        if apply_dropout and self.training and self.attr_dropout_prob > 0:
            drop = torch.rand(y_idx.shape, device=y_idx.device) < self.attr_dropout_prob
            y_idx = torch.where(drop, torch.full_like(y_idx, 2), y_idx)
        return torch.stack([emb(y_idx[:, i]) for i, emb in enumerate(self.embeddings)], dim=0).sum(dim=0)


class PerBlockStyle(nn.Module):
    """One linear style projection per decoder block."""

    def __init__(self, level_dims: Sequence[int], block_to_level: Sequence[int],
                 attr_embed_dim: int, embed_channels: int):
        super().__init__()
        self.block_to_level = list(block_to_level)
        self.projections = nn.ModuleList([
            nn.Linear(level_dims[level] + attr_embed_dim, embed_channels)
            for level in self.block_to_level
        ])

    def forward(self, zs, attr_emb: torch.Tensor):
        return [proj(torch.cat([zs[level], attr_emb], dim=1))
                for level, proj in zip(self.block_to_level, self.projections)]
