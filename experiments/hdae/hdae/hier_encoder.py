"""Hierarchical semantic encoder with explicit upstream block taps."""
from typing import Dict, List, Sequence

import torch
from torch import nn

from model.unet import BeatGANsEncoderConfig, BeatGANsEncoderModel


def stage_channels(conf: BeatGANsEncoderConfig) -> Dict[int, int]:
    return {conf.image_size // (2 ** i): int(mult * conf.model_channels)
            for i, mult in enumerate(conf.channel_mult)}


class MeanStdProjectionHead(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * in_channels)
        self.proj = nn.Linear(2 * in_channels, out_dim)

    def forward(self, x):
        flat = x.flatten(2)
        return self.proj(self.norm(torch.cat([flat.mean(-1), flat.std(-1, unbiased=False)], dim=1)))


class HierarchicalSemanticEncoder(nn.Module):
    """Emit K latents from configured input-block taps and optional ``mid`` tap."""

    def __init__(self, base_encoder_config: BeatGANsEncoderConfig,
                 tap_block_ids: Sequence[int | str], level_dims: Sequence[int]):
        super().__init__()
        self.backbone = BeatGANsEncoderModel(base_encoder_config)
        self.tap_block_ids = list(tap_block_ids)
        self.tap_to_slot = {tap: i for i, tap in enumerate(self.tap_block_ids)}
        channels = self._tap_channels(base_encoder_config)
        self.heads = nn.ModuleList([MeanStdProjectionHead(channels[tap], dim)
                                    for tap, dim in zip(self.tap_block_ids, level_dims)])

    def _tap_channels(self, conf: BeatGANsEncoderConfig):
        channels = {}
        h = torch.zeros(1, conf.in_channels, conf.image_size, conf.image_size)
        with torch.no_grad():
            for i, block in enumerate(self.backbone.input_blocks):
                h = block(h, emb=None)
                if i in self.tap_to_slot:
                    channels[i] = h.shape[1]
            h = self.backbone.middle_block(h, emb=None)
            if "mid" in self.tap_to_slot:
                channels["mid"] = h.shape[1]
        return channels

    def forward(self, x) -> List[torch.Tensor]:
        h = x
        taps = [None] * len(self.tap_block_ids)
        for i, block in enumerate(self.backbone.input_blocks):
            h = block(h, emb=None)
            if i in self.tap_to_slot:
                taps[self.tap_to_slot[i]] = h
        h = self.backbone.middle_block(h, emb=None)
        if "mid" in self.tap_to_slot:
            taps[self.tap_to_slot["mid"]] = h
        return [head(tap) for head, tap in zip(self.heads, taps)]
