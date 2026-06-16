"""Semantic encoder with intermediate taps over the upstream BeatGANs down path."""
from typing import Dict, List, Sequence

import torch
from torch import nn

from model.unet import BeatGANsEncoderConfig, BeatGANsEncoderModel


def stage_channels(conf: BeatGANsEncoderConfig) -> Dict[int, int]:
    """Map stage output resolution to channels, including the deepest stage."""
    return {conf.image_size // (2 ** i): int(mult * conf.model_channels)
            for i, mult in enumerate(conf.channel_mult)}


class ProjectionHead(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, proj: str):
        super().__init__()
        layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(in_channels)]
        if proj == "linear":
            layers.append(nn.Linear(in_channels, out_dim))
        elif proj == "mlp":
            layers.extend([nn.Linear(in_channels, in_channels), nn.SiLU(),
                           nn.Linear(in_channels, out_dim)])
        else:
            raise ValueError(f"unsupported projection: {proj}")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HierarchicalSemanticEncoder(nn.Module):
    """The upstream encoder backbone plus pooled heads, returned coarse-to-fine.

    ``tap_resolutions`` and ``level_dims`` must both be ordered coarse-to-fine
    (smallest spatial resolution first). Hooks are deliberately avoided.
    """
    def __init__(self, base_encoder_config: BeatGANsEncoderConfig,
                 tap_resolutions: Sequence[int], level_dims: Sequence[int],
                 pool: str = "adaptive_avg", proj: str = "linear"):
        super().__init__()
        if len(tap_resolutions) != len(level_dims) or not tap_resolutions:
            raise ValueError("tap_resolutions and level_dims must have equal non-zero length")
        if pool != "adaptive_avg":
            raise ValueError("only adaptive_avg pooling is supported")
        if list(tap_resolutions) != sorted(tap_resolutions):
            raise ValueError("tap_resolutions must be coarse-to-fine (ascending)")
        channels = stage_channels(base_encoder_config)
        invalid = sorted(set(tap_resolutions) - set(channels))
        if invalid:
            raise ValueError(f"invalid tap resolutions {invalid}; valid stages: {sorted(channels)}")
        self.backbone = BeatGANsEncoderModel(base_encoder_config)
        self.tap_resolutions = list(tap_resolutions)
        self.heads = nn.ModuleList([ProjectionHead(channels[r], d, proj)
                                    for r, d in zip(tap_resolutions, level_dims)])

    def forward(self, x) -> List[torch.Tensor]:
        h = x.type(self.backbone.dtype)
        taps = {}
        resolution = self.backbone.conf.image_size
        # The stem and every res/down block are exactly the upstream modules.
        for module in self.backbone.input_blocks:
            h = module(h, emb=None)
            resolution = h.shape[-1]
            if resolution in self.tap_resolutions:
                taps[resolution] = h.type(x.dtype)
        h = self.backbone.middle_block(h, emb=None)
        taps[resolution] = h.type(x.dtype)  # deepest tap includes upstream middle block
        return [head(taps[r]) for r, head in zip(self.tap_resolutions, self.heads)]
