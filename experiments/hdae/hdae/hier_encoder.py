"""Semantic encoder with explicit block taps over the upstream BeatGANs down path."""
from typing import Dict, List, Sequence, Union

import torch
from torch import nn

from model.unet import BeatGANsEncoderConfig, BeatGANsEncoderModel

BlockTap = Union[int, str]


def stage_channels(conf: BeatGANsEncoderConfig) -> Dict[int, int]:
    """Map stage output resolution to channels, including the deepest stage."""
    return {conf.image_size // (2 ** i): int(mult * conf.model_channels)
            for i, mult in enumerate(conf.channel_mult)}


class MeanStdProjectionHead(nn.Module):
    """LayerNorm(2C) -> Linear(2C, d) over spatial mean and std statistics."""

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * in_channels), nn.Linear(2 * in_channels, out_dim))

    def forward(self, x):
        flat = x.flatten(2)
        stats = torch.cat([flat.mean(dim=-1), flat.std(dim=-1, unbiased=False)], dim=1)
        return self.net(stats)


class HierarchicalSemanticEncoder(nn.Module):
    """Upstream encoder backbone plus mean/std heads, returned coarse-to-fine.

    ``tap_block_ids`` indexes ``BeatGANsEncoderModel.input_blocks`` and may use
    the sentinel ``"mid"`` for the post-middle-block feature. Hooks are avoided
    so K=11 can tap multiple blocks at the same spatial resolution.
    """
    def __init__(self, base_encoder_config: BeatGANsEncoderConfig,
                 tap_block_ids: Sequence[BlockTap], level_dims: Sequence[int],
                 pool: str = "mean_std", proj: str = "linear"):
        super().__init__()
        if len(tap_block_ids) != len(level_dims) or not tap_block_ids:
            raise ValueError("tap_block_ids and level_dims must have equal non-zero length")
        if proj != "linear":
            raise ValueError("only linear mean/std projection is supported")
        if pool not in {"mean_std", "adaptive_avg"}:
            raise ValueError("only mean_std pooling is supported")
        self.backbone = BeatGANsEncoderModel(base_encoder_config)
        self.tap_block_ids = list(tap_block_ids)
        channels = self._infer_tap_channels(base_encoder_config)
        self.heads = nn.ModuleList([MeanStdProjectionHead(channels[tap], d)
                                    for tap, d in zip(self.tap_block_ids, level_dims)])

    def _infer_tap_channels(self, conf: BeatGANsEncoderConfig):
        dummy = torch.zeros(1, conf.in_channels, conf.image_size, conf.image_size)
        h = dummy.type(self.backbone.dtype)
        channels = {}
        requested = set(self.tap_block_ids)
        with torch.no_grad():
            for i, module in enumerate(self.backbone.input_blocks):
                h = module(h, emb=None)
                if i in requested:
                    channels[i] = h.shape[1]
            h = self.backbone.middle_block(h, emb=None)
            if "mid" in requested:
                channels["mid"] = h.shape[1]
        missing = [tap for tap in self.tap_block_ids if tap not in channels]
        if missing:
            valid = list(range(len(self.backbone.input_blocks))) + ["mid"]
            raise ValueError(f"invalid tap block ids {missing}; valid taps: {valid}")
        return channels

    def forward(self, x) -> List[torch.Tensor]:
        h = x.type(self.backbone.dtype)
        taps = {}
        requested = set(self.tap_block_ids)
        for i, module in enumerate(self.backbone.input_blocks):
            h = module(h, emb=None)
            if i in requested:
                taps[i] = h.type(x.dtype)
        h = self.backbone.middle_block(h, emb=None)
        if "mid" in requested:
            taps["mid"] = h.type(x.dtype)
        return [head(taps[tap]) for tap, head in zip(self.tap_block_ids, self.heads)]
