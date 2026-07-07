"""Typed configuration for conditional HDAE."""
from dataclasses import dataclass, field
from typing import List, Union

from model.unet_autoenc import BeatGANsAutoencConfig

BlockTap = Union[int, str]


@dataclass
class EncoderHierarchyConfig:
    type: str = "hierarchical"
    hier_tap_block_ids: List[BlockTap] = field(default_factory=lambda: [0, 2, 4, 6, "mid"])
    hier_level_dims: List[int] = field(default_factory=lambda: [103, 103, 102, 102, 102])
    hier_block_to_level: List[int] = field(default_factory=lambda: [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    n_decoder_output_blocks: int = 11
    n_attributes: int = 5
    conditioning_attrs: List[str] = field(default_factory=lambda: ["Smiling", "Eyeglasses", "Male", "Young", "Wearing_Lipstick"])
    attr_embed_dim: int = 128
    attr_dropout_prob: float = 0.1
    attr_input_range: str = "pm1"
    hier_proj: str = "linear"
    lambda_indep: float = 0.0

    @property
    def level_dims(self):
        return self.hier_level_dims


@dataclass
class ConditioningConfig:
    strategy: str = "per_block_attr"
    style_ch: int = 512
    latent_drop_prob: float = 0.0


@dataclass
class HDAEConfig:
    encoder: EncoderHierarchyConfig = field(default_factory=EncoderHierarchyConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)


@dataclass
class HierarchicalBeatGANsAutoencConfig(BeatGANsAutoencConfig):
    hdae_conf: HDAEConfig = None

    def make_model(self):
        from .hier_autoenc import HierarchicalAutoencModel
        return HierarchicalAutoencModel(self, self.hdae_conf)
