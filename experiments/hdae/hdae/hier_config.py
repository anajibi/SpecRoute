"""Typed configuration owned by the HDAE experiment."""
from dataclasses import dataclass, field
from typing import List, Tuple, Union

from model.unet_autoenc import BeatGANsAutoencConfig

BlockTap = Union[int, str]


def _default_block_to_level():
    # Default DiffAE decoder has 11 output blocks for the CelebA64 template.
    # K=1 preset: [0] * n_decoder_output_blocks.
    return [0] * 11


@dataclass
class EncoderHierarchyConfig:
    type: str = "hierarchical"
    # Legacy resolution-based knobs are kept for YAML compatibility, but new
    # conditional HDAE uses explicit block taps below.
    tap_resolutions: List[int] = field(default_factory=lambda: [16, 8, 4])
    level_dims: List[int] = field(default_factory=lambda: [256, 192, 64])
    pool: str = "adaptive_avg"
    proj: str = "linear"

    # New per-block-injected hierarchy controls. ``hier_tap_block_ids`` indexes
    # encoder input_blocks and may include the sentinel "mid".  Presets:
    # K=1: hier_block_to_level=[0]*n, K=5: contiguous decoder bands,
    # K=11: hier_block_to_level=list(range(n)).
    hier_tap_block_ids: List[BlockTap] = field(default_factory=list)
    hier_level_dims: List[int] = field(default_factory=list)
    hier_block_to_level: List[int] = field(default_factory=_default_block_to_level)
    n_decoder_output_blocks: int = 11
    n_attributes: int = 5
    attr_embed_dim: int = 128
    attr_dropout_prob: float = 0.1
    hier_proj: str = "linear"
    attr_input_range: str = "auto"
    lambda_indep: float = 0.0

    def __post_init__(self):
        if not self.hier_level_dims:
            self.hier_level_dims = list(self.level_dims)
        if not self.hier_tap_block_ids:
            # Backward-compatible fallback; config_io can also fill concrete
            # block IDs after it has the upstream encoder config.
            self.hier_tap_block_ids = ["mid"] if len(self.hier_level_dims) == 1 else list(range(len(self.hier_level_dims)))
        if len(self.hier_tap_block_ids) != len(self.hier_level_dims):
            raise ValueError("len(hier_tap_block_ids) must equal len(hier_level_dims)")
        if len(self.hier_block_to_level) != int(self.n_decoder_output_blocks):
            raise ValueError("len(hier_block_to_level) must equal n_decoder_output_blocks")
        if max(self.hier_block_to_level, default=-1) == 0 and len(self.hier_level_dims) > 1:
            # Backward-compatible direct dataclass construction: if callers only
            # supplied level_dims, synthesize contiguous decoder bands.
            k = len(self.hier_level_dims)
            n = int(self.n_decoder_output_blocks)
            self.hier_block_to_level = [min(k - 1, i * k // n) for i in range(n)]
        if max(self.hier_block_to_level, default=-1) != len(self.hier_level_dims) - 1:
            raise ValueError("max(hier_block_to_level) must be len(hier_level_dims)-1")
        if min(self.hier_block_to_level, default=0) < 0:
            raise ValueError("hier_block_to_level entries must be non-negative")
        if self.attr_input_range not in {"auto", "pm1", "01"}:
            raise ValueError("attr_input_range must be one of {'auto','pm1','01'}")
        if not 0 <= self.attr_dropout_prob < 1:
            raise ValueError("attr_dropout_prob must be in [0, 1)")
        if self.hier_proj != "linear":
            raise ValueError("hier_proj currently supports only 'linear'")


@dataclass
class ConditioningConfig:
    strategy: str = "per_block_attr"
    style_ch: int = 512
    latent_drop_prob: float = 0.12


@dataclass
class HDAEConfig:
    encoder: EncoderHierarchyConfig = field(default_factory=EncoderHierarchyConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)


@dataclass
class HierarchicalBeatGANsAutoencConfig(BeatGANsAutoencConfig):
    """Upstream model config carrying the experiment-only hierarchy config."""
    hdae_conf: HDAEConfig = None

    def make_model(self):
        from .hier_autoenc import HierarchicalAutoencModel
        if self.hdae_conf is None:
            raise ValueError("Hierarchical model requires hdae_conf")
        return HierarchicalAutoencModel(self, self.hdae_conf)
