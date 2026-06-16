"""Typed configuration owned by the HDAE experiment."""
from dataclasses import dataclass, field
from typing import List

from model.unet_autoenc import BeatGANsAutoencConfig


@dataclass
class EncoderHierarchyConfig:
    type: str = "hierarchical"
    tap_resolutions: List[int] = field(default_factory=lambda: [16, 8, 4])
    level_dims: List[int] = field(default_factory=lambda: [256, 192, 64])
    pool: str = "adaptive_avg"
    proj: str = "linear"


@dataclass
class ConditioningConfig:
    strategy: str = "concat_proj"
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
