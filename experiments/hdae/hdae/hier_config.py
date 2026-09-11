"""Typed configuration for conditional HDAE."""
from dataclasses import dataclass, field
from typing import List, Optional, Union

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
    # Optional: path to a causal_graph_*.yaml. When set, conditioning_attrs' kind/range/
    # num_classes are read from its `nodes:` section (single source of truth shared with the SCM)
    # and `cond_specs` is populated by config_io.load_hdae_config -- this switches the model onto
    # MixedAttributeEmbedding (continuous+categorical) instead of the legacy binary-only
    # AttributeEmbedding. Left None/empty, behavior is unchanged (existing CelebA binary configs).
    causal_graph_path: Optional[str] = None
    cond_specs: List = field(default_factory=list, repr=False)
    # Fourier features for CONTINUOUS conditioning attributes. 0 = off (bare normalized value
    # into a Linear, the original behavior). >0 emits 2*n sin/cos features per component plus
    # the raw value, matching how upstream embeds the continuous diffusion timestep -- a bare
    # scalar through a Linear gives neighbouring values near-identical embeddings, so the
    # attribute-CFG delta is tiny and continuous attributes under-edit.
    fourier_freqs: int = 0
    fourier_max_freq: float = 1000.0
    # Per-attribute RMSNorm (learnable gain) on each attribute's embedding before fusion, so no
    # attribute dominates the fused vector by raw magnitude alone. False = off (original).
    attr_norm: bool = False
    # ---- Adversarial tap invariance (gradient reversal) --------------------------------
    # A linear probe showed every k=11 tap recovers `hue` at 97-100% accuracy (spread 0.0275)
    # -- eleven redundant copies of the source attribute that an intervention never touches,
    # so the edit has to fight them. k=1's single tap has 10x the width yet encodes hue LESS
    # (0.7968) and the unmodelled attributes 1.8x BETTER. These knobs train the taps to stop
    # duplicating what the attribute pathway already supplies.
    #
    # adv_tap_lambda = 0.0 leaves training completely unchanged (no heads built, no loss term).
    adv_tap_lambda: float = 0.0
    # Fraction of max_steps over which lambda ramps 0 -> adv_tap_lambda. The encoder should
    # learn to reconstruct before it is penalised for carrying attributes; a full-strength
    # adversary from step 0 collapses the latent. Standard DANN practice.
    adv_tap_warmup_frac: float = 0.3
    # Hidden width of each per-tap adversary MLP. The head is deliberately STRONGER than the
    # linear probe used for diagnosis: beating a linear probe only proves the attribute was
    # made linearly inaccessible, not that it is gone.
    adv_tap_hidden: int = 128
    adv_tap_layers: int = 3

    @property
    def level_dims(self):
        return self.hier_level_dims


@dataclass
class ConditioningConfig:
    strategy: str = "per_block_attr"
    style_ch: int = 512
    latent_drop_prob: float = 0.0
    cfg_drop_prob: float = 0.1
    cfg_guidance_scale: float = 2.0
    # "sum" (default): MixedAttributeEmbedding sums per-attribute embeddings, PerBlockStyle
    # concatenates the sum with zs before one shared Linear -- the original, still-default
    # behavior every existing checkpoint/config was trained with, unchanged.
    # "concat_film": ConcatAttributeEmbedding (protected per-attribute slice, not summed) +
    # PerBlockStyleFiLM (attribute embedding FiLM-modulates a zs-derived style vector instead of
    # being concatenated with it). Opt-in only -- see attr_conditioner.py's class docstrings.
    attr_fusion: str = "sum"

    def __post_init__(self):
        if not 0 <= float(self.cfg_drop_prob) < 1:
            raise ValueError("cfg_drop_prob must be in [0, 1)")
        if float(self.cfg_guidance_scale) < 1:
            raise ValueError("cfg_guidance_scale must be >= 1")
        if self.attr_fusion not in ("sum", "concat_film"):
            raise ValueError(f"attr_fusion must be 'sum' or 'concat_film', got {self.attr_fusion!r}")


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
