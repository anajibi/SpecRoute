"""Attribute embeddings and per-decoder-block style projections."""
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import yaml
from torch import nn


class AttributeEmbedding(nn.Module):
    """Sum per-attribute embeddings. Inputs are integer indices 0=neg, 1=pos, 2=CFG-null.

    Binary-only (CelebA path). Left unchanged for backward compatibility with existing
    checkpoints/configs -- ``MixedAttributeEmbedding`` below is the continuous/categorical
    generalization used by conditioning_attrs sourced from a causal_graph's ``nodes:`` section
    (see ``load_cond_specs``).
    """

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


@dataclass
class AttrCondSpec:
    """One conditioning attribute's kind + kind-specific parameters for MixedAttributeEmbedding.

    Mirrors ``causal/scm.py``'s ``NodeSpec`` (same "kind" vocabulary) deliberately -- the model's
    conditioning attributes and the SCM's graph nodes are the same 4 attributes for MorphoMNIST,
    and duplicating separate range declarations for them in two places would let the two silently
    drift. ``load_cond_specs`` builds this directly from ``causal_graph_morpho.yaml``.
    """
    name: str
    kind: str  # "categorical" | "continuous"
    lo: Optional[float] = None
    hi: Optional[float] = None
    num_classes: Optional[int] = None

    def __post_init__(self):
        if self.kind == "categorical" and not self.num_classes:
            raise ValueError(f"categorical attr {self.name!r} needs num_classes")
        if self.kind == "continuous" and (self.lo is None or self.hi is None):
            raise ValueError(f"continuous attr {self.name!r} needs lo/hi")
        if self.kind not in ("categorical", "continuous"):
            raise ValueError(f"unsupported AttrCondSpec kind {self.kind!r} for {self.name!r} "
                             "(binary attrs use the legacy AttributeEmbedding path instead)")


def load_cond_specs(causal_graph_path: str, conditioning_attrs: Sequence[str]) -> list:
    """Build ``AttrCondSpec`` list, in ``conditioning_attrs`` order, from a causal_graph yaml's
    ``nodes:`` section -- single source of truth shared with the SCM (see docstring above)."""
    with open(causal_graph_path) as f:
        raw = yaml.safe_load(f)
    nodes = raw.get("nodes", {})
    specs = []
    for name in conditioning_attrs:
        n = nodes.get(name)
        if n is None:
            raise ValueError(f"{causal_graph_path}: no 'nodes.{name}' entry for conditioning attr {name!r}")
        kind = n["kind"]
        if kind == "categorical":
            # `range`, if present, means the raw stored value is NOT already the class index (e.g.
            # hue: sampled from 10 fixed bin centers in [0,1], not stored as 0..9) -- lo/hi let
            # the embedding lookup bin it correctly instead of truncating/rounding, which would
            # collapse most values into one or two classes. Nodes without `range` (e.g. digit,
            # raw value already 0..9) keep lo=hi=None, preserving the original round()-based path
            # exactly -- see MixedAttributeEmbedding._embed_one.
            lo, hi = (n["range"] if "range" in n else (None, None))
            specs.append(AttrCondSpec(name=name, kind="categorical", num_classes=int(n["num_classes"]),
                                      lo=float(lo) if lo is not None else None,
                                      hi=float(hi) if hi is not None else None))
        elif kind == "continuous":
            lo, hi = n["range"]
            specs.append(AttrCondSpec(name=name, kind="continuous", lo=float(lo), hi=float(hi)))
        else:
            raise ValueError(f"{causal_graph_path}: node {name!r} has kind={kind!r}, "
                             "only categorical/continuous are supported as model conditioning attrs")
    return specs


class MixedAttributeEmbedding(nn.Module):
    """Sum per-attribute embeddings for a mix of categorical and continuous attributes.

    - categorical (e.g. digit): a plain lookup table, ``nn.Embedding(num_classes, dim)``.
    - continuous (e.g. thickness/intensity/hue): min-max normalized to [-1, 1] via the spec's
      lo/hi, then a small 2-layer MLP -- there is no reserved "null" scalar value a continuous
      attribute could take (unlike binary's spare index 2), so CFG/attribute dropout is applied
      via an explicit boolean mask, not a magic input value (a mask substitutes each dropped
      attribute's embedding with a learned per-attribute null vector). This also fixes a latent
      bug in the binary path's inference-time null: writing a literal sentinel value into a
      tensor that's about to carry real continuous data would silently produce a wrong "null"
      pass (e.g. thickness=2.0 is a valid thickness) instead of erroring.
    """

    def __init__(self, specs: Sequence[AttrCondSpec], attr_embed_dim: int, attr_dropout_prob: float,
                 cfg_drop_prob: float = 0.0):
        super().__init__()
        self.specs = list(specs)
        self.attr_dropout_prob = float(attr_dropout_prob)
        self.cfg_drop_prob = float(cfg_drop_prob)
        if not 0 <= self.cfg_drop_prob < 1:
            raise ValueError("cfg_drop_prob must be in [0, 1)")
        embedders = []
        for spec in self.specs:
            if spec.kind == "categorical":
                embedders.append(nn.Embedding(spec.num_classes, attr_embed_dim))
            else:
                embedders.append(nn.Sequential(
                    nn.Linear(1, attr_embed_dim), nn.SiLU(), nn.Linear(attr_embed_dim, attr_embed_dim)))
        self.embedders = nn.ModuleList(embedders)
        self.null_vectors = nn.Parameter(torch.zeros(len(self.specs), attr_embed_dim))

    def _embed_one(self, i: int, spec: AttrCondSpec, col: torch.Tensor) -> torch.Tensor:
        if spec.kind == "categorical":
            if spec.lo is not None and spec.hi is not None:
                # raw value is a bin-center float (e.g. hue: 0.05..0.95), not already the class
                # index -- bin via lo/hi (same convention as causal/scm.py's
                # categorical_class_index) instead of round(), which would collapse nearly every
                # value into class 0 or 1.
                frac = (col - spec.lo) / (spec.hi - spec.lo)
                idx = (frac * spec.num_classes).long().clamp(0, spec.num_classes - 1)
            else:
                # raw value already IS the class index (e.g. digit: 0..9) -- unchanged behavior.
                idx = col.round().long().clamp(0, spec.num_classes - 1)
            return self.embedders[i](idx)
        norm = (2.0 * (col - spec.lo) / (spec.hi - spec.lo) - 1.0).clamp(-1.0, 1.0).unsqueeze(-1)
        return self.embedders[i](norm)

    def forward(self, y_val: torch.Tensor, null_mask: Optional[torch.Tensor] = None,
                apply_dropout: bool = True) -> torch.Tensor:
        y_val = y_val.float()
        batch_size, n_attr = y_val.shape
        mask = torch.zeros(batch_size, n_attr, dtype=torch.bool, device=y_val.device)
        if null_mask is not None:
            mask = mask | null_mask.to(device=y_val.device, dtype=torch.bool)
        if apply_dropout and self.training and self.cfg_drop_prob > 0:
            drop_sample = torch.rand(batch_size, 1, device=y_val.device) < self.cfg_drop_prob
            mask = mask | drop_sample.expand(-1, n_attr)
        if apply_dropout and self.training and self.attr_dropout_prob > 0:
            drop = torch.rand(batch_size, n_attr, device=y_val.device) < self.attr_dropout_prob
            mask = mask | drop
        total = torch.zeros(batch_size, self.null_vectors.shape[1], device=y_val.device,
                            dtype=self.null_vectors.dtype)
        for i, spec in enumerate(self.specs):
            emb = self._embed_one(i, spec, y_val[:, i])
            emb = torch.where(mask[:, i:i + 1], self.null_vectors[i].unsqueeze(0).expand_as(emb), emb)
            total = total + emb
        return total


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


class ConcatAttributeEmbedding(nn.Module):
    """Like ``MixedAttributeEmbedding``, but concatenates per-attribute embeddings instead of
    summing them (opt-in via ``conditioning.attr_fusion: concat_film`` -- ``MixedAttributeEmbedding``
    is untouched and remains the default, so every existing checkpoint/config keeps working
    unchanged).

    Summing gives every attribute the same shared 128-dim space to write into, with nothing
    stopping a strong-gradient attribute (digit, thickness -- large pixel-loss footprint) from
    overwriting dimensions a weak one (hue) was starting to use. Concatenating instead gives each
    attribute a fixed, protected slice of the combined vector -- ``attr_embed_dim`` is the *total*
    concatenated width, split evenly across attributes (e.g. 128 total / 4 attrs = 32 each), not
    the per-attribute width. Everything else (per-kind embedder choice, null-mask-based dropout)
    is identical to ``MixedAttributeEmbedding``.
    """

    def __init__(self, specs: Sequence[AttrCondSpec], attr_embed_dim: int, attr_dropout_prob: float,
                 cfg_drop_prob: float = 0.0):
        super().__init__()
        self.specs = list(specs)
        n = len(self.specs)
        if attr_embed_dim % n != 0:
            raise ValueError(f"attr_embed_dim ({attr_embed_dim}) must be divisible by n_attributes "
                             f"({n}) for concat fusion -- it's split evenly per attribute")
        self.per_attr_dim = attr_embed_dim // n
        self.attr_dropout_prob = float(attr_dropout_prob)
        self.cfg_drop_prob = float(cfg_drop_prob)
        if not 0 <= self.cfg_drop_prob < 1:
            raise ValueError("cfg_drop_prob must be in [0, 1)")
        embedders = []
        for spec in self.specs:
            if spec.kind == "categorical":
                embedders.append(nn.Embedding(spec.num_classes, self.per_attr_dim))
            else:
                embedders.append(nn.Sequential(
                    nn.Linear(1, self.per_attr_dim), nn.SiLU(), nn.Linear(self.per_attr_dim, self.per_attr_dim)))
        self.embedders = nn.ModuleList(embedders)
        self.null_vectors = nn.Parameter(torch.zeros(n, self.per_attr_dim))

    def _embed_one(self, i: int, spec: AttrCondSpec, col: torch.Tensor) -> torch.Tensor:
        if spec.kind == "categorical":
            if spec.lo is not None and spec.hi is not None:
                frac = (col - spec.lo) / (spec.hi - spec.lo)
                idx = (frac * spec.num_classes).long().clamp(0, spec.num_classes - 1)
            else:
                idx = col.round().long().clamp(0, spec.num_classes - 1)
            return self.embedders[i](idx)
        norm = (2.0 * (col - spec.lo) / (spec.hi - spec.lo) - 1.0).clamp(-1.0, 1.0).unsqueeze(-1)
        return self.embedders[i](norm)

    def forward(self, y_val: torch.Tensor, null_mask: Optional[torch.Tensor] = None,
                apply_dropout: bool = True) -> torch.Tensor:
        y_val = y_val.float()
        batch_size, n_attr = y_val.shape
        mask = torch.zeros(batch_size, n_attr, dtype=torch.bool, device=y_val.device)
        if null_mask is not None:
            mask = mask | null_mask.to(device=y_val.device, dtype=torch.bool)
        if apply_dropout and self.training and self.cfg_drop_prob > 0:
            drop_sample = torch.rand(batch_size, 1, device=y_val.device) < self.cfg_drop_prob
            mask = mask | drop_sample.expand(-1, n_attr)
        if apply_dropout and self.training and self.attr_dropout_prob > 0:
            drop = torch.rand(batch_size, n_attr, device=y_val.device) < self.attr_dropout_prob
            mask = mask | drop
        parts = []
        for i, spec in enumerate(self.specs):
            emb = self._embed_one(i, spec, y_val[:, i])
            emb = torch.where(mask[:, i:i + 1], self.null_vectors[i].unsqueeze(0).expand_as(emb), emb)
            parts.append(emb)
        return torch.cat(parts, dim=-1)


class PerBlockStyleFiLM(nn.Module):
    """FiLM-style merge of the (now concatenated) attribute embedding into the per-block style
    vector, instead of ``PerBlockStyle``'s concat-then-one-Linear.

    ``PerBlockStyle`` concatenates ``zs`` (up to 512-dim for k=1) with ``attr_emb`` (128-dim)
    before a single shared Linear -- ``zs`` structurally outnumbers ``attr_emb`` there, so the
    attribute signal is diluted before the projection even has a chance to weight it. FiLM instead
    gives the attribute embedding a dedicated multiplicative+additive channel of influence on a
    zs-derived style vector: ``style = z_style * (1 + scale(attr_emb)) + shift(attr_emb)``.

    ``scale``/``shift`` projections are zero-initialized (weight and bias), so at the start of
    training ``style == z_style`` exactly -- the attribute pathway starts as a true no-op and has
    to earn its influence through gradient, rather than injecting random-init noise into the style
    vector from step 0 (standard FiLM/AdaIN initialization practice).
    """

    def __init__(self, level_dims: Sequence[int], block_to_level: Sequence[int],
                 attr_embed_dim: int, embed_channels: int):
        super().__init__()
        self.block_to_level = list(block_to_level)
        self.z_proj = nn.ModuleList([nn.Linear(level_dims[level], embed_channels) for level in self.block_to_level])
        self.scale_proj = nn.ModuleList([nn.Linear(attr_embed_dim, embed_channels) for _ in self.block_to_level])
        self.shift_proj = nn.ModuleList([nn.Linear(attr_embed_dim, embed_channels) for _ in self.block_to_level])
        for lin in list(self.scale_proj) + list(self.shift_proj):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, zs, attr_emb: torch.Tensor):
        styles = []
        for level, z_lin, scale_lin, shift_lin in zip(self.block_to_level, self.z_proj, self.scale_proj, self.shift_proj):
            z_style = z_lin(zs[level])
            scale = scale_lin(attr_emb)
            shift = shift_lin(attr_emb)
            styles.append(z_style * (1.0 + scale) + shift)
        return styles
