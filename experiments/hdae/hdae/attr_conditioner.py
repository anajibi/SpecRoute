"""Attribute embeddings and per-decoder-block style projections."""
from dataclasses import dataclass
from typing import Optional, Sequence

import math

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
    dim: int = 1  # >1 for vector-valued attributes (e.g. Causal3DIdent's pos_obj/rot_obj = 3)

    def __post_init__(self):
        self.dim = int(self.dim)
        if self.dim < 1:
            raise ValueError(f"attr {self.name!r} needs dim >= 1, got {self.dim}")
        if self.kind == "categorical" and self.dim != 1:
            raise ValueError(f"categorical attr {self.name!r} must be scalar (dim=1), got {self.dim}")
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
                                      hi=float(hi) if hi is not None else None,
                                      dim=int(n.get("dim", 1))))
        elif kind == "continuous":
            lo, hi = n["range"]
            specs.append(AttrCondSpec(name=name, kind="continuous", lo=float(lo), hi=float(hi),
                                      dim=int(n.get("dim", 1))))
        else:
            raise ValueError(f"{causal_graph_path}: node {name!r} has kind={kind!r}, "
                             "only categorical/continuous are supported as model conditioning attrs")
    return specs


def attr_column_offsets(specs) -> list:
    """Start column of each attribute inside a raw attribute row.

    With every spec at dim=1 this is just [0,1,2,...] and the row layout is unchanged from
    the scalar-only era; a vector attribute (dim>1) occupies `dim` consecutive columns.
    """
    offs, c = [], 0
    for sp in specs:
        offs.append(c)
        c += int(getattr(sp, "dim", 1))
    return offs


def attr_total_columns(specs) -> int:
    return sum(int(getattr(sp, "dim", 1)) for sp in specs)


class FourierFeatures(nn.Module):
    """Sinusoidal features for a continuous scalar, log-spaced like upstream's
    ``timestep_embedding`` (diffae_upstream/model/nn.py).

    A bare ``Linear(1, d)`` maps an attribute's whole range onto a smooth, essentially
    affine 1-D curve: neighbouring values get near-identical embeddings, so the CFG delta
    ``e(target) - e(null)`` is tiny and the attribute under-edits no matter how large the
    guidance scale. Categorical attributes never have this problem -- a lookup table places
    each class freely and far apart. That asymmetry is why continuous attributes score much
    worse on CC than categorical ones at the same guidance.

    Emits ``2 * n_freqs`` sin/cos features per component, plus the raw value passthrough.
    """

    def __init__(self, n_freqs: int = 16, max_freq: float = 1000.0, include_input: bool = True):
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.include_input = bool(include_input)
        freqs = torch.exp(torch.linspace(0.0, math.log(float(max_freq)), self.n_freqs))
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim_per_component(self) -> int:
        return 2 * self.n_freqs + (1 if self.include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C) normalized to [-1, 1] -> (B, C * out_dim_per_component)."""
        args = x.unsqueeze(-1) * self.freqs.to(x.dtype)          # (B, C, F)
        feats = [torch.sin(args), torch.cos(args)]
        if self.include_input:
            feats.insert(0, x.unsqueeze(-1))
        return torch.cat(feats, dim=-1).flatten(1)


class RMSNorm(nn.Module):
    """RMS normalization with a learnable per-channel gain (torch 2.2 has no nn.RMSNorm).

    Applied per attribute so no attribute dominates the fused conditioning vector by raw
    magnitude alone -- an embedding-table lookup and an MLP output have no reason to end up
    on the same scale, and under summation (or FiLM) the larger one simply wins.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight



def _build_embedders(specs, out_dim, fourier_freqs, fourier_max_freq):
    """Per-attribute embedder + (for continuous attrs) its Fourier front-end.

    categorical -> nn.Embedding(num_classes, out_dim)
    continuous  -> [FourierFeatures] -> MLP(-> out_dim).  With fourier_freqs=0 the input is
                   the bare normalized value(s), i.e. exactly the original Linear(dim, out_dim)
                   path, so existing configs/checkpoints are unaffected.
    """
    embedders, fouriers = [], []
    for spec in specs:
        if spec.kind == "categorical":
            embedders.append(nn.Embedding(spec.num_classes, out_dim))
            fouriers.append(None)
            continue
        d = int(getattr(spec, "dim", 1))
        if fourier_freqs and fourier_freqs > 0:
            ff = FourierFeatures(fourier_freqs, fourier_max_freq)
            in_dim = d * ff.out_dim_per_component
        else:
            ff, in_dim = None, d
        fouriers.append(ff)
        embedders.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(),
                                       nn.Linear(out_dim, out_dim)))
    return nn.ModuleList(embedders), nn.ModuleList([f if f is not None else nn.Identity()
                                                    for f in fouriers])


def _embed_attr(i, spec, cols, embedders, fouriers, fourier_on):
    """cols: (B, spec.dim) raw values for this attribute."""
    if spec.kind == "categorical":
        col = cols[:, 0]
        if spec.lo is not None and spec.hi is not None:
            frac = (col - spec.lo) / (spec.hi - spec.lo)
            idx = (frac * spec.num_classes).long().clamp(0, spec.num_classes - 1)
        else:
            idx = col.round().long().clamp(0, spec.num_classes - 1)
        return embedders[i](idx)
    norm = (2.0 * (cols - spec.lo) / (spec.hi - spec.lo) - 1.0).clamp(-1.0, 1.0)
    if fourier_on:
        norm = fouriers[i](norm)
    return embedders[i](norm)


def _apply_dropout_mask(self, y_val, null_mask):
    batch_size = y_val.shape[0]
    n_attr = len(self.specs)
    mask = torch.zeros(batch_size, n_attr, dtype=torch.bool, device=y_val.device)
    if null_mask is not None:
        nm = null_mask.to(device=y_val.device, dtype=torch.bool)
        if nm.shape[1] != n_attr:      # tolerate a per-column mask from older callers
            offs = attr_column_offsets(self.specs)
            nm = torch.stack([nm[:, o] for o in offs], dim=1)
        mask = mask | nm
    if self.training and self.cfg_drop_prob > 0:
        drop_sample = torch.rand(batch_size, 1, device=y_val.device) < self.cfg_drop_prob
        mask = mask | drop_sample.expand(-1, n_attr)
    if self.training and self.attr_dropout_prob > 0:
        mask = mask | (torch.rand(batch_size, n_attr, device=y_val.device) < self.attr_dropout_prob)
    return mask


class MixedAttributeEmbedding(nn.Module):
    """Sum per-attribute embeddings for a mix of categorical and continuous attributes.

    - categorical (e.g. digit, class): a lookup table, ``nn.Embedding(num_classes, dim)``.
    - continuous (e.g. thickness, pos_obj): min-max normalized to [-1, 1] via the spec's
      lo/hi, then optional Fourier features, then a small MLP.

    CFG/attribute dropout is applied via an explicit boolean mask (one column per
    *attribute*, not per raw column), substituting a learned per-attribute null vector --
    there is no reserved "null" scalar a continuous attribute could safely take.

    ``fourier_freqs > 0`` and ``attr_norm=True`` are both opt-in; with the defaults this
    class behaves exactly as before.
    """

    def __init__(self, specs: Sequence[AttrCondSpec], attr_embed_dim: int, attr_dropout_prob: float,
                 cfg_drop_prob: float = 0.0, fourier_freqs: int = 0,
                 fourier_max_freq: float = 1000.0, attr_norm: bool = False):
        super().__init__()
        self.specs = list(specs)
        self.attr_dropout_prob = float(attr_dropout_prob)
        self.cfg_drop_prob = float(cfg_drop_prob)
        if not 0 <= self.cfg_drop_prob < 1:
            raise ValueError("cfg_drop_prob must be in [0, 1)")
        self.fourier_freqs = int(fourier_freqs)
        self.attr_norm = bool(attr_norm)
        self.embedders, self.fouriers = _build_embedders(self.specs, attr_embed_dim,
                                                         self.fourier_freqs, fourier_max_freq)
        self.norms = nn.ModuleList([RMSNorm(attr_embed_dim) if self.attr_norm else nn.Identity()
                                    for _ in self.specs])
        self.null_vectors = nn.Parameter(torch.zeros(len(self.specs), attr_embed_dim))
        self.offsets = attr_column_offsets(self.specs)

    def forward(self, y_val: torch.Tensor, null_mask: Optional[torch.Tensor] = None,
                apply_dropout: bool = True) -> torch.Tensor:
        y_val = y_val.float()
        mask = _apply_dropout_mask(self, y_val, null_mask) if apply_dropout else \
            (torch.zeros(y_val.shape[0], len(self.specs), dtype=torch.bool, device=y_val.device)
             if null_mask is None else null_mask.to(y_val.device).bool())
        total = torch.zeros(y_val.shape[0], self.null_vectors.shape[1], device=y_val.device,
                            dtype=self.null_vectors.dtype)
        for i, spec in enumerate(self.specs):
            o, d = self.offsets[i], int(getattr(spec, "dim", 1))
            emb = _embed_attr(i, spec, y_val[:, o:o + d], self.embedders, self.fouriers,
                              self.fourier_freqs > 0)
            emb = self.norms[i](emb)
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
    summing them (``conditioning.attr_fusion: concat_film``).

    Each attribute gets a fixed, protected slice of the combined vector -- ``attr_embed_dim``
    is the *total* concatenated width, split evenly across attributes -- so a strong-gradient
    attribute cannot overwrite dimensions a weak one was using.

    ``fourier_freqs`` / ``attr_norm`` as in ``MixedAttributeEmbedding``; both default off.
    """

    def __init__(self, specs: Sequence[AttrCondSpec], attr_embed_dim: int, attr_dropout_prob: float,
                 cfg_drop_prob: float = 0.0, fourier_freqs: int = 0,
                 fourier_max_freq: float = 1000.0, attr_norm: bool = False):
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
        self.fourier_freqs = int(fourier_freqs)
        self.attr_norm = bool(attr_norm)
        self.embedders, self.fouriers = _build_embedders(self.specs, self.per_attr_dim,
                                                         self.fourier_freqs, fourier_max_freq)
        self.norms = nn.ModuleList([RMSNorm(self.per_attr_dim) if self.attr_norm else nn.Identity()
                                    for _ in self.specs])
        self.null_vectors = nn.Parameter(torch.zeros(n, self.per_attr_dim))
        self.offsets = attr_column_offsets(self.specs)

    def forward(self, y_val: torch.Tensor, null_mask: Optional[torch.Tensor] = None,
                apply_dropout: bool = True) -> torch.Tensor:
        y_val = y_val.float()
        mask = _apply_dropout_mask(self, y_val, null_mask) if apply_dropout else \
            (torch.zeros(y_val.shape[0], len(self.specs), dtype=torch.bool, device=y_val.device)
             if null_mask is None else null_mask.to(y_val.device).bool())
        parts = []
        for i, spec in enumerate(self.specs):
            o, d = self.offsets[i], int(getattr(spec, "dim", 1))
            emb = _embed_attr(i, spec, y_val[:, o:o + d], self.embedders, self.fouriers,
                              self.fourier_freqs > 0)
            emb = self.norms[i](emb)
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
