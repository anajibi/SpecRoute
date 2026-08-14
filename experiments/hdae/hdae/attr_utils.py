"""Attribute label range detection shared by conditional HDAE training/eval."""
from typing import Sequence

import torch

_ALLOWED = {"auto", "pm1", "01"}


def observed_unique(y_raw, max_values=16):
    vals = torch.unique(y_raw.detach().cpu())
    if vals.numel() > max_values:
        vals = vals[:max_values]
    return [float(v) for v in vals.tolist()]


def to_index_space(y_raw: torch.Tensor, attr_input_range: str = "auto") -> torch.Tensor:
    """Convert raw binary attributes to {0, 1} index space.

    ``attr_input_range`` may be ``auto``, ``pm1`` for {-1, 1}, or ``01`` for
    {0, 1}. Any other values, floats, or NaNs raise with the observed uniques.
    """
    if attr_input_range not in _ALLOWED:
        raise ValueError(f"attr_input_range must be one of {_ALLOWED}, got {attr_input_range!r}")
    if torch.isnan(y_raw.float()).any():
        raise ValueError(f"attribute tensor contains NaN; observed={observed_unique(y_raw)}")
    y = y_raw.detach() if not y_raw.requires_grad else y_raw
    vals = set(float(v) for v in torch.unique(y.detach().cpu()).tolist())
    if attr_input_range == "auto":
        if vals.issubset({-1.0, 1.0}):
            mode = "pm1"
        elif vals.issubset({0.0, 1.0}):
            mode = "01"
        else:
            raise ValueError(f"cannot infer attribute input range from observed values {sorted(vals)}")
    else:
        mode = attr_input_range
    if mode == "pm1":
        if not vals.issubset({-1.0, 1.0}):
            raise ValueError(f"attr_input_range='pm1' but observed values {sorted(vals)}")
        return ((y_raw.long() + 1) // 2).long()
    if not vals.issubset({0.0, 1.0}):
        raise ValueError(f"attr_input_range='01' but observed values {sorted(vals)}")
    return y_raw.long()


def to_cond_values(y_raw: torch.Tensor, specs: Sequence) -> torch.Tensor:
    """Validate/clamp raw mixed-kind attribute columns for ``MixedAttributeEmbedding``.

    Unlike ``to_index_space`` (binary-only, converts to a {0,1} lookup index), this leaves values
    in raw units -- continuous columns are only range-clamped, categorical columns are clamped in
    raw units too (NOT converted to a class index here). Per-kind normalization (min-max,
    lo/hi -> class-index binning, embedding lookup) happens inside the embedder, which already
    needs each spec's kind/lo/hi/num_classes -- this function must not duplicate that logic with
    a different convention, or it silently double-converts.

    Real bug fixed here (2026-08-14): this used to do ``col.round().clamp(0, num_classes - 1)``
    unconditionally for every categorical column, which is only correct for a spec with no
    declared lo/hi (e.g. ``digit``: raw storage already IS the class index 0..9, so round() is a
    no-op). For a spec WITH lo/hi (e.g. ``hue``: raw storage is a bin-center float, 0.05..0.95),
    that same round() collapses all ``num_classes`` classes down to just {0, 1} -- and because this
    function feeds every training batch (``lit_module.py``'s ``_batch_y_idx``), the model never
    saw more than 2 distinguishable hue values during training, regardless of how correctly the
    embedder's own lo/hi binning (``attr_conditioner.py``'s ``_embed_one``) was implemented
    downstream. Fix: for a spec with lo/hi, just clamp to that range and let the embedder do the
    real binning -- same branch condition as ``_embed_one`` and
    ``causal/scm.py``'s ``categorical_class_index``, but the action here is "pass through raw
    units", not "compute a class index" (computing one here and letting the embedder bin it again
    would double-convert).
    """
    if torch.isnan(y_raw.float()).any():
        raise ValueError(f"attribute tensor contains NaN; observed={observed_unique(y_raw)}")
    out = y_raw.clone().float()
    for i, spec in enumerate(specs):
        col = y_raw[:, i]
        if spec.kind == "categorical":
            if spec.lo is not None and spec.hi is not None:
                out[:, i] = col.clamp(spec.lo, spec.hi)
            else:
                out[:, i] = col.round().clamp(0, spec.num_classes - 1)
        else:
            out[:, i] = col.clamp(spec.lo, spec.hi)
    return out
