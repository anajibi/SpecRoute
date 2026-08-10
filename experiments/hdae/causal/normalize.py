"""Per-node-kind normalization for the SCM (causal/scm.py).

Three node kinds, three normalizations into/out of the flow's continuous
working space:

- ``binary`` (CelebA attributes): {0,1} -> logit of smoothed probability.
  ``eps`` (from a node's ``logit_smoothing_eps``) smooths {0,1} towards
  (eps, 1-eps) before the logit transform so neither endpoint maps to
  +/-inf. Resolves TODO item 2's "binary attributes" open decision.
- ``continuous`` (e.g. MorphoMNIST's thickness/intensity/hue): min-max to
  [-1, 1] -- genuinely continuous data doesn't need the logit transform,
  just a bounded-to-bounded affine map. ``lo``/``hi`` are declared per-node
  in the causal-graph config, not inferred, so a node's range is a visible
  modeling choice like ``eps`` is for binary nodes.
- ``categorical`` (e.g. MorphoMNIST's digit): no continuous transform at
  all -- handled directly as class indices in ``causal/scm.py``.
"""
import torch


def to_continuous(y01: torch.Tensor, eps: float) -> torch.Tensor:
    """{0,1} -> logit of smoothed probability."""
    p = y01 * (1 - 2 * eps) + eps
    return torch.log(p / (1 - p))


def to_prob(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def to_binary(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (prob >= threshold).float()


def minmax_to_continuous(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """[lo, hi] -> [-1, 1]."""
    return (x - lo) / (hi - lo) * 2.0 - 1.0


def minmax_to_raw(z: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """[-1, 1] -> [lo, hi]."""
    return (z + 1.0) / 2.0 * (hi - lo) + lo
