"""Binary attribute <-> continuous logit-space normalization for the SCM.

CelebA attributes are binary; normalizing flows need a continuous target.
``eps`` (from ``causal_graph.yaml``'s ``logit_smoothing_eps``) smooths {0,1}
towards (eps, 1-eps) before the logit transform so neither endpoint maps to
+/-inf. This is a real modeling choice (resolves TODO item 2's "binary
attributes" open decision via the logit-of-smoothed-probability approach) --
kept visible in config rather than hardcoded here.
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
