"""Helpers for test-time HDAE learned-null-token ablations."""
from typing import Iterable, Sequence


def parse_null_levels(spec) -> Sequence[int]:
    """Parse ``"0,2"``/``[0, 2]``/``None`` into a sorted list of levels."""
    if spec is None or spec == "":
        return []
    if isinstance(spec, str):
        return sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    return sorted({int(level) for level in spec})


def encode_with_null_levels(model, x, null_levels: Iterable[int]):
    """Return ``{'cond', 'zs', 'null_mask'}`` with selected levels forced to null tokens."""
    if not hasattr(model, "encode_with_nulls"):
        raise TypeError("model does not expose encode_with_nulls(); expected HierarchicalAutoencModel")
    return model.encode_with_nulls(x, parse_null_levels(null_levels))


def reconstruct_batch_with_null_levels(module, x, null_levels: Iterable[int], T=None):
    """DDIM reconstruct a batch while forcing selected semantic levels to null tokens.

    ``module`` is the HDAE Lightning module. The EMA model is used to match the
    standard reconstruction/evaluation path.
    """
    encoded = encode_with_null_levels(module.ema_model, x, null_levels)
    cond = encoded["cond"]
    x_t = module.encode_stochastic(x, cond, T=T)
    recon = module.render(x_t, cond, T=T)
    return recon, encoded
