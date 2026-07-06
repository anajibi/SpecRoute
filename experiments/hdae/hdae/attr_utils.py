"""Attribute label range detection shared by conditional HDAE training/eval."""
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
