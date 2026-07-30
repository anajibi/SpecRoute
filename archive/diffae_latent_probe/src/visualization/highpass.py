from __future__ import annotations

import torch


def vis_residual(x: torch.Tensor, scale: float = 4.0) -> torch.Tensor:
    return (0.5 + scale * x).clamp(0.0, 1.0)

