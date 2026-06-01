from __future__ import annotations

import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch is required for image utility tests")

if _HAS_TORCH:
    import torch

    from experiments.no_z_classifier_guided_diffusion.src.utils import tensor_to_uint8_image


def test_tensor_to_uint8_image_rejects_non_finite_values() -> None:
    image = torch.zeros(3, 2, 2)
    image[0, 0, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        tensor_to_uint8_image(image)
