from __future__ import annotations

import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
pytestmark = pytest.mark.skipif(not (_HAS_TORCH and _HAS_TORCHVISION), reason="torch and torchvision are required")

if _HAS_TORCH and _HAS_TORCHVISION:
    import torch

    from experiments.no_z_classifier_guided_diffusion.src.pretrained_attribute_classifier import (
        SelectedAttributeClassifier,
    )

    class ShapeRecordingBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_shape: tuple[int, ...] | None = None

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            self.last_shape = tuple(images.shape)
            batch_size = images.shape[0]
            return torch.arange(batch_size * 3, dtype=images.dtype, device=images.device).reshape(batch_size, 3)


def test_selected_attribute_classifier_resizes_and_selects_single_logit() -> None:
    backbone = ShapeRecordingBackbone()
    classifier = SelectedAttributeClassifier(backbone, attribute_index=2, input_size=64)
    images = torch.zeros(2, 3, 256, 256)

    logits = classifier(images)

    assert backbone.last_shape == (2, 3, 64, 64)
    assert logits.shape == (2, 1)
    assert torch.equal(logits.squeeze(1), torch.tensor([2.0, 5.0]))
