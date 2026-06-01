from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch is required for guidance tests")

if _HAS_TORCH:
    import torch

    from experiments.no_z_classifier_guided_diffusion.src.classifier_guidance import (
        _straight_through_clamp,
        classifier_guided_ddim_sample,
    )

    class ZeroNoiseUNet(torch.nn.Module):
        def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(sample=torch.zeros_like(sample))

    class IdentityScheduler:
        def __init__(self) -> None:
            self.alphas_cumprod = torch.ones(1)
            self.timesteps = torch.tensor([0])

        def set_timesteps(self, num_inference_steps: int, device: torch.device) -> None:
            self.timesteps = torch.tensor([0], device=device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)

        def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(prev_sample=sample)

    class MeanLogitClassifier(torch.nn.Module):
        def forward(self, sample: torch.Tensor) -> torch.Tensor:
            return sample.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)


def test_straight_through_clamp_keeps_gradients_outside_display_range() -> None:
    x = torch.tensor([2.0], requires_grad=True)
    y = _straight_through_clamp(x, -1, 1)

    y.backward()

    assert y.item() == 1.0
    assert x.grad is not None
    assert x.grad.item() == 1.0


def test_classifier_guidance_changes_saturated_x0_when_clamped() -> None:
    device = torch.device("cpu")
    x_t = torch.full((1, 1, 2, 2), 1.01)
    diagnostics: list[dict[str, float]] = []

    edited = classifier_guided_ddim_sample(
        unet=ZeroNoiseUNet(),
        scheduler=IdentityScheduler(),
        classifier=MeanLogitClassifier(),
        x_T=x_t,
        target_attribute_index=0,
        target_value=0,
        guidance_scale=0.1,
        num_guidance_steps_per_timestep=1,
        guidance_start_step=0,
        guidance_end_step=1,
        device=device,
        clamp_x0=True,
        guidance_on_x0_pred=True,
        use_amp=False,
        num_inference_steps=1,
        diagnostics=diagnostics,
    )

    assert torch.all(edited < x_t)
    assert diagnostics
    assert diagnostics[0]["gradient_rms"] > 0
    assert diagnostics[0]["update_rms"] > 0
    assert diagnostics[0]["x0_pred_outside_range_fraction"] == 1.0
