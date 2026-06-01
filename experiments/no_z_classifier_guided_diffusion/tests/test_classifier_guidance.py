from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch is required for guidance tests")

if _HAS_TORCH:
    import torch

    from experiments.no_z_classifier_guided_diffusion.src.classifier_guidance import (
        _normalize_gradient,
        _straight_through_clamp,
        classifier_guided_ddim_sample,
    )

    class ZeroNoiseUNet(torch.nn.Module):
        def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(sample=torch.zeros_like(sample))

    class IdentityScheduler:
        def __init__(self, alpha_cumprod: float = 1.0) -> None:
            self.alphas_cumprod = torch.tensor([alpha_cumprod])
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


def test_normalize_gradient_handles_large_fp16_values() -> None:
    gradient = torch.full((1, 1, 2, 2), 60000.0, dtype=torch.float16)

    normalized = _normalize_gradient(gradient)

    assert torch.isfinite(normalized).all()
    assert torch.all(normalized > 0)


def test_classifier_guidance_changes_saturated_x0_when_clamped() -> None:
    device = torch.device("cpu")
    x_t = torch.full((1, 1, 2, 2), 1.01)
    diagnostics: list[dict[str, float | str]] = []

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
        guidance_step_size=1.0,
        diagnostics=diagnostics,
    )

    assert torch.all(edited < x_t)
    assert diagnostics
    assert diagnostics[0]["gradient_rms"] > 0
    assert diagnostics[0]["update_rms"] > 0
    assert diagnostics[0]["x0_pred_outside_range_fraction"] == 1.0


def test_classifier_guidance_skips_low_alpha_timesteps() -> None:
    device = torch.device("cpu")
    x_t = torch.full((1, 1, 2, 2), 1.01)
    diagnostics: list[dict[str, float | str]] = []

    edited = classifier_guided_ddim_sample(
        unet=ZeroNoiseUNet(),
        scheduler=IdentityScheduler(alpha_cumprod=1.0e-5),
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
        min_guidance_alpha_cumprod=1.0e-3,
        diagnostics=diagnostics,
    )

    assert torch.allclose(edited, x_t.clamp(-1, 1))
    assert diagnostics
    assert diagnostics[0]["skip_reason"] == "alpha_cumprod_below_min_guidance_alpha_cumprod"
