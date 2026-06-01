from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from .ddim_inversion import _to_unet_dtype, predict_x0_from_noise
except ImportError:  # pragma: no cover
    from ddim_inversion import _to_unet_dtype, predict_x0_from_noise


def _normalize_gradient(gradient: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    reduce_dims = tuple(range(1, gradient.ndim))
    norm = gradient.square().mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(eps)
    return gradient / norm


def _straight_through_clamp(tensor: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Clamp in the forward pass while preserving identity gradients.

    Hard ``torch.clamp`` is a common reason classifier guidance silently becomes
    a no-op: if the DDIM predicted-x0 image is outside the display range, clamp's
    backward pass returns zero and the classifier loss cannot push the sample.
    The straight-through variant gives the classifier a valid image-valued
    forward input but still propagates a useful gradient to the diffusion sample.
    """
    clamped = tensor.clamp(min_value, max_value)
    return tensor + (clamped - tensor).detach()


def _ensure_finite(tensor: torch.Tensor, name: str, *, step_index: int, timestep: torch.Tensor | int) -> None:
    if torch.isfinite(tensor).all():
        return
    if isinstance(timestep, torch.Tensor):
        timestep_value = int(timestep.detach().cpu().item())
    else:
        timestep_value = int(timestep)
    finite_fraction = float(torch.isfinite(tensor).float().mean().detach().cpu().item())
    raise FloatingPointError(
        f"Non-finite {name} during classifier guidance at "
        f"step_index={step_index}, timestep={timestep_value}, finite_fraction={finite_fraction:.6f}"
    )


def _append_guidance_diagnostic(
    diagnostics: list[dict[str, float]] | None,
    *,
    step_index: int,
    timestep: torch.Tensor | int,
    inner_step_index: int,
    loss: torch.Tensor,
    target_prob: torch.Tensor,
    classifier_input_unclamped: torch.Tensor,
    gradient: torch.Tensor,
    sample_before: torch.Tensor,
    sample_after: torch.Tensor,
) -> None:
    if diagnostics is None:
        return
    with torch.no_grad():
        reduce_dims = tuple(range(1, gradient.ndim))
        grad_rms = gradient.square().mean(dim=reduce_dims).sqrt().mean().item()
        grad_abs_mean = gradient.abs().mean(dim=reduce_dims).mean().item()
        grad_abs_max = gradient.abs().amax(dim=reduce_dims).mean().item()
        update_rms = (sample_after - sample_before).square().mean(dim=reduce_dims).sqrt().mean().item()
        outside_range = ((classifier_input_unclamped < -1) | (classifier_input_unclamped > 1)).float().mean().item()
        if isinstance(timestep, torch.Tensor):
            timestep_value = int(timestep.detach().cpu().item())
        else:
            timestep_value = int(timestep)
        diagnostics.append(
            {
                "step_index": int(step_index),
                "timestep": timestep_value,
                "inner_step_index": int(inner_step_index),
                "loss": float(loss.detach().float().cpu().item()),
                "target_prob": float(target_prob.detach().float().mean().cpu().item()),
                "gradient_rms": float(grad_rms),
                "gradient_abs_mean": float(grad_abs_mean),
                "gradient_abs_max": float(grad_abs_max),
                "update_rms": float(update_rms),
                "x0_pred_min": float(classifier_input_unclamped.detach().float().amin().cpu().item()),
                "x0_pred_max": float(classifier_input_unclamped.detach().float().amax().cpu().item()),
                "x0_pred_outside_range_fraction": float(outside_range),
            }
        )


def classifier_guided_ddim_sample(
    unet,
    scheduler,
    classifier,
    x_T: torch.Tensor,
    target_attribute_index: int,
    target_value: int | float,
    guidance_scale: float,
    num_guidance_steps_per_timestep: int,
    guidance_start_step: int,
    guidance_end_step: int,
    device: torch.device,
    clamp_x0: bool = True,
    guidance_on_x0_pred: bool = True,
    use_amp: bool = True,
    num_inference_steps: int = 50,
    guidance_step_size: float = 1.0,
    max_guided_sample_abs: float | None = None,
    diagnostics: list[dict[str, float]] | None = None,
) -> torch.Tensor:
    scheduler.set_timesteps(num_inference_steps, device=device)
    sample = _to_unet_dtype(x_T.detach().to(device), unet)
    classifier.eval()
    unet.eval()
    target = torch.full((sample.shape[0],), float(target_value), device=device, dtype=torch.float32)
    amp_enabled = use_amp and device.type == "cuda"
    compute_dtype = sample.dtype

    for step_index, timestep in enumerate(scheduler.timesteps):
        do_guidance = (
            guidance_scale > 0
            and guidance_start_step <= step_index < guidance_end_step
            and num_guidance_steps_per_timestep > 0
        )
        if do_guidance:
            for inner_step_index in range(num_guidance_steps_per_timestep):
                sample = sample.detach().clone().requires_grad_(True)
                autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)
                with torch.enable_grad(), autocast_ctx:
                    noise_pred = unet(sample, timestep).sample
                    if guidance_on_x0_pred:
                        classifier_input_unclamped = predict_x0_from_noise(sample, timestep, noise_pred, scheduler)
                    else:
                        classifier_input_unclamped = sample
                    classifier_input = classifier_input_unclamped
                    if clamp_x0:
                        classifier_input = _straight_through_clamp(classifier_input_unclamped, -1, 1)
                    logits = classifier(classifier_input.float())
                    logit = logits[:, target_attribute_index].float()
                    target_prob = torch.sigmoid(logit)
                    loss = F.binary_cross_entropy_with_logits(logit, target)
                _ensure_finite(loss, "classifier guidance loss", step_index=step_index, timestep=timestep)
                gradient = torch.autograd.grad(loss, sample, retain_graph=False, create_graph=False)[0]
                _ensure_finite(gradient, "classifier guidance gradient", step_index=step_index, timestep=timestep)
                gradient = _normalize_gradient(gradient)
                _ensure_finite(
                    gradient,
                    "normalized classifier guidance gradient",
                    step_index=step_index,
                    timestep=timestep,
                )
                sample_before = sample.detach()
                guidance_update = float(guidance_scale) * float(guidance_step_size) * gradient
                sample_after = sample - guidance_update
                if max_guided_sample_abs is not None:
                    sample_after = sample_after.clamp(-float(max_guided_sample_abs), float(max_guided_sample_abs))
                _ensure_finite(sample_after, "guided sample", step_index=step_index, timestep=timestep)
                _append_guidance_diagnostic(
                    diagnostics,
                    step_index=step_index,
                    timestep=timestep,
                    inner_step_index=inner_step_index,
                    loss=loss,
                    target_prob=target_prob,
                    classifier_input_unclamped=classifier_input_unclamped,
                    gradient=gradient,
                    sample_before=sample_before,
                    sample_after=sample_after.detach(),
                )
                sample = _to_unet_dtype(sample_after.detach().to(dtype=compute_dtype), unet)
                del gradient, guidance_update, loss, logits, logit, noise_pred, classifier_input
                del classifier_input_unclamped, target_prob

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            noise_pred = unet(sample, timestep).sample
            sample = _to_unet_dtype(scheduler.step(noise_pred, timestep, sample).prev_sample.detach(), unet)
            _ensure_finite(sample, "DDIM scheduler sample", step_index=step_index, timestep=timestep)
    return sample.detach().clone().clamp(-1, 1)
