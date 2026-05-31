from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from .ddim_inversion import predict_x0_from_noise
except ImportError:  # pragma: no cover
    from ddim_inversion import predict_x0_from_noise


def _normalize_gradient(gradient: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    reduce_dims = tuple(range(1, gradient.ndim))
    norm = gradient.square().mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(eps)
    return gradient / norm


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
) -> torch.Tensor:
    scheduler.set_timesteps(num_inference_steps, device=device)
    sample = x_T.detach().to(device)
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
            for _ in range(num_guidance_steps_per_timestep):
                sample = sample.detach().requires_grad_(True)
                with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    model_input = scheduler.scale_model_input(sample, timestep)
                    noise_pred = unet(model_input, timestep).sample
                    classifier_input = predict_x0_from_noise(sample, timestep, noise_pred, scheduler) if guidance_on_x0_pred else sample
                    if clamp_x0:
                        classifier_input = classifier_input.clamp(-1, 1)
                    logits = classifier(classifier_input.float())
                    logit = logits[:, target_attribute_index].float()
                    loss = F.binary_cross_entropy_with_logits(logit, target)
                gradient = torch.autograd.grad(loss, sample, retain_graph=False, create_graph=False)[0]
                gradient = _normalize_gradient(gradient)
                sample = (sample - float(guidance_scale) * gradient).detach().to(dtype=compute_dtype)
                del gradient, loss, logits, logit, noise_pred, classifier_input, model_input

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            model_input = scheduler.scale_model_input(sample, timestep)
            noise_pred = unet(model_input, timestep).sample
            sample = scheduler.step(noise_pred, timestep, sample).prev_sample.detach()
    return sample.detach().clamp(-1, 1)
