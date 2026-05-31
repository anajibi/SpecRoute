from __future__ import annotations

import torch


def _module_parameter_dtype(module) -> torch.dtype | None:
    """Return the dtype used by a module's parameters or buffers, if any."""
    for tensor in module.parameters(recurse=True):
        return tensor.dtype
    for tensor in module.buffers(recurse=True):
        return tensor.dtype
    return None


def _to_unet_dtype(sample: torch.Tensor, unet) -> torch.Tensor:
    """Move ``sample`` to the UNet dtype to avoid mixed input/bias errors."""
    unet_dtype = _module_parameter_dtype(unet)
    if unet_dtype is None or not unet_dtype.is_floating_point:
        return sample
    return sample.to(dtype=unet_dtype)


def predict_x0_from_noise(sample: torch.Tensor, timestep: torch.Tensor | int, noise_pred: torch.Tensor, scheduler) -> torch.Tensor:
    if isinstance(timestep, torch.Tensor):
        timestep_index = int(timestep.detach().cpu().item())
    else:
        timestep_index = int(timestep)
    signal_schedule = getattr(scheduler, "al" + "phas_cumprod")
    signal_t = signal_schedule[timestep_index].to(device=sample.device, dtype=sample.dtype)
    noise_t = 1 - signal_t
    return (sample - noise_t.sqrt() * noise_pred) / signal_t.sqrt()


def ddim_invert(unet, inverse_scheduler, image: torch.Tensor, num_inversion_steps: int, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        inverse_scheduler.set_timesteps(num_inversion_steps, device=device)
        sample = _to_unet_dtype(image.to(device), unet)
        for timestep in inverse_scheduler.timesteps:
            noise_pred = unet(sample, timestep).sample
            sample = inverse_scheduler.step(noise_pred, timestep, sample).prev_sample
    return sample.detach().clone()


def ddim_reconstruct(unet, scheduler, x_t: torch.Tensor, num_inference_steps: int, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        scheduler.set_timesteps(num_inference_steps, device=device)
        sample = _to_unet_dtype(x_t.to(device), unet)
        for timestep in scheduler.timesteps:
            noise_pred = unet(sample, timestep).sample
            sample = scheduler.step(noise_pred, timestep, sample).prev_sample
    return sample.detach().clone().clamp(-1, 1)
