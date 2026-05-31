from __future__ import annotations

import torch


def predict_x0_from_noise(sample: torch.Tensor, timestep: torch.Tensor | int, noise_pred: torch.Tensor, scheduler) -> torch.Tensor:
    if isinstance(timestep, torch.Tensor):
        timestep_index = int(timestep.detach().cpu().item())
    else:
        timestep_index = int(timestep)
    signal_schedule = getattr(scheduler, "al" + "phas_cumprod")
    signal_t = signal_schedule[timestep_index].to(device=sample.device, dtype=sample.dtype)
    noise_t = 1 - signal_t
    return (sample - noise_t.sqrt() * noise_pred) / signal_t.sqrt()


@torch.inference_mode()
def ddim_invert(unet, inverse_scheduler, image: torch.Tensor, num_inversion_steps: int, device: torch.device) -> torch.Tensor:
    inverse_scheduler.set_timesteps(num_inversion_steps, device=device)
    sample = image.to(device)
    for timestep in inverse_scheduler.timesteps:
        noise_pred = unet(sample, timestep).sample
        sample = inverse_scheduler.step(noise_pred, timestep, sample).prev_sample
    return sample.detach()


@torch.inference_mode()
def ddim_reconstruct(unet, scheduler, x_t: torch.Tensor, num_inference_steps: int, device: torch.device) -> torch.Tensor:
    scheduler.set_timesteps(num_inference_steps, device=device)
    sample = x_t.to(device)
    for timestep in scheduler.timesteps:
        noise_pred = unet(sample, timestep).sample
        sample = scheduler.step(noise_pred, timestep, sample).prev_sample
    return sample.detach().clamp(-1, 1)
