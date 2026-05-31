from __future__ import annotations

from pathlib import Path

import torch

from diffae_tools.model_loader import DiffAEModelWrapper


class DiffAEWrapper:
    def __init__(self, repo_root: str | Path, checkpoint_path: str | Path, device: str):
        self.wrapper = DiffAEModelWrapper(repo_root=repo_root, checkpoint_path=checkpoint_path, device=device).load()

    def encode_semantic(self, images: torch.Tensor) -> torch.Tensor:
        return self.wrapper.encode_semantic(images, return_cpu=False)

    def encode_stochastic(self, images: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        return self.wrapper.encode_stochastic(images, z_sem=z_sem, return_cpu=False)

    def decode(self, z_sem: torch.Tensor, x_t: torch.Tensor, ddim_steps: int | None = None) -> torch.Tensor:
        if ddim_steps is not None:
            return self.wrapper.reconstruct(z_sem=z_sem, stochastic=x_t, ddim_steps=ddim_steps, return_cpu=False)
        return self.wrapper.decode_from_latents(z_sem, x_t, return_cpu=False)

