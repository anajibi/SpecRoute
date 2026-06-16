from pathlib import Path

import torch
from torch import nn

from dino_vae_hierarchical_diffusion.src.utils import atomic_torch_save, cpu_state_dict


def test_cpu_state_dict_is_detached_and_on_cpu():
    module = nn.Linear(2, 3)
    state = cpu_state_dict(module)
    assert state
    assert all(tensor.device.type == "cpu" for tensor in state.values())
    assert all(not tensor.requires_grad for tensor in state.values())


def test_atomic_torch_save_replaces_target_without_temporary_file(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"step": 1}, path)
    atomic_torch_save({"step": 2}, path)
    assert torch.load(path, weights_only=False) == {"step": 2}
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()
