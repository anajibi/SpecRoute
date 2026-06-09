"""Configuration, checkpoint, and experiment construction utilities."""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_output(cfg: dict[str, Any]) -> Path:
    path = Path(__file__).parents[1] / cfg["train"]["output_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a module state dict to CPU before serialization.

    Copying explicitly makes the potentially slow GPU synchronization happen before
    checkpoint writing, so callers can report each phase to the user.
    """
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def atomic_torch_save(payload: object, path: str | Path) -> Path:
    """Write a torch payload atomically, leaving no partial final checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    return path


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def levels(output: dict[str, torch.Tensor], k: int) -> list[torch.Tensor]:
    return [output[f"Z{i}"] for i in range(k, 0, -1)]


def build_trainable(k: int) -> tuple[torch.nn.Module, ...]:
    from .decoders import DeterministicLatentDecoder, LatentDecoderDiffusion32x32
    from .encoders import HierarchicalEncoderK3, HierarchicalEncoderK5
    from .evidence import EvidencePyramid

    encoder = HierarchicalEncoderK3() if k == 3 else HierarchicalEncoderK5()
    return EvidencePyramid(), encoder, DeterministicLatentDecoder(k), LatentDecoderDiffusion32x32(k)
