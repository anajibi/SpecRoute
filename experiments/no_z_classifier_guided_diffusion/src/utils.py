from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def add_project_paths() -> None:
    import sys

    root = project_root()
    for path in (root, root / "diffae_latent_probe"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def tensor_to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(-1, 1)
    image = (image + 1.0) / 2.0
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_uint8_image(tensor).save(path)
    return path


def image_file_stem(path: str | Path, fallback_index: int) -> str:
    stem = Path(path).stem
    return stem if stem else f"image_{fallback_index:06d}"


def maybe_autocast(device: torch.device, enabled: bool, dtype: torch.dtype = torch.float16):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled and device.type == "cuda")
