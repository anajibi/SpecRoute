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


class _UnavailableXPU:
    """Minimal Intel XPU compatibility shim for older CPU/CUDA-only torch builds.

    Some versions of Diffusers/Accelerate/PyTorch device helpers probe
    ``torch.xpu`` during import. Older PyTorch builds do not define that
    namespace at all, which raises ``AttributeError: module 'torch' has no
    attribute 'xpu'`` before the experiment can fall back to CPU/CUDA.
    """

    class Event:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def record(self, *args, **kwargs) -> None:
            return None

        def wait(self, *args, **kwargs) -> None:
            return None

        def synchronize(self) -> None:
            return None

    class Stream:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def wait_event(self, *args, **kwargs) -> None:
            return None

        def wait_stream(self, *args, **kwargs) -> None:
            return None

        def synchronize(self) -> None:
            return None

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def is_initialized() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def set_device(device: object) -> None:
        return None

    @staticmethod
    def device(device: object | None = None):
        return torch.device("cpu")

    @staticmethod
    def current_stream(device: object | None = None):
        return _UnavailableXPU.Stream()

    @staticmethod
    def set_stream(stream: object) -> None:
        return None

    @staticmethod
    def stream(stream: object):
        return stream

    @staticmethod
    def _set_stream_by_id(*args, **kwargs) -> None:
        return None

    @staticmethod
    def get_device_properties(device: object | None = None) -> None:
        return None

    @staticmethod
    def get_device_name(device: object | None = None) -> str:
        return "unavailable-xpu"

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        return None

    @staticmethod
    def synchronize(device: object | None = None) -> None:
        return None


def ensure_torch_xpu_compat() -> None:
    if not hasattr(torch, "xpu"):
        setattr(torch, "xpu", _UnavailableXPU())


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
    ensure_torch_xpu_compat()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    ensure_torch_xpu_compat()
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
