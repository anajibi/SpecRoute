"""Helpers for consuming dictionary batches returned by image datasets."""
from __future__ import annotations

from typing import Any

import torch


def batch_images(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return batch["image"].to(device, non_blocking=True)


def batch_image_ids(batch: dict[str, Any]) -> list[str]:
    if "image_id" in batch:
        return [str(value) for value in batch["image_id"]]
    if "image_path" in batch:
        return [str(value) for value in batch["image_path"]]
    return [str(value) for value in batch["index"]]
