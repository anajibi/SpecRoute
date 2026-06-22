"""Utilities for saving image grids with row labels."""
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont


def _to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape {tuple(tensor.shape)}")
    arr = tensor.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def save_labeled_grid(rows: Sequence[torch.Tensor], row_labels: Sequence[str], path,
                      label_width: int = 180, pad: int = 2, bg=(255, 255, 255)):
    """Save a grid where each input row has a text label at the left.

    Args:
        rows: sequence of tensors shaped ``(N, C, H, W)`` in ``[0, 1]``.
        row_labels: one label per row.
        path: output image path.
    """
    if len(rows) != len(row_labels):
        raise ValueError("rows and row_labels must have the same length")
    if not rows:
        raise ValueError("at least one row is required")
    n, c, h, w = rows[0].shape
    for row in rows:
        if row.ndim != 4 or row.shape[0] != n or row.shape[1:] != (c, h, w):
            raise ValueError("all rows must have the same (N, C, H, W) shape")
    font = ImageFont.load_default()
    width = label_width + n * w + max(0, n - 1) * pad
    height = len(rows) * h + max(0, len(rows) - 1) * pad
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    for row_idx, (row, label) in enumerate(zip(rows, row_labels)):
        y = row_idx * (h + pad)
        draw.text((8, y + max(0, (h - 10) // 2)), str(label), fill=(0, 0, 0), font=font)
        for col_idx in range(n):
            x = label_width + col_idx * (w + pad)
            canvas.paste(_to_uint8_image(row[col_idx]), (x, y))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
