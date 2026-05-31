from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _make_grid_canvas(
    grid_tensor: torch.Tensor,
    col_labels: list[str],
    image_size: int,
    padding: int,
    row_labels: list[str] | None = None,
    title: str | None = None,
) -> Image.Image:
    grid_pil = Image.fromarray((grid_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    font = _get_font(22)
    header_height = 60
    left_margin = 220 if row_labels else 20
    top_margin = header_height + (40 if title else 0)

    canvas = Image.new(
        "RGB",
        (grid_pil.width + left_margin, grid_pil.height + top_margin),
        (255, 255, 255),
    )
    canvas.paste(grid_pil, (left_margin, top_margin))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((left_margin + grid_pil.width // 2, 20), title, fill=(0, 0, 0), font=font, anchor="mm")

    for i, label in enumerate(col_labels):
        x_center = left_margin + padding + (image_size + padding) * i + image_size // 2
        draw.text((x_center, top_margin - 30), label, fill=(0, 0, 0), font=font, anchor="mm", align="center")

    if row_labels:
        for i, label in enumerate(row_labels):
            y_center = top_margin + padding + (image_size + padding) * i + image_size // 2
            draw.text((left_margin // 2, y_center), label, fill=(0, 0, 0), font=font, anchor="mm", align="center")

    return canvas


def save_labeled_grid(
    images: list[torch.Tensor],
    col_labels: list[str],
    out_path: str | Path,
    image_size: int,
    padding: int = 4,
    row_labels: list[str] | None = None,
    title: str | None = None,
    nrow: int | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = [img.detach().cpu().float().clamp(0.0, 1.0) for img in images]
    nrow = nrow or len(col_labels)
    grid_tensor = make_grid(torch.stack(tensors), nrow=nrow, padding=padding, normalize=False, value_range=(0, 1))
    canvas = _make_grid_canvas(
        grid_tensor,
        col_labels=col_labels,
        image_size=image_size,
        padding=padding,
        row_labels=row_labels,
        title=title,
    )
    canvas.save(out_path)

