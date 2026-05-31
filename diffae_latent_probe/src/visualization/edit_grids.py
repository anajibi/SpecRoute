from __future__ import annotations

from pathlib import Path

import torch

from src.visualization.grids import save_labeled_grid


def save_attribute_edit_grid(
    images: list[torch.Tensor],
    col_labels: list[str],
    out_path: str | Path,
    image_size: int,
    row_labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    save_labeled_grid(
        images=images,
        col_labels=col_labels,
        out_path=out_path,
        image_size=image_size,
        row_labels=row_labels,
        title=title,
        nrow=len(col_labels),
    )

