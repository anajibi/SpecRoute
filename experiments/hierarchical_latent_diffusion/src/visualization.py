"""Visualization helpers for reconstructed and intervened image grids."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


def prepare_grid_columns(columns: Sequence[torch.Tensor]) -> torch.Tensor:
    """Interleave batched image columns so each input occupies one grid row."""
    if not columns:
        raise ValueError("At least one image column is required")
    batch_size = columns[0].shape[0]
    if any(column.shape != columns[0].shape for column in columns):
        raise ValueError("All image columns must have the same shape")
    return torch.stack(list(columns), dim=1).reshape(batch_size * len(columns), *columns[0].shape[1:])


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    from torchvision.utils import save_image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.clamp(-1, 1).add(1).div(2), path, nrow=nrow)


def save_labeled_grid(
    columns: Sequence[torch.Tensor],
    labels: Sequence[str],
    path: str | Path,
    row_labels: Sequence[str] | None = None,
) -> None:
    """Save a publication-friendly grid with one image batch per labeled column."""
    if len(columns) != len(labels):
        raise ValueError("There must be exactly one label per image column")
    if not columns:
        raise ValueError("At least one image column is required")

    import matplotlib.pyplot as plt

    batch_size = columns[0].shape[0]
    if any(column.shape[0] != batch_size for column in columns):
        raise ValueError("All image columns must have the same batch size")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(batch_size, len(columns), squeeze=False, figsize=(2.5 * len(columns), 2.7 * batch_size))
    for row in range(batch_size):
        for col, column in enumerate(columns):
            image = column[row].detach().cpu().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).numpy()
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(labels[col], fontsize=9)
            if col == 0 and row_labels is not None:
                axes[row, col].set_ylabel(str(row_labels[row]), fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
