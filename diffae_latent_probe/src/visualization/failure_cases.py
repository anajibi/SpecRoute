from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.visualization.grids import save_labeled_grid


def save_case_grids(
    cases: pd.DataFrame,
    image_lookup: dict[str, torch.Tensor],
    out_dir: str | Path,
    image_size: int,
    title: str,
    max_cases: int = 8,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = cases.head(max_cases)
    if selected.empty:
        return

    images = []
    row_labels = []
    for _, row in selected.iterrows():
        image_id = row["image_id"]
        if image_id not in image_lookup:
            continue
        images.append(image_lookup[image_id])
        row_labels.append(image_id)

    if images:
        save_labeled_grid(
            images=images,
            col_labels=["image"],
            out_path=out_dir / f"{title}.png",
            image_size=image_size,
            row_labels=row_labels,
            title=title,
            nrow=1,
        )

