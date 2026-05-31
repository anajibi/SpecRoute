from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

try:
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn is optional
    sns = None

from .image_io import tensor_to_pil


def _ensure_dir(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_attribute_bars(df: pd.DataFrame, out_path, score_col: str = "test_primary_score"):
    out_path = _ensure_dir(out_path)
    pivot = df.pivot_table(index="attribute", columns="latent_source", values=score_col, aggfunc="mean")
    plt.figure(figsize=(max(10, 0.6 * len(pivot)), max(6, 0.35 * len(pivot))))
    if sns is not None:
        pivot.plot(kind="bar", ax=plt.gca())
    else:
        pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel(score_col)
    plt.title("Latent predictability by attribute")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_attribute_heatmap(df: pd.DataFrame, out_path, score_col: str = "test_primary_score"):
    out_path = _ensure_dir(out_path)
    pivot = df.pivot_table(index="attribute", columns="latent_source", values=score_col, aggfunc="mean")
    plt.figure(figsize=(max(8, 0.7 * len(pivot.columns)), max(6, 0.35 * len(pivot))))
    if sns is not None:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0)
    else:
        plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_reconstruction_panel(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    out_path,
    max_items: int = 32,
):
    out_path = _ensure_dir(out_path)

    originals = originals[:max_items].detach().cpu().float().clamp(0.0, 1.0)
    reconstructions = reconstructions[:max_items].detach().cpu().float().clamp(0.0, 1.0)

    if originals.shape != reconstructions.shape:
        raise ValueError(
            f"originals and reconstructions must have same shape, "
            f"got {tuple(originals.shape)} and {tuple(reconstructions.shape)}"
        )

    rows = []

    for orig, recon in zip(originals, reconstructions):
        if orig.ndim != 3 or recon.ndim != 3:
            raise ValueError(
                f"Expected [C,H,W] tensors, got {tuple(orig.shape)} and {tuple(recon.shape)}"
            )

        if orig.shape[0] != 3 or recon.shape[0] != 3:
            raise ValueError(
                f"Expected 3-channel tensors, got {tuple(orig.shape)} and {tuple(recon.shape)}"
            )



        # Each entry must be [3, H, W].
        rows.extend([orig, recon])

    grid = make_grid(
        torch.stack(rows),
        nrow=8,
        normalize=False,
        value_range=(0, 1),
    )

    tensor_to_pil(grid, denormalize=False).save(out_path)
    return out_path


def save_swap_panel(images: Iterable[torch.Tensor], out_path, nrow: int = 6):
    out_path = _ensure_dir(out_path)
    tensors = []
    for img in images:
        if img.ndim == 3:
            tensors.append(img.detach().cpu())
        else:
            tensors.append(img.squeeze(0).detach().cpu())
    grid = make_grid(torch.stack(tensors), nrow=nrow, normalize=True, value_range=(0, 1))
    tensor_to_pil(grid, denormalize=False).save(out_path)
    return out_path


def save_image_copy(src, dst):
    dst = _ensure_dir(dst)
    shutil.copy2(src, dst)
    return dst


