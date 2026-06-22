"""Utilities for linear-probe latent directions and preservation metrics."""
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence
import csv
import inspect

import numpy as np


def read_probe_metrics(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def choose_probe_row(metrics_csv: str, attribute_name: str, level="best") -> dict:
    """Pick the row that defines a latent edit direction for an attribute.

    ``level='best'`` selects the level with highest validation balanced accuracy;
    otherwise ``level`` is interpreted as an integer latent level.
    """
    rows = [r for r in read_probe_metrics(metrics_csv) if r["attribute_name"] == attribute_name]
    if not rows:
        raise ValueError(f"attribute {attribute_name!r} not found in {metrics_csv}")
    if level == "best":
        return max(rows, key=lambda r: float(r.get("val_balanced_accuracy", r.get("test_balanced_accuracy", 0))))
    level = int(level)
    for row in rows:
        if int(row["level"]) == level:
            return row
    raise ValueError(f"attribute {attribute_name!r} has no probe for level {level}")


def probe_weight_path(weights_dir: str, row: Mapping[str, str]) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row["attribute_name"])
    return Path(weights_dir) / f"level{int(row['level']):02d}_attr{int(row['attribute_index']):02d}_{safe}.pt"


def torch_load_probe_checkpoint(path: str):
    """Load local probe checkpoints across PyTorch versions.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``. Older
    probe files can contain NumPy arrays for standardization stats, so local
    trusted probe checkpoints must be loaded with ``weights_only=False`` when the
    runtime exposes that argument. New probe files store tensor stats, but this
    keeps previously generated probes usable without retraining.
    """
    import torch
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def direction_from_probe_checkpoint(path: str):
    """Return a raw-latent-space direction from a saved standardized linear probe.

    The probe was trained on standardized features ``(z - mean) / std``.  A unit
    step in raw latent space aligned with the classifier therefore scales the
    learned standardized weight by ``1 / std`` before normalization.
    """
    state = torch_load_probe_checkpoint(path)
    weight = state["state_dict"]["weight"].detach().cpu().numpy().reshape(-1)
    std_value = state["std"]
    if hasattr(std_value, "detach"):
        std_value = std_value.detach().cpu().numpy()
    std = np.asarray(std_value).reshape(-1)
    std = np.where(std < 1e-6, 1.0, std)
    direction = weight / std
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError(f"zero probe direction in {path}")
    return direction / norm, state


def summarize_attribute_changes(before: np.ndarray, after: np.ndarray, target_index: int,
                                severe_threshold: float = 0.25) -> dict:
    """Summarize target edit strength and non-target preservation.

    ``before`` and ``after`` are probabilities/logits already converted to
    probabilities with shape ``(N, 40)``.
    """
    delta = after - before
    non_target = [i for i in range(delta.shape[1]) if i != target_index]
    abs_non_target = np.abs(delta[:, non_target])
    return {
        "target_delta_mean": float(delta[:, target_index].mean()),
        "target_delta_abs_mean": float(np.abs(delta[:, target_index]).mean()),
        "non_target_abs_delta_mean": float(abs_non_target.mean()),
        "non_target_abs_delta_max_mean": float(abs_non_target.max(axis=1).mean()),
        "non_target_severe_fraction": float((abs_non_target > severe_threshold).mean()),
        "num_images": int(delta.shape[0]),
        "num_non_target_attributes": int(len(non_target)),
    }
