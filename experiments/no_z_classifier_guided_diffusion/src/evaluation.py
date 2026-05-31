from __future__ import annotations

import csv
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .datasets import ImagePathDataset, write_dicts_csv
except ImportError:  # pragma: no cover
    from datasets import ImagePathDataset, write_dicts_csv


def read_dicts_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def predict_paths_to_csv(
    classifier,
    image_paths: Sequence[str | Path],
    attr_names: Sequence[str],
    output_csv: str | Path,
    device: torch.device,
    image_size: int = 256,
    batch_size: int = 16,
) -> list[dict[str, object]]:
    dataset = ImagePathDataset(image_paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict[str, object]] = []
    classifier.eval()
    with torch.inference_mode():
        cursor = 0
        for batch in loader:
            images = batch["image"].to(device)
            probs = torch.sigmoid(classifier(images)).cpu().numpy()
            paths = batch["image_path"]
            for row_idx, path in enumerate(paths):
                row: dict[str, object] = {"row_id": cursor, "image_path": str(path)}
                row.update({attr: float(probs[row_idx, attr_idx]) for attr_idx, attr in enumerate(attr_names)})
                rows.append(row)
                cursor += 1
    write_dicts_csv(rows, output_csv)
    return rows


def _load_image_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    from PIL import Image
    transforms = importlib.import_module("torchvision.transforms")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(Image.open(path).convert("RGB"))


def compute_edit_metrics(
    records: Sequence[dict[str, object]],
    original_predictions: Sequence[dict[str, object]],
    edited_predictions: Sequence[dict[str, object]],
    attr_names: Sequence[str],
    output_metrics_csv: str | Path,
    output_summary_csv: str | Path,
    image_size: int = 256,
    compute_mse: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    original_by_path = {str(row["image_path"]): row for row in original_predictions}
    edited_by_path = {str(row["image_path"]): row for row in edited_predictions}
    metric_rows: list[dict[str, object]] = []
    for record in records:
        original_path = str(record["original_path"])
        edited_path = str(record["edited_path"])
        target_attr = str(record["target_attribute"])
        target_value = int(record["target_value"])
        p_original = {attr: float(original_by_path[original_path][attr]) for attr in attr_names}
        p_edited = {attr: float(edited_by_path[edited_path][attr]) for attr in attr_names}
        target_delta = p_edited[target_attr] - p_original[target_attr]
        non_targets = [attr for attr in attr_names if attr != target_attr]
        row: dict[str, object] = {
            "image_id": record["image_id"],
            "original_path": original_path,
            "edited_path": edited_path,
            "target_attribute": target_attr,
            "target_value": target_value,
            "guidance_scale": float(record["guidance_scale"]),
            "target_success": bool(target_delta >= 0 if target_value == 1 else target_delta <= 0),
            "target_prob_original": p_original[target_attr],
            "target_prob_edited": p_edited[target_attr],
            "target_prob_delta": target_delta,
            "non_target_mean_abs_delta": float(np.mean([abs(p_edited[attr] - p_original[attr]) for attr in non_targets])) if non_targets else 0.0,
            "non_target_flip_rate": float(np.mean([(p_original[attr] >= 0.5) != (p_edited[attr] >= 0.5) for attr in non_targets])) if non_targets else 0.0,
        }
        if compute_mse:
            original_tensor = _load_image_tensor(original_path, image_size)
            edited_tensor = _load_image_tensor(edited_path, image_size)
            row["image_mse"] = float(torch.mean((original_tensor - edited_tensor) ** 2).item())
        metric_rows.append(row)

    grouped: dict[tuple[str, int, float], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["target_attribute"]), int(row["target_value"]), float(row["guidance_scale"]))].append(row)
    summary_rows: list[dict[str, object]] = []
    for (target_attr, target_value, guidance_scale), rows in sorted(grouped.items()):
        summary: dict[str, object] = {
            "target_attribute": target_attr,
            "target_value": target_value,
            "guidance_scale": guidance_scale,
            "num_images": len(rows),
            "target_success_rate": float(np.mean([bool(row["target_success"]) for row in rows])),
            "mean_target_prob_delta": float(np.mean([float(row["target_prob_delta"]) for row in rows])),
            "mean_non_target_abs_delta": float(np.mean([float(row["non_target_mean_abs_delta"]) for row in rows])),
            "mean_non_target_flip_rate": float(np.mean([float(row["non_target_flip_rate"]) for row in rows])),
        }
        if compute_mse:
            summary["mean_image_mse"] = float(np.mean([float(row["image_mse"]) for row in rows]))
        summary_rows.append(summary)

    write_dicts_csv(metric_rows, output_metrics_csv)
    write_dicts_csv(summary_rows, output_summary_csv)
    return metric_rows, summary_rows
