from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.celeba_attributes import compute_attribute_prevalence, filter_attributes_by_prevalence
from src.models.attribute_classifier import AttributeClassifier, AttributeClassifierConfig, predict_attribute_probabilities, train_attribute_classifier


class AttributeSubsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, attr_indices: list[int], attr_names: list[str]):
        self.base_dataset = base_dataset
        self.attr_indices = attr_indices
        self.attr_names = attr_names

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        batch = self.base_dataset[index]
        attrs = batch["attributes"][self.attr_indices]
        return {
            "image_id": batch["image_id"],
            "image": batch["image"],
            "attributes": attrs,
        }


def build_other_attribute_list(
    attributes_df: pd.DataFrame,
    all_attr_names: list[str],
    target_attributes: list[str],
    prevalence_min: float,
    prevalence_max: float,
) -> tuple[list[str], pd.DataFrame]:
    candidate_attrs = [name for name in all_attr_names if name not in target_attributes]
    prevalence_df = compute_attribute_prevalence(attributes_df, candidate_attrs)
    prevalence_df["kept"] = prevalence_df["prevalence"].between(prevalence_min, prevalence_max)
    kept = filter_attributes_by_prevalence(attributes_df, candidate_attrs, prevalence_min, prevalence_max)
    return kept, prevalence_df


def train_or_load_other_attribute_classifier(
    output_dir: str | Path,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    attr_names: list[str],
    device: torch.device,
    cfg: AttributeClassifierConfig,
    force_retrain: bool = False,
) -> tuple[AttributeClassifier, list[str], bool]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    meta_path = output_dir / "metadata.json"

    if model_path.exists() and meta_path.exists() and not force_retrain:
        metadata = json.loads(meta_path.read_text())
        model = AttributeClassifier(
            num_attributes=metadata["num_attributes"],
            backbone=metadata["backbone"],
            pretrained=False,
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model.to(device), metadata["attr_names"], True

    model = AttributeClassifier(
        num_attributes=len(attr_names),
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
    )
    model = train_attribute_classifier(model, train_loader, val_loader, device, cfg)
    torch.save(model.state_dict(), model_path)
    meta_path.write_text(
        json.dumps(
            {
                "attr_names": attr_names,
                "num_attributes": len(attr_names),
                "backbone": cfg.backbone,
                "cfg": asdict(cfg),
            },
            indent=2,
        )
    )
    return model, attr_names, False


def evaluate_other_attribute_classifier(
    model: AttributeClassifier,
    val_loader: DataLoader,
    val_attr_frame: pd.DataFrame,
    attr_names: list[str],
    device: torch.device,
) -> pd.DataFrame:
    probs = predict_attribute_probabilities(model, val_loader, device)
    probs_df = pd.DataFrame(probs.numpy(), columns=attr_names)
    probs_df.insert(0, "image_id", val_attr_frame["image_id"].values)

    rows = []
    for name in attr_names:
        preds = probs_df[["image_id", name]].rename(columns={name: "pred"})
        truth = val_attr_frame[["image_id", name]].rename(columns={name: "gt"})
        merged = preds.merge(truth, on="image_id", how="inner")
        if merged.empty:
            continue
        pred_labels = (merged["pred"] >= 0.5).astype(int)
        gt_labels = merged["gt"].astype(int)
        accuracy = float((pred_labels == gt_labels).mean())
        rows.append({"attribute": name, "accuracy": accuracy, "num_samples": int(len(merged))})

    return pd.DataFrame(rows)
