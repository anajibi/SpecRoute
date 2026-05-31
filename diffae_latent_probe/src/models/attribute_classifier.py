from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

logger = logging.getLogger(__name__)


@dataclass
class AttributeClassifierConfig:
    num_attributes: int
    lr: float = 1e-3
    epochs: int = 3
    backbone: str = "resnet18"
    pretrained: bool = False


class AttributeClassifier(nn.Module):
    def __init__(self, num_attributes: int, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone}")
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = "IMAGENET1K_V1"
        backbone_model = models.resnet18(weights=weights)
        backbone_model.fc = nn.Linear(backbone_model.fc.in_features, num_attributes)
        self.backbone = backbone_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def train_attribute_classifier(
    model: AttributeClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    cfg: AttributeClassifierConfig,
) -> AttributeClassifier:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    logger.info(
        "Starting attribute classifier training: epochs=%d lr=%.4f device=%s",
        cfg.epochs,
        cfg.lr,
        device,
    )

    for epoch_idx in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["attributes"].to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % max(1, len(train_loader) // 5) == 0 and batch_idx > 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    "Epoch %d/%d | batch %d/%d | avg loss=%.4f",
                    epoch_idx + 1,
                    cfg.epochs,
                    batch_idx,
                    len(train_loader),
                    avg_loss,
                )

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logger.info("Epoch %d/%d complete | avg loss=%.4f", epoch_idx + 1, cfg.epochs, avg_epoch_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            with torch.inference_mode():
                for val_batch in val_loader:
                    images = val_batch["image"].to(device, non_blocking=True)
                    labels = val_batch["attributes"].to(device, non_blocking=True)
                    logits = model(images)
                    val_loss += loss_fn(logits, labels).item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
            logger.info("Epoch %d/%d | validation loss=%.4f", epoch_idx + 1, cfg.epochs, avg_val_loss)

    logger.info("Attribute classifier training complete")
    return model


def predict_attribute_probabilities(
    model: AttributeClassifier,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    probs = []
    logger.info("Starting attribute probability predictions")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            batch_probs = torch.sigmoid(logits).cpu()
            probs.append(batch_probs)

            if batch_idx % max(1, len(loader) // 5) == 0 and batch_idx > 0:
                logger.info("Predictions: batch %d/%d", batch_idx, len(loader))

    result = torch.cat(probs, dim=0)
    logger.info("Attribute probability predictions complete: %d samples", result.shape[0])
    return result
