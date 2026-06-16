"""Small multi-label CelebA attribute classifier used to score pseudo-counterfactuals."""
import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class CelebAAttributeCNN(nn.Module):
    def __init__(self, num_attributes: int = 40, width: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GroupNorm(8, width * 2), nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.GroupNorm(8, width * 4), nn.SiLU(),
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1), nn.GroupNorm(8, width * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(width * 4, num_attributes)

    def forward(self, x):
        return self.head(self.features(x))


def batch_targets(batch):
    return (batch["attr"].float() > 0).float()


def train_epoch(model, loader, optimizer, device):
    model.train(); total_loss = total = 0
    for batch in loader:
        x = batch["img"].to(device); y = batch_targets(batch).to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(x); total += len(x)
    return total_loss / max(total, 1)


def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval(); total_loss = total = correct = labels = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["img"].to(device); y = batch_targets(batch).to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += float(loss.detach().cpu()) * len(x); total += len(x)
            correct += int((pred == y).sum().detach().cpu()); labels += y.numel()
    return {"loss": total_loss / max(total, 1), "label_accuracy": correct / max(labels, 1)}


def save_classifier(path, model, attribute_names, image_size, metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "attribute_names": list(attribute_names),
                "image_size": image_size, "metrics": metrics}, path)
    Path(path).with_suffix(".json").write_text(json.dumps(metrics, indent=2))


def load_classifier(path, device="cpu"):
    state = torch.load(path, map_location="cpu")
    model = CelebAAttributeCNN(num_attributes=len(state["attribute_names"]))
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), state
