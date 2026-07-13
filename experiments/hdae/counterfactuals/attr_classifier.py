import logging
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


class HuggingFaceResNetWrapper(nn.Module):
    """Strong ResNet attribute classifier that runs directly on native 64x64 tensors."""

    def __init__(self, model_name="microsoft/resnet-50", num_attributes=40):
        super().__init__()
        self.model_name = str(model_name)
        logging.info("Initializing CNN backbone: %s", self.model_name)

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.register_buffer("mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1))

        self.cnn = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=num_attributes,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        if x.min() < 0:
            x = (x + 1.0) / 2.0

        # Native-resolution path: the ResNet backbone is convolutional/global-pooling
        # based and accepts 64x64 tensors directly.
        x_norm = (x.clamp(0, 1) - self.mean) / self.std
        outputs = self.cnn(pixel_values=x_norm)
        return outputs.logits


def _targets(batch, device):
    return (batch["attr"].to(device, non_blocking=True) > 0).float()


def train_epoch(model, loader, optimizer, device, *, grad_clip=1.0):
    """Robust single-epoch multilabel training loop for the ResNet classifier."""
    model.train()
    total_loss, total_items = 0.0, 0
    loss_fn = nn.BCEWithLogitsLoss()
    for step, batch in enumerate(loader):
        x = batch["img"].to(device, non_blocking=True)
        y = _targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite classifier loss at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach()) * len(x)
        total_items += len(x)
    return total_loss / max(1, total_items)


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, total_labels, total_items = 0.0, 0, 0, 0
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            x = batch["img"].to(device, non_blocking=True)
            y = _targets(batch, device)
            logits = model(x)
            loss = loss_fn(logits, y)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += float(loss.detach()) * len(x)
            total_correct += int((pred == y).sum().detach().cpu())
            total_labels += int(y.numel())
            total_items += len(x)
    return {"loss": total_loss / max(1, total_items),
            "label_accuracy": total_correct / max(1, total_labels)}


def save_classifier(path, model, attribute_names, image_size, metrics=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "attribute_names": list(attribute_names),
        "image_size": int(image_size),
        "metrics": metrics or {},
        "model_class": "HuggingFaceResNetWrapper",
        "model_name": getattr(model, "model_name", "microsoft/resnet-50"),
        "native_resolution": True,
    }, path)


def load_classifier(checkpoint_path=None, device="cpu"):
    """Load the native-resolution HuggingFace ResNet attribute classifier."""
    checkpoint_path = checkpoint_path or "/home/anajibi/HDM/experiments/hdae/outputs/finetuned_attr_classifier.pt"
    logging.info("Loading attribute classifier from %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    attribute_names = checkpoint["attribute_names"]
    model_class = checkpoint.get("model_class", "HuggingFaceResNetWrapper")
    if model_class != "HuggingFaceResNetWrapper":
        raise ValueError(f"unsupported classifier checkpoint model_class={model_class!r}; expected HuggingFaceResNetWrapper")

    model = HuggingFaceResNetWrapper(
        model_name=checkpoint.get("model_name", "microsoft/resnet-50"),
        num_attributes=len(attribute_names),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, checkpoint
