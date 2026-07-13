import torch
import torch.nn as nn
import  logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

class HuggingFaceResNetWrapper(nn.Module):
    """
    Optimized for 64x64 inputs. Uses a ResNet backbone which natively handles
    variable spatial resolutions without requiring aggressive, blurry upscaling.
    """

    def __init__(self, model_name="microsoft/resnet-50", num_attributes=40):
        super().__init__()
        logging.info(f"Initializing CNN backbone: {model_name}")

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.register_buffer("mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1))

        self.cnn = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_attributes,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        # 1. Convert [-1, 1] diffusion range to [0, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0

        # 2. Apply ImageNet normalization directly at native 64x64 resolution.
        # HuggingFace ResNet uses convolutional/global-pooling heads, so it can
        # consume 64x64 tensors without the blur and extra cost of upsampling.
        x_norm = (x - self.mean) / self.std

        outputs = self.cnn(pixel_values=x_norm)
        return outputs.logits


def load_classifier(checkpoint_path=None, device="cpu"):
    """
    Load a HuggingFace ResNet-based attribute classifier from a checkpoint.
    """
    import torch

    checkpoint_path = checkpoint_path or "/home/anajibi/HDM/experiments/hdae/outputs/finetuned_attr_classifier.pt"
    logging.info(f"Loading attribute classifier from {checkpoint_path}")

    # Load the state dict and attribute names
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    attribute_names = checkpoint["attribute_names"]

    # Initialize the model with the correct number of attributes/backbone.
    model_class = checkpoint.get("model_class", "HuggingFaceResNetWrapper")
    if model_class == "CelebAAttributeCNN":
        model = CelebAAttributeCNN(num_attributes=len(attribute_names))
    else:
        model = HuggingFaceResNetWrapper(num_attributes=len(attribute_names))
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model, checkpoint

class CelebAAttributeCNN(nn.Module):
    """Small native-64x64 CNN for multi-label CelebA attribute training."""

    def __init__(self, num_attributes=40):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, num_attributes)

    def forward(self, x):
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        feats = self.features(x).flatten(1)
        return self.head(feats)


def _targets(batch, device):
    return (batch["attr"].to(device) > 0).float()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_items = 0.0, 0
    loss_fn = nn.BCEWithLogitsLoss()
    for batch in loader:
        x = batch["img"].to(device)
        y = _targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
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
            x = batch["img"].to(device)
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
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "attribute_names": list(attribute_names),
        "image_size": int(image_size),
        "metrics": metrics or {},
        "model_class": model.__class__.__name__,
    }, path)
