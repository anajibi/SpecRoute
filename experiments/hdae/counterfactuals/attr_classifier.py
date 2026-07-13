import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


class TimmAttrClassifier(nn.Module):
    """Attribute classifier on a small-image-stem timm backbone (resnet34d by default).

    The `d` variant replaces ResNet's 7x7 stride-2 stem with three 3x3 convs, which
    preserves spatial resolution at 64px so small attributes (glasses, earrings)
    survive to the feature layers. Input is expected in [0,1] or [-1,1]; ImageNet
    normalization is applied internally.
    """

    def __init__(self, model_name="resnet34d", num_attributes=40, pretrained=True,
                 upsample_to=None):
        super().__init__()
        import timm
        self.model_name = str(model_name)
        # optional: upsample tiny inputs before the backbone. None = feed 64px native
        # (resnet34d handles 64px fine; set e.g. 128 only if you want more headroom).
        self.upsample_to = upsample_to
        self.backbone = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=num_attributes)
        # ImageNet normalization (matches timm pretrained stats)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        if self.upsample_to is not None and x.shape[-1] != self.upsample_to:
            x = nn.functional.interpolate(x, size=(self.upsample_to, self.upsample_to),
                                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)


def _targets(batch, device):
    return (batch["attr"].to(device, non_blocking=True) > 0).float()


def compute_pos_weight(loader, num_attributes, device, max_batches=200):
    """Per-attribute pos_weight = (#neg / #pos), so rare attributes aren't ignored."""
    pos = torch.zeros(num_attributes)
    total = 0
    for i, batch in enumerate(loader):
        y = (batch["attr"] > 0).float()
        pos += y.sum(0)
        total += len(y)
        if i + 1 >= max_batches:
            break
    pos_rate = (pos / max(1, total)).clamp(min=1e-4, max=1 - 1e-4)
    return ((1 - pos_rate) / pos_rate).to(device)


def train_epoch(model, loader, optimizer, device, *, pos_weight=None, grad_clip=1.0):
    model.train()
    total_loss, total_items = 0.0, 0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for step, batch in enumerate(loader):
        x = batch["img"].to(device, non_blocking=True)
        y = _targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach()) * len(x)
        total_items += len(x)
    return total_loss / max(1, total_items)


@torch.no_grad()
def evaluate(model, loader, device, attribute_names=None, pos_weight=None):
    """Return loss, mean acc, and PER-ATTRIBUTE accuracy + F1 (the numbers that matter)."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n_attr = len(attribute_names) if attribute_names else None
    all_logits, all_y = [], []
    total_loss, total_items = 0.0, 0
    for batch in loader:
        x = batch["img"].to(device, non_blocking=True)
        y = _targets(batch, device)
        logits = model(x)
        total_loss += float(loss_fn(logits, y).detach()) * len(x)
        total_items += len(x)
        all_logits.append(torch.sigmoid(logits).cpu())
        all_y.append(y.cpu())
    probs = torch.cat(all_logits).numpy()
    y = torch.cat(all_y).numpy()
    n_attr = n_attr or probs.shape[1]

    pred = (probs >= 0.5).astype(np.int8)
    per_acc, per_f1 = {}, {}
    for j in range(n_attr):
        name = attribute_names[j] if attribute_names else str(j)
        yj, pj = y[:, j], pred[:, j]
        per_acc[name] = float((pj == yj).mean())
        tp = float(((pj == 1) & (yj == 1)).sum())
        fp = float(((pj == 1) & (yj == 0)).sum())
        fn = float(((pj == 0) & (yj == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_f1[name] = float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0

    return {"loss": total_loss / max(1, total_items),
            "label_accuracy": float((pred == y).mean()),
            "macro_f1": float(np.mean(list(per_f1.values()))),
            "per_attr_acc": per_acc, "per_attr_f1": per_f1,
            "_probs": probs, "_y": y}


def calibrate_thresholds(probs, y, attribute_names):
    """Per-attribute F1-optimal threshold on held-out real data."""
    grid = np.linspace(0.05, 0.95, 19)
    thr = {}
    for j, name in enumerate(attribute_names):
        yj = y[:, j]
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pj = (probs[:, j] >= t).astype(np.int8)
            tp = ((pj == 1) & (yj == 1)).sum()
            fp = ((pj == 1) & (yj == 0)).sum()
            fn = ((pj == 0) & (yj == 1)).sum()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[name] = best_t
    return thr


def save_classifier(path, model, attribute_names, image_size, metrics=None, thresholds=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "attribute_names": list(attribute_names),
        "image_size": int(image_size),
        "metrics": metrics or {},
        "thresholds": thresholds or {},
        "model_class": "TimmAttrClassifier",
        "model_name": getattr(model, "model_name", "resnet34d"),
        "upsample_to": getattr(model, "upsample_to", None),
    }, path)


def load_classifier(checkpoint_path=None, device="cpu"):
    checkpoint_path = checkpoint_path or "/home/anajibi/HDM/experiments/hdae/outputs/finetuned_attr_classifier.pt"
    logging.info("Loading attribute classifier from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    attribute_names = checkpoint["attribute_names"]
    model = TimmAttrClassifier(
        model_name=checkpoint.get("model_name", "resnet34d"),
        num_attributes=len(attribute_names),
        pretrained=False,
        upsample_to=checkpoint.get("upsample_to", None),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint