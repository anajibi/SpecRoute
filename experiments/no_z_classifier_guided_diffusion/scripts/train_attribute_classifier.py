from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
import sys
for candidate in (REPO_ROOT, REPO_ROOT / "diffae_latent_probe", EXPERIMENT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.datasets import CelebAAttributeDataset, write_dicts_csv  # noqa: E402
from src.utils import ensure_dir, ensure_torch_xpu_compat, load_yaml, resolve_device, save_json, set_seed  # noqa: E402
ensure_torch_xpu_compat()
from diffae_latent_probe.src.models.attribute_classifier import (  # noqa: E402
    AttributeClassifier,
    AttributeClassifierConfig,
    train_attribute_classifier,
)


def _target_attr_names(cfg: dict) -> list[str]:
    target_attrs = list(cfg.get("editing", {}).get("target_attributes", []))
    if not target_attrs:
        raise ValueError("editing.target_attributes must contain at least one attribute for the classifier.")
    return target_attrs


@torch.inference_mode()
def evaluate_attribute_classifier(
    model: AttributeClassifier,
    val_loader: DataLoader,
    attr_names: list[str],
    device: torch.device,
    output_csv: Path,
) -> list[dict[str, object]]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["attributes"].to(device, non_blocking=True)
        logits = model(images)
        total_loss += F.binary_cross_entropy_with_logits(logits, labels).item()
        num_batches += 1
        logits_chunks.append(logits.detach().cpu())
        label_chunks.append(labels.detach().cpu())

    if not logits_chunks:
        raise ValueError("Validation loader is empty; cannot evaluate the attribute classifier.")

    logits_all = torch.cat(logits_chunks, dim=0)
    labels_all = torch.cat(label_chunks, dim=0)
    probs_all = torch.sigmoid(logits_all)
    preds_all = (probs_all >= 0.5).float()

    rows: list[dict[str, object]] = []
    for attr_idx, attr_name in enumerate(attr_names):
        labels = labels_all[:, attr_idx]
        preds = preds_all[:, attr_idx]
        probs = probs_all[:, attr_idx]
        tp = int(((preds == 1) & (labels == 1)).sum().item())
        tn = int(((preds == 0) & (labels == 0)).sum().item())
        fp = int(((preds == 1) & (labels == 0)).sum().item())
        fn = int(((preds == 0) & (labels == 1)).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "attribute": attr_name,
                "num_examples": int(labels.numel()),
                "positive_rate": float(labels.mean().item()),
                "bce_loss": float(F.binary_cross_entropy_with_logits(logits_all[:, attr_idx], labels).item()),
                "accuracy": float((preds == labels).float().mean().item()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mean_prob_positive_labels": float(probs[labels == 1].mean().item()) if int((labels == 1).sum().item()) else "",
                "mean_prob_negative_labels": float(probs[labels == 0].mean().item()) if int((labels == 0).sum().item()) else "",
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            }
        )

    rows.append(
        {
            "attribute": "__macro__",
            "num_examples": int(labels_all.shape[0]),
            "positive_rate": float(labels_all.mean().item()),
            "bce_loss": float(total_loss / max(num_batches, 1)),
            "accuracy": float((preds_all == labels_all).float().mean().item()),
            "precision": float(sum(float(row["precision"]) for row in rows) / len(rows)),
            "recall": float(sum(float(row["recall"]) for row in rows) / len(rows)),
            "f1": float(sum(float(row["f1"]) for row in rows) / len(rows)),
            "mean_prob_positive_labels": "",
            "mean_prob_negative_labels": "",
            "true_positive": int(((preds_all == 1) & (labels_all == 1)).sum().item()),
            "true_negative": int(((preds_all == 0) & (labels_all == 0)).sum().item()),
            "false_positive": int(((preds_all == 1) & (labels_all == 0)).sum().item()),
            "false_negative": int(((preds_all == 0) & (labels_all == 1)).sum().item()),
        }
    )
    write_dicts_csv(rows, output_csv)
    return rows


def _build_val_loader(dataset_cfg: dict, classifier_cfg: dict, attr_names: list[str], device: torch.device) -> DataLoader:
    val_dataset = CelebAAttributeDataset(
        image_dir=dataset_cfg["image_dir"],
        attr_path=dataset_cfg["attr_path"],
        partition_path=dataset_cfg.get("partition_path"),
        split="val",
        image_size=int(dataset_cfg.get("image_size", 256)),
        attr_names=attr_names,
    )
    return DataLoader(
        val_dataset,
        batch_size=int(classifier_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(classifier_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )


def train_or_load_classifier(cfg: dict, device: torch.device):
    dataset_cfg = cfg["dataset"]
    classifier_cfg = cfg["classifier"]
    output_dir = ensure_dir(EXPERIMENT_DIR / "outputs" / "attribute_classifier")
    checkpoint_path = output_dir / "attribute_classifier.pt"
    metadata_path = output_dir / "metadata.json"
    metrics_path = output_dir / "validation_metrics.csv"
    attr_names = _target_attr_names(cfg)

    if checkpoint_path.exists() and not bool(classifier_cfg.get("retrain", False)):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_attr_names = payload.get("attr_names") if isinstance(payload, dict) else None
        if checkpoint_attr_names == attr_names:
            model = AttributeClassifier(
                num_attributes=len(attr_names),
                backbone=classifier_cfg.get("backbone", "resnet18"),
                pretrained=bool(classifier_cfg.get("pretrained", False)),
            )
            model.load_state_dict(payload["model_state_dict"] if "model_state_dict" in payload else payload)
            model.to(device).eval()
            if not metrics_path.exists():
                val_loader = _build_val_loader(dataset_cfg, classifier_cfg, attr_names, device)
                evaluate_attribute_classifier(model, val_loader, attr_names, device, metrics_path)
            return model, attr_names, checkpoint_path, True
        print(
            "Cached classifier attributes do not match editing.target_attributes; "
            f"retraining classifier for {attr_names}."
        )

    train_dataset = CelebAAttributeDataset(
        image_dir=dataset_cfg["image_dir"],
        attr_path=dataset_cfg["attr_path"],
        partition_path=dataset_cfg.get("partition_path"),
        split="train",
        image_size=int(dataset_cfg.get("image_size", 256)),
        attr_names=attr_names,
    )
    val_loader = _build_val_loader(dataset_cfg, classifier_cfg, attr_names, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(classifier_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(classifier_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    model = AttributeClassifier(
        num_attributes=len(attr_names),
        backbone=classifier_cfg.get("backbone", "resnet18"),
        pretrained=bool(classifier_cfg.get("pretrained", False)),
    )
    ac_cfg = AttributeClassifierConfig(
        num_attributes=len(attr_names),
        lr=float(classifier_cfg.get("lr", 1e-3)),
        epochs=int(classifier_cfg.get("epochs", 15)),
        backbone=classifier_cfg.get("backbone", "resnet18"),
        pretrained=bool(classifier_cfg.get("pretrained", False)),
    )
    model = train_attribute_classifier(model, train_loader, val_loader, device, ac_cfg)
    validation_metrics = evaluate_attribute_classifier(model, val_loader, attr_names, device, metrics_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "attr_names": attr_names,
            "classifier_config": dict(classifier_cfg),
            "validation_metrics_path": str(metrics_path),
        },
        checkpoint_path,
    )
    save_json(
        {
            "attr_names": attr_names,
            "checkpoint_path": str(checkpoint_path),
            "validation_metrics_path": str(metrics_path),
            "validation_macro": validation_metrics[-1],
        },
        metadata_path,
    )
    model.eval()
    return model, attr_names, checkpoint_path, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or load the no-Z POC CelebA attribute classifier.")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config" / "no_z_classifier_guided_poc.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    device = resolve_device(cfg["experiment"].get("device", "cuda"))
    _, attr_names, checkpoint_path, loaded = train_or_load_classifier(cfg, device)
    action = "Loaded" if loaded else "Trained and saved"
    print(f"{action} classifier checkpoint: {checkpoint_path}")
    print(f"Classifier attributes: {', '.join(attr_names)}")


if __name__ == "__main__":
    main()
