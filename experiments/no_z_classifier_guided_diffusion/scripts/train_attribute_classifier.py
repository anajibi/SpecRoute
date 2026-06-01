from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Setup structured logger
logger = logging.getLogger("attribute_classifier")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]

for candidate in (REPO_ROOT, REPO_ROOT / "diffae_latent_probe", EXPERIMENT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.datasets import CelebAAttributeDataset, write_dicts_csv  # noqa: E402
from src.utils import ensure_dir, load_yaml, resolve_device, save_json, set_seed  # noqa: E402

from diffae_latent_probe.src.models.attribute_classifier import (  # noqa: E402
    AttributeClassifier,
    AttributeClassifierConfig,
    train_attribute_classifier,
)


def _safe_attribute_name(attribute: str) -> str:
    return attribute.replace("/", "_").replace(" ", "_")


def _target_attr_names(cfg: dict, target_attribute: str | None = None) -> list[str]:
    target_attrs = list(cfg.get("editing", {}).get("target_attributes", []))
    if target_attribute is not None:
        if target_attribute not in target_attrs:
            raise ValueError(f"{target_attribute!r} is not listed in editing.target_attributes: {target_attrs}")
        return [target_attribute]
    if len(target_attrs) != 1:
        raise ValueError(
            "The no-Z POC trains one binary classifier per edited attribute. "
            "Pass target_attribute when multiple editing.target_attributes are configured."
        )
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

    logger.info(f"Evaluating classifier against {len(val_loader)} validation batches...")
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["attributes"].to(device, non_blocking=True)
        logits = model(images)
        total_loss += F.binary_cross_entropy_with_logits(logits, labels).item()
        num_batches += 1
        logits_chunks.append(logits.detach().cpu())
        label_chunks.append(labels.detach().cpu())

    logits_all = torch.cat(logits_chunks, dim=0)
    labels_all = torch.cat(label_chunks, dim=0)
    preds_all = (logits_all > 0).float()

    rows = []
    rows.append(
        {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "accuracy": (preds_all == labels_all).float().mean().item(),
            "true_positive": int(((preds_all == 1) & (labels_all == 1)).sum().item()),
            "true_negative": int(((preds_all == 0) & (labels_all == 0)).sum().item()),
            "false_positive": int(((preds_all == 1) & (labels_all == 0)).sum().item()),
            "false_negative": int(((preds_all == 0) & (labels_all == 1)).sum().item()),
        }
    )
    write_dicts_csv(rows, output_csv)
    logger.info(f"Validation metrics saved to {output_csv}")
    return rows


def train_or_load_classifier(cfg: dict, device: torch.device, target_attribute: str | None = None):
    dataset_cfg = cfg["dataset"]
    classifier_cfg = cfg["classifier"]

    attr_names = _target_attr_names(cfg, target_attribute=target_attribute)
    classifier_name = _safe_attribute_name(attr_names[0])
    output_dir = ensure_dir(EXPERIMENT_DIR / "outputs" / "attribute_classifier" / classifier_name)

    checkpoint_path = output_dir / "attribute_classifier.pt"
    metadata_path = output_dir / "metadata.json"
    metrics_path = output_dir / "validation_metrics.csv"

    num_attributes = len(attr_names)

    retrain_flag = bool(classifier_cfg.get("retrain", False))
    if checkpoint_path.exists() and not retrain_flag:
        logger.info(f"Found existing checkpoint at {checkpoint_path}. Attempting to load model.")

        backbone = classifier_cfg.get("backbone", "resnet18")
        pretrained = bool(classifier_cfg.get("pretrained", False))

        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_attr_names = payload.get("attr_names") if isinstance(payload, dict) else None

        if checkpoint_attr_names == attr_names:
            model = AttributeClassifier(
                num_attributes=num_attributes,
                backbone=backbone,
                pretrained=pretrained,
            )
            logger.info(f"Instantiated base model architecture with backbone: {backbone}")

            state_dict = payload["model_state_dict"] if isinstance(payload,
                                                                   dict) and "model_state_dict" in payload else payload
            model.load_state_dict(state_dict)

            logger.info("Successfully mapped checkpoint state dict to model architecture.")
            model.to(device).eval()
            logger.info(f"Model transferred to target execution device: {device} and set to evaluation mode.")

            if not metrics_path.exists():
                logger.info("Validation metrics missing for cached model. Generating metrics...")
                val_dataset = CelebAAttributeDataset(
                    image_dir=dataset_cfg["image_dir"],
                    attr_path=dataset_cfg["attr_path"],
                    partition_path=dataset_cfg.get("partition_path"),
                    split="val",
                    image_size=int(dataset_cfg.get("image_size", 256)),
                    attr_names=attr_names,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=int(classifier_cfg.get("batch_size", 64)),
                    shuffle=False,
                    num_workers=int(classifier_cfg.get("num_workers", 2)),
                    pin_memory=device.type == "cuda",
                )
                evaluate_attribute_classifier(model, val_loader, attr_names, device, metrics_path)

            return model, attr_names, checkpoint_path, True
        else:
            logger.info(
                f"Cached classifier attributes ({checkpoint_attr_names}) do not match requested target attribute; retraining classifier for {attr_names}.")

    if retrain_flag:
        logger.warning("Checkpoint execution bypassed: 'retrain' flag is explicitly set to True.")
    else:
        logger.info(f"No valid checkpoint discovered at {checkpoint_path}. Initiating standard training pipeline.")

    logger.info("Initializing CelebA training dataset structure for validation and naming reference...")
    train_dataset = CelebAAttributeDataset(
        image_dir=dataset_cfg["image_dir"],
        attr_path=dataset_cfg["attr_path"],
        partition_path=dataset_cfg.get("partition_path"),
        split="train",
        image_size=int(dataset_cfg.get("image_size", 256)),
        attr_names=attr_names,
    )

    logger.info("Initializing validation dataset...")
    val_dataset = CelebAAttributeDataset(
        image_dir=dataset_cfg["image_dir"],
        attr_path=dataset_cfg["attr_path"],
        partition_path=dataset_cfg.get("partition_path"),
        split="val",
        image_size=int(dataset_cfg.get("image_size", 256)),
        attr_names=attr_names,
    )

    logger.info(
        f"Dataset configurations verified. Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    batch_size = int(classifier_cfg.get("batch_size", 64))
    num_workers = int(classifier_cfg.get("num_workers", 2))
    pin_memory = device.type == "cuda"

    logger.info(f"Building DataLoaders (Batch Size: {batch_size}, Workers: {num_workers}, Pin Memory: {pin_memory})")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    backbone = classifier_cfg.get("backbone", "resnet18")
    pretrained = bool(classifier_cfg.get("pretrained", False))
    lr = float(classifier_cfg.get("lr", 1e-3))
    epochs = int(classifier_cfg.get("epochs", 3))

    logger.info(f"Initializing target AttributeClassifier model with backbone {backbone}...")
    model = AttributeClassifier(
        num_attributes=num_attributes,
        backbone=backbone,
        pretrained=pretrained,
    )

    ac_cfg = AttributeClassifierConfig(
        num_attributes=num_attributes,
        lr=lr,
        epochs=epochs,
        backbone=backbone,
        pretrained=pretrained,
    )

    logger.info(f"Starting model execution training for {epochs} epochs with learning rate: {lr}...")
    model = train_attribute_classifier(model, train_loader, val_loader, device, ac_cfg)
    logger.info("Model training loop execution concluded.")

    logger.info("Running final evaluation to store metrics...")
    validation_metrics = evaluate_attribute_classifier(model, val_loader, attr_names, device, metrics_path)

    logger.info(f"Saving artifacts to disk. Checkpoint target: {checkpoint_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "attr_names": attr_names,
            "classifier_config": dict(classifier_cfg),
            "validation_metrics_path": str(metrics_path),
        },
        checkpoint_path,
    )

    logger.info(f"Saving metadata verification file to: {metadata_path}")
    save_json(
        {
            "attr_names": attr_names,
            "checkpoint_path": str(checkpoint_path),
            "validation_metrics_path": str(metrics_path),
            "validation_macro": validation_metrics[-1] if validation_metrics else None,
        },
        metadata_path
    )

    model.eval()
    logger.info("Model configured to evaluation mode.")
    return model, attr_names, checkpoint_path, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or load the no-Z POC CelebA attribute classifier.")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config" / "no_z_classifier_guided_poc.yaml"))
    args = parser.parse_args()

    logger.info(f"Parsing system runtime settings. Target config: {args.config}")
    cfg = load_yaml(args.config)

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)
    logger.info(f"Global runtime reproducibility seed locked to: {seed}")

    device_str = cfg["experiment"].get("device", "cuda")
    device = resolve_device(device_str)
    logger.info(f"System hardware resource target assigned: {device}")

    target_attrs = list(cfg.get("editing", {}).get("target_attributes", []))
    if not target_attrs:
        raise ValueError("editing.target_attributes must contain at least one attribute.")

    for target_attr in target_attrs:
        logger.info(f"Processing targeted attribute: {target_attr}")
        _, attr_names, checkpoint_path, loaded = train_or_load_classifier(cfg, device, target_attribute=target_attr)

        action = "Loaded" if loaded else "Trained and saved"
        logger.info(f"Execution complete. Action taken: {action} classifier checkpoint at '{checkpoint_path}'")
        logger.info(f"Classifier attribute registered: {attr_names[0]}")


if __name__ == "__main__":
    main()