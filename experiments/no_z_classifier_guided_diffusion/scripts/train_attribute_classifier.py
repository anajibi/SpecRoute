from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import torch
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

from src.datasets import CelebAAttributeDataset  # noqa: E402
from src.utils import ensure_dir, load_yaml, resolve_device, save_json, set_seed  # noqa: E402
from diffae_latent_probe.src.models.attribute_classifier import (  # noqa: E402
    AttributeClassifier,
    AttributeClassifierConfig,
    train_attribute_classifier,
)


def train_or_load_classifier(cfg: dict, device: torch.device):
    dataset_cfg = cfg["dataset"]
    classifier_cfg = cfg["classifier"]
    output_dir = ensure_dir(EXPERIMENT_DIR / "outputs" / "attribute_classifier")
    checkpoint_path = output_dir / "attribute_classifier.pt"
    metadata_path = output_dir / "metadata.json"

    logger.info("Initializing CelebA training dataset structure for validation and naming reference...")
    train_dataset = CelebAAttributeDataset(
        image_dir=dataset_cfg["image_dir"],
        attr_path=dataset_cfg["attr_path"],
        partition_path=dataset_cfg.get("partition_path"),
        split="train",
        image_size=int(dataset_cfg.get("image_size", 256)),
    )
    attr_names = train_dataset.attr_names
    num_attributes = len(attr_names)
    logger.info(f"Loaded attribute metadata. Number of attributes to classify: {num_attributes}")

    retrain_flag = bool(classifier_cfg.get("retrain", False))
    if checkpoint_path.exists() and not retrain_flag:
        logger.info(f"Found existing checkpoint at {checkpoint_path}. Attempting to load model.")

        backbone = classifier_cfg.get("backbone", "resnet18")
        pretrained = bool(classifier_cfg.get("pretrained", False))

        model = AttributeClassifier(
            num_attributes=num_attributes,
            backbone=backbone,
            pretrained=pretrained,
        )

        logger.info(f"Instantiated base model architecture with backbone: {backbone}")
        payload = torch.load(checkpoint_path, map_location="cpu")

        state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
        model.load_state_dict(state_dict)

        logger.info(f"Successfully mapped checkpoint state dict to model architecture.")
        model.to(device).eval()
        logger.info(f"Model transferred to target execution device: {device} and set to evaluation mode.")

        return model, attr_names, checkpoint_path, True

    if retrain_flag:
        logger.warning("Checkpoint execution bypassed: 'retrain' flag is explicitly set to True.")
    else:
        logger.info(f"No checkpoint discovered at {checkpoint_path}. Initiating standard training pipeline.")

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

    logger.info(f"Saving artifacts to disk. Checkpoint target: {checkpoint_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "attr_names": attr_names,
            "classifier_config": dict(classifier_cfg),
        },
        checkpoint_path,
    )

    logger.info(f"Saving metadata verification file to: {metadata_path}")
    save_json({"attr_names": attr_names, "checkpoint_path": str(checkpoint_path)}, metadata_path)

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

    _, _, checkpoint_path, loaded = train_or_load_classifier(cfg, device)

    action = "Loaded" if loaded else "Trained and saved"
    logger.info(f"Execution complete. Action taken: {action} classifier checkpoint at '{checkpoint_path}'")


if __name__ == "__main__":
    main()