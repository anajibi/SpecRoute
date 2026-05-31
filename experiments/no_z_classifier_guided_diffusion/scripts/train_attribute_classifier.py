from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
import sys
for candidate in (REPO_ROOT, REPO_ROOT / "diffae_latent_probe", EXPERIMENT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.datasets import CelebAAttributeDataset  # noqa: E402
from src.utils import ensure_dir, ensure_torch_xpu_compat, load_yaml, resolve_device, save_json, set_seed  # noqa: E402
ensure_torch_xpu_compat()
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

    train_dataset = CelebAAttributeDataset(
        image_dir=dataset_cfg["image_dir"],
        attr_path=dataset_cfg["attr_path"],
        partition_path=dataset_cfg.get("partition_path"),
        split="train",
        image_size=int(dataset_cfg.get("image_size", 256)),
    )
    attr_names = train_dataset.attr_names

    if checkpoint_path.exists() and not bool(classifier_cfg.get("retrain", False)):
        model = AttributeClassifier(
            num_attributes=len(attr_names),
            backbone=classifier_cfg.get("backbone", "resnet18"),
            pretrained=bool(classifier_cfg.get("pretrained", False)),
        )
        payload = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"] if "model_state_dict" in payload else payload)
        model.to(device).eval()
        return model, attr_names, checkpoint_path, True

    val_dataset = CelebAAttributeDataset(
        image_dir=dataset_cfg["image_dir"],
        attr_path=dataset_cfg["attr_path"],
        partition_path=dataset_cfg.get("partition_path"),
        split="val",
        image_size=int(dataset_cfg.get("image_size", 256)),
        attr_names=attr_names,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(classifier_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(classifier_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(classifier_cfg.get("batch_size", 64)),
        shuffle=False,
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
        epochs=int(classifier_cfg.get("epochs", 3)),
        backbone=classifier_cfg.get("backbone", "resnet18"),
        pretrained=bool(classifier_cfg.get("pretrained", False)),
    )
    model = train_attribute_classifier(model, train_loader, val_loader, device, ac_cfg)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "attr_names": attr_names,
            "classifier_config": dict(classifier_cfg),
        },
        checkpoint_path,
    )
    save_json({"attr_names": attr_names, "checkpoint_path": str(checkpoint_path)}, metadata_path)
    model.eval()
    return model, attr_names, checkpoint_path, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or load the no-Z POC CelebA attribute classifier.")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config" / "no_z_classifier_guided_poc.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    device = resolve_device(cfg["experiment"].get("device", "cuda"))
    _, _, checkpoint_path, loaded = train_or_load_classifier(cfg, device)
    action = "Loaded" if loaded else "Trained and saved"
    print(f"{action} classifier checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
