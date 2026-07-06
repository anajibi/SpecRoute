#!/usr/bin/env python
"""Train an image-space multi-label CelebA attribute classifier for CF scoring."""
import argparse, logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3];
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoImageProcessor

from experiments.hdae.data.datamodule import CelebAHQDataModule
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.counterfactuals.attribute_classifier import (
    evaluate, save_classifier, train_epoch,
)

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

        # 2. ResNets can technically process 64x64 natively, but the pretrained ImageNet
        # weights expect features at a slightly larger scale. Interpolating to 128x128
        # is a highly calculated middle-ground: it prevents the CNN's pooling layers
        # from crushing the 64x64 feature map down to a 2x2 grid before the final layer,
        # without introducing the severe blurring of a 224x224 upscale.
        x_resized = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        # 3. Apply ImageNet Normalization
        x_norm = (x_resized - self.mean) / self.std

        outputs = self.cnn(pixel_values=x_norm)
        return outputs.logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="/home/anajibi/HDM/experiments/hdae/configs/celeba64_hier_k3.yaml")
    p.add_argument("--output", default="/home/anajibi/HDM/experiments/hdae/outputs/finetuned_attr_classifier.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    # CRITICAL: Dropped default from 1e-3 to 2e-5 to protect pre-trained weights
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)

    logging.info("Building packed CelebA-HQ datamodule from %s", data["lmdb_path"])
    dm = CelebAHQDataModule(data["lmdb_path"], data["attr_npz"], args.batch_size, args.num_workers, flip_aug=True)
    dm.setup()

    # Swapped out CelebAAttributeCNN for our self-contained ViT wrapper
    model = HuggingFaceResNetWrapper(num_attributes=len(dm.attribute_names)).to(device)

    # Using AdamW with standard transformer weight decay configurations
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best = {"loss": float("inf")}
    for epoch in range(args.epochs):
        model.train()
        train_loss = train_epoch(model, dm.train_dataloader(), opt, device)

        model.eval()
        val = evaluate(model, dm.val_dataloader(), device)

        logging.info("epoch=%d train_loss=%.4f val_loss=%.4f val_label_acc=%.4f", epoch, train_loss, val["loss"],
                     val["label_accuracy"])
        if val["loss"] < best["loss"]:
            best = {"epoch": epoch, "train_loss": train_loss, **val}
            # Passes 224 as the explicit image dimension the classifier operates on
            save_classifier(args.output, model, dm.attribute_names, 224, best)
            logging.info("saved new best classifier to %s", args.output)

    logging.info("done; best=%s", best)


if __name__ == "__main__":
    main()