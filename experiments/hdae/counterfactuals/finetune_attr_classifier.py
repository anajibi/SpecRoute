#!/usr/bin/env python
"""Train an image-space multi-label CelebA attribute classifier for CF scoring."""
import argparse, logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from experiments.hdae.counterfactuals.attr_classifier import (
    TimmAttrClassifier, train_epoch, evaluate, save_classifier, load_classifier,
    compute_pos_weight, calibrate_thresholds,
)
from experiments.hdae.data.datamodule import CelebAHQDataModule
from experiments.hdae.hdae.config_io import load_hdae_config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def log_worst_attributes(val, k=10):
    """Print the k lowest-F1 attributes so failures are visible, not hidden in the mean."""
    f1 = val.get("per_attr_f1", {})
    if not f1:
        return
    worst = sorted(f1.items(), key=lambda kv: kv[1])[:k]
    logging.info("worst %d attributes by F1: %s", k,
                 ", ".join(f"{n}={v:.3f}" for n, v in worst))
    for key in ("Eyeglasses", "Young", "Male", "Smiling"):
        if key in f1:
            logging.info("  modeled: %s F1=%.3f acc=%.3f", key, f1[key],
                         val["per_attr_acc"].get(key, float("nan")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="/home/anajibi/HDM/experiments/hdae/configs/hier_k1.yaml")
    p.add_argument("--output", default="/home/anajibi/HDM/experiments/hdae/outputs/finetuned_attr_classifier.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)  # timm backbone trains from a higher LR than a frozen HF head
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--model-name", default="resnet34d")  # small-image stem: preserves 64px detail
    p.add_argument("--upsample-to", type=int, default=None,
                   help="Optional: upsample inputs to this size before the backbone (e.g. 128). None = 64px native.")
    p.add_argument("--no-pretrained", action="store_true", help="Train backbone from scratch instead of ImageNet init.")
    p.add_argument("--resume", action="store_true", help="Resume from --output if it exists.", default=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)

    logging.info("Building packed CelebA-HQ datamodule from %s", data["lmdb_path"])
    dm = CelebAHQDataModule(data["lmdb_path"], data["attr_npz"], args.batch_size, args.num_workers, flip_aug=True)
    dm.setup()
    attribute_names = list(dm.attribute_names)

    model = TimmAttrClassifier(
        model_name=args.model_name, num_attributes=len(attribute_names),
        pretrained=not args.no_pretrained, upsample_to=args.upsample_to,
    ).to(device)
    if args.resume and Path(args.output).exists():
        logging.info("resuming classifier from %s", args.output)
        model, _ = load_classifier(args.output, device)

    # Per-attribute pos_weight so rare attributes (Eyeglasses, Bald, Hat) are learned, not abstained on.
    logging.info("computing per-attribute pos_weight from training labels ...")
    pos_weight = compute_pos_weight(dm.train_dataloader(), len(attribute_names), device)
    logging.info("pos_weight range: min=%.2f max=%.2f", float(pos_weight.min()), float(pos_weight.max()))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"macro_f1": -1.0}
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, dm.train_dataloader(), opt, device,
                                 pos_weight=pos_weight, grad_clip=args.grad_clip)
        val = evaluate(model, dm.val_dataloader(), device,
                       attribute_names=attribute_names, pos_weight=pos_weight)

        logging.info("epoch=%d train_loss=%.4f val_loss=%.4f mean_acc=%.4f macro_f1=%.4f",
                     epoch, train_loss, val["loss"], val["label_accuracy"], val["macro_f1"])
        log_worst_attributes(val)

        # Select on macro-F1, NOT loss/accuracy: macro-F1 rewards learning the rare attributes.
        if val["macro_f1"] > best["macro_f1"]:
            thresholds = calibrate_thresholds(val["_probs"], val["_y"], attribute_names)
            best = {"epoch": epoch, "train_loss": train_loss,
                    "loss": val["loss"], "label_accuracy": val["label_accuracy"],
                    "macro_f1": val["macro_f1"], "per_attr_f1": val["per_attr_f1"],
                    "per_attr_acc": val["per_attr_acc"]}
            save_classifier(args.output, model, attribute_names, data["image_size"],
                            metrics=best, thresholds=thresholds)
            logging.info("saved new best classifier (macro_f1=%.4f) to %s", val["macro_f1"], args.output)

    logging.info("done; best epoch=%s macro_f1=%.4f", best.get("epoch"), best.get("macro_f1", -1))


if __name__ == "__main__":
    main()