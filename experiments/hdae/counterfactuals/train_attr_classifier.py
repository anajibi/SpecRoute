#!/usr/bin/env python
"""Train an image-space multi-label CelebA attribute classifier for CF scoring."""
import argparse, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import torch
from experiments.hdae.data.datamodule import CelebAHQDataModule
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.counterfactuals.attribute_classifier import (
    CelebAAttributeCNN, evaluate, save_classifier, train_epoch,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    logging.info("Building packed CelebA-HQ datamodule from %s", data["lmdb_path"])
    dm = CelebAHQDataModule(data["lmdb_path"], data["attr_npz"], args.batch_size, args.num_workers, flip_aug=True)
    dm.setup()
    model = CelebAAttributeCNN(num_attributes=len(dm.attribute_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = {"loss": float("inf")}
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, dm.train_dataloader(), opt, device)
        val = evaluate(model, dm.val_dataloader(), device)
        logging.info("epoch=%d train_loss=%.4f val_loss=%.4f val_label_acc=%.4f", epoch, train_loss, val["loss"], val["label_accuracy"])
        if val["loss"] < best["loss"]:
            best = {"epoch": epoch, "train_loss": train_loss, **val}
            save_classifier(args.output, model, dm.attribute_names, data["image_size"], best)
            logging.info("saved new best classifier to %s", args.output)
    logging.info("done; best=%s", best)


if __name__ == "__main__":
    main()
