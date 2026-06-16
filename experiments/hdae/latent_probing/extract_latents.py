#!/usr/bin/env python
"""Extract HDAE semantic latents level-by-level for linear probing."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-online-model", action="store_true",
                   help="Use model weights instead of EMA weights from the checkpoint.")
    args = p.parse_args()
    cfg = load_hdae_config(args.config)
    data = cfg.raw["data"]
    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu")
    model = module.model if args.use_online_model else module.ema_model
    model.to(device).eval()
    zs_by_level, attrs, partitions, indices = None, [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["img"].to(device)
            encoded = model.encode(x)
            zs = [z.detach().cpu().float().numpy() for z in encoded["zs"]]
            if zs_by_level is None:
                zs_by_level = [[] for _ in zs]
            for bucket, z in zip(zs_by_level, zs):
                bucket.append(z)
            attrs.append(batch["attr"].numpy())
            partitions.append(batch["partition"].numpy())
            indices.append(batch["index"].numpy())
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"z_level_{i}": np.concatenate(chunks, axis=0)
               for i, chunks in enumerate(zs_by_level)}
    payload.update(attrs=np.concatenate(attrs, axis=0).astype(np.int8),
                   partitions=np.concatenate(partitions, axis=0).astype(np.int8),
                   indices=np.concatenate(indices, axis=0),
                   attribute_names=np.asarray(ds.attribute_names))
    np.savez_compressed(output, **payload)
    meta = {"config": args.config, "ckpt": args.ckpt, "output": str(output),
            "num_levels": len(zs_by_level), "num_images": int(len(ds)),
            "level_dims": [int(payload[f"z_level_{i}"].shape[1]) for i in range(len(zs_by_level))],
            "model_weights": "online" if args.use_online_model else "ema"}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
