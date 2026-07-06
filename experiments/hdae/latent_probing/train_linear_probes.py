#!/usr/bin/env python
"""CLI for the first HDAE latent-probing experiment: 40 attrs × K levels."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.hdae.latent_probing.linear_probe import train_all_probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", required=True, help="NPZ written by extract_latents.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-type", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    args = p.parse_args()
    print(f"training {args.probe_type} probes on {args.latents}...")
    rows = train_all_probes(args.latents, args.output_dir, lr=args.lr,
                            weight_decay=args.weight_decay, max_epochs=args.max_epochs,
                            batch_size=args.batch_size, patience=args.patience,
                            device=args.device, seed=args.seed, probe_type=args.probe_type,
                            hidden_dim=args.hidden_dim, dropout=args.dropout)
    print(f"trained {len(rows)} {args.probe_type} probes; metrics: {Path(args.output_dir) / 'probe_metrics.csv'}")


if __name__ == "__main__":
    main()
