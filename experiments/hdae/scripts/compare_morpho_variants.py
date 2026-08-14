#!/usr/bin/env python
"""Aggregate reconstruction metrics (reconstruct.py) across the k=1/5/11 MorphoMNIST HDAE
variants into one comparison table -- "what does k buy you" (see PROGRESS-SUMMARY / advisor
guidance: reconstruction quality vs k is the cheap, intervention-machinery-free comparison to
get working first; full counterfactual (CF1-style) evaluation is a larger, separately-scoped
follow-up -- see counterfactual_smoke_test.py for the lighter substitute built instead).

Run per-variant reconstruction first (reconstruct.py), then this script just reads each variant's
recon_summary.json and prints/writes a combined table. Safe to re-run at any point during training
-- always reflects whatever checkpoint each variant's recon_summary.json was last computed from.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["morpho_hier_k1", "morpho_hier_k5", "morpho_hier_k11"])
    p.add_argument("--outputs-dir", default="experiments/hdae/outputs")
    p.add_argument("--output", default="experiments/hdae/outputs/morpho_variant_comparison.csv")
    args = p.parse_args()

    rows = []
    for name in args.variants:
        summary_path = Path(args.outputs_dir) / name / "reconstruction" / "recon_summary.json"
        if not summary_path.exists():
            print(f"[skip] {name}: no recon_summary.json at {summary_path} (run reconstruct.py first)")
            continue
        summary = json.loads(summary_path.read_text())
        k = {"morpho_hier_k1": 1, "morpho_hier_k5": 5, "morpho_hier_k11": 11}.get(name)
        rows.append({
            "variant": name, "k": k,
            "lpips_mean": summary["lpips"]["mean"], "lpips_std": summary["lpips"]["std"],
            "mse_mean": summary["mse"]["mean"], "mse_std": summary["mse"]["std"],
            "ssim_mean": summary["ssim"]["mean"], "ssim_std": summary["ssim"]["std"],
        })

    if not rows:
        print("no variant reconstruction summaries found; nothing to compare")
        return

    rows.sort(key=lambda r: (r["k"] is None, r["k"]))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'variant':<16}{'k':>4}{'lpips':>10}{'mse':>10}{'ssim':>10}")
    for r in rows:
        print(f"{r['variant']:<16}{r['k']:>4}{r['lpips_mean']:>10.4f}{r['mse_mean']:>10.4f}{r['ssim_mean']:>10.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
