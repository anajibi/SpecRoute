#!/usr/bin/env python
"""One-command HDAE pipeline: train, reconstruct, extract/probe, and CF eval.

The script is intentionally a transparent orchestrator around the individual
entrypoints. It logs every subprocess command so long runs are restartable and
inspectable.
"""
import argparse, logging, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def run(cmd, skip=False):
    logging.info("%s", "SKIP " + " ".join(cmd) if skip else "RUN  " + " ".join(cmd))
    if not skip:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--ckpt", default=None, help="Existing HDAE checkpoint. If omitted, train.py is run and last.ckpt is used.")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-attr-classifier", action="store_true")
    p.add_argument("--attribute", default="Smiling")
    p.add_argument("--cf-level", default="best")
    p.add_argument("--cf-strength", type=float, default=2.0)
    p.add_argument("--num-cf-images", type=int, default=64)
    args = p.parse_args()
    import yaml
    raw = yaml.safe_load(open(args.config))
    out = Path(args.output_dir or raw["output_dir"])
    latents = out / "latent_probing" / "latents.npz"
    probes = out / "latent_probing" / "probes"
    attr_ckpt = out / "counterfactuals" / "attr_classifier.pt"
    probe_analysis = out / "latent_probing" / "analysis"
    swap_grid = out / "latent_probing" / "swap_null_grid.png"
    cf_out = out / "counterfactuals" / args.attribute
    ckpt = Path(args.ckpt) if args.ckpt else out / "checkpoints" / "last.ckpt"
    py = sys.executable
    run([py, "experiments/hdae/scripts/preprocess_data.py", "--config", args.config], skip=args.skip_preprocess)
    run([py, "experiments/hdae/scripts/train.py", "--config", args.config], skip=args.skip_train or args.ckpt is not None)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}. Pass --ckpt or run training first.")
    run([py, "experiments/hdae/scripts/reconstruct.py", "--config", args.config, "--ckpt", str(ckpt)])
    run([py, "experiments/hdae/latent_probing/extract_latents.py", "--config", args.config, "--ckpt", str(ckpt), "--output", str(latents)])
    run([py, "experiments/hdae/latent_probing/train_linear_probes.py", "--latents", str(latents), "--output-dir", str(probes)])
    run([py, "experiments/hdae/latent_probing/analyze_probe_results.py", "--probe-metrics", str(probes / "probe_metrics.csv"), "--output-dir", str(probe_analysis)])
    run([py, "experiments/hdae/latent_probing/swap_null_grid.py", "--config", args.config, "--ckpt", str(ckpt), "--output", str(swap_grid)])
    run([py, "experiments/hdae/counterfactuals/train_attr_classifier.py", "--config", args.config, "--output", str(attr_ckpt)], skip=args.skip_attr_classifier and attr_ckpt.exists())
    run([py, "experiments/hdae/counterfactuals/run_counterfactual_eval.py", "--config", args.config, "--ckpt", str(ckpt),
         "--probe-metrics", str(probes / "probe_metrics.csv"), "--probe-weights-dir", str(probes / "weights"),
         "--attr-classifier", str(attr_ckpt), "--attribute", args.attribute, "--level", str(args.cf_level),
         "--strength", str(args.cf_strength), "--num-images", str(args.num_cf_images), "--output-dir", str(cf_out)])
    logging.info("Pipeline complete. Outputs are under %s", out)


if __name__ == "__main__":
    main()
