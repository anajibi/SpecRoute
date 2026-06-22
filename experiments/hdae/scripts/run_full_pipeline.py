#!/usr/bin/env python
"""One-command HDAE pipeline: train, reconstruct, extract/probe, and CF eval.

The script is intentionally a transparent orchestrator around the individual
entrypoints. It logs every subprocess command and skips stages whose expected
outputs already exist, so long runs are restartable and inspectable.
"""
import argparse, logging, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def outputs_exist(paths):
    return all(Path(path).exists() for path in paths)


def run(cmd, *, outputs=(), force=False, skip=False, reason=""):
    if skip:
        logging.info("SKIP %s%s", " ".join(cmd), f" ({reason})" if reason else "")
        return
    if outputs and outputs_exist(outputs) and not force:
        logging.info("SKIP %s (already done: %s)", " ".join(cmd), ", ".join(map(str, outputs)))
        return
    logging.info("RUN  %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--ckpt", default=None, help="Existing HDAE checkpoint. If omitted, train.py is run and last.ckpt is used.")
    p.add_argument("--force", action="store_true", help="Re-run stages even if their expected outputs already exist.")
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
    data = raw["data"]
    out = Path(args.output_dir or raw["output_dir"])
    lmdb_meta = Path(data["lmdb_path"]) / "meta.json"
    attr_npz = Path(data["attr_npz"])
    recon_summary = out / "reconstruction" / "recon_summary.json"
    latents = out / "latent_probing" / "latents.npz"
    probes = out / "latent_probing" / "probes"
    probe_metrics = probes / "probe_metrics.csv"
    attr_ckpt = out / "counterfactuals" / "attr_classifier.pt"
    probe_analysis = out / "latent_probing" / "analysis"
    probe_analysis_summary = probe_analysis / "analysis_summary.json"
    swap_grid = out / "latent_probing" / "swap_null_grid.png"
    abduct_grid = out / "latent_probing" / "abduct_xt_z_grid.png"
    cf_out = out / "counterfactuals" / args.attribute
    cf_summary = cf_out / "summary.json"
    ckpt = Path(args.ckpt) if args.ckpt else out / "checkpoints" / "last.ckpt"
    py = sys.executable
    run([py, "experiments/hdae/scripts/preprocess_data.py", "--config", args.config],
        outputs=[lmdb_meta, attr_npz], force=args.force, skip=args.skip_preprocess, reason="requested")
    run([py, "experiments/hdae/scripts/train.py", "--config", args.config],
        outputs=[ckpt], force=args.force, skip=args.skip_train or args.ckpt is not None,
        reason="requested/external checkpoint")
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}. Pass --ckpt or run training first.")
    run([py, "experiments/hdae/scripts/reconstruct.py", "--config", args.config, "--ckpt", str(ckpt)],
        outputs=[recon_summary], force=args.force)
    run([py, "experiments/hdae/latent_probing/extract_latents.py", "--config", args.config, "--ckpt", str(ckpt), "--output", str(latents)],
        outputs=[latents], force=args.force)
    run([py, "experiments/hdae/latent_probing/train_linear_probes.py", "--latents", str(latents), "--output-dir", str(probes)],
        outputs=[probe_metrics], force=args.force)
    run([py, "experiments/hdae/latent_probing/analyze_probe_results.py", "--probe-metrics", str(probe_metrics), "--output-dir", str(probe_analysis)],
        outputs=[probe_analysis_summary, probe_analysis / "probe_heatmap.png", probe_analysis / "best_level_counts.png"], force=args.force)
    run([py, "experiments/hdae/latent_probing/swap_null_grid.py", "--config", args.config, "--ckpt", str(ckpt), "--output", str(swap_grid)],
        outputs=[swap_grid, swap_grid.with_suffix(".json")], force=args.force)
    run([py, "experiments/hdae/latent_probing/abduct_xt_z_grid.py", "--config", args.config, "--ckpt", str(ckpt), "--output", str(abduct_grid)],
        outputs=[abduct_grid, abduct_grid.with_suffix(".json")], force=args.force)
    run([py, "experiments/hdae/counterfactuals/train_attr_classifier.py", "--config", args.config, "--output", str(attr_ckpt)],
        outputs=[attr_ckpt], force=args.force, skip=args.skip_attr_classifier and attr_ckpt.exists(), reason="requested and checkpoint exists")
    run([py, "experiments/hdae/counterfactuals/run_counterfactual_eval.py", "--config", args.config, "--ckpt", str(ckpt),
         "--probe-metrics", str(probe_metrics), "--probe-weights-dir", str(probes / "weights"),
         "--attr-classifier", str(attr_ckpt), "--attribute", args.attribute, "--level", str(args.cf_level),
         "--strength", str(args.cf_strength), "--num-images", str(args.num_cf_images), "--output-dir", str(cf_out)],
        outputs=[cf_summary], force=args.force)
    logging.info("Pipeline complete. Outputs are under %s", out)


if __name__ == "__main__":
    main()
