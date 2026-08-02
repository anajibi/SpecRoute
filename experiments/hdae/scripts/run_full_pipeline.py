#!/usr/bin/env python
"""One-command HDAE pipeline: train, reconstruct, extract/probe, and CF eval.

The script is intentionally a transparent orchestrator around the individual
entrypoints. It logs every subprocess command and skips stages whose expected
outputs already exist, so long runs are restartable and inspectable.
"""
import argparse
import logging
import subprocess
import sys
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
    p.add_argument("--config", default="/home/anajibi/HDM/experiments/hdae/configs/hier_k5.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--ckpt", default=None,
                   help="Existing HDAE checkpoint. If omitted, train.py is run and last.ckpt is used.")
    p.add_argument("--force", action="store_true", help="Re-run stages even if their expected outputs already exist.")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-attr-classifier", action="store_true")
    p.add_argument("--attribute", default="Smiling")
    p.add_argument("--num-cf-images", type=int, default=64)
    p.add_argument("--cohort-config", default="experiments/hdae/configs/cohorts.yaml",
                   help="Global, model-agnostic cohort settings shared across all model configs.")
    p.add_argument("--cf1-edit-strength", type=float, default=5.0,
                   help="HDAE attribute-CFG guidance scale passed to CF1 rendering; 1.0 disables guidance blending.")
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph.yaml",
                   help="Shared causal-DAG config (TODO item 2), fit into an SCM before CF1 eval.")
    p.add_argument("--skip-cf1", action="store_true")
    args = p.parse_args()
    import yaml
    raw = yaml.safe_load(open(args.config))
    cohort_raw = yaml.safe_load(open(args.cohort_config))
    causal_raw = yaml.safe_load(open(args.causal_graph))
    scm_ckpt = Path(causal_raw["scm_checkpoint"])
    data = raw["data"]
    out = Path(args.output_dir or raw["output_dir"])
    lmdb_meta = Path(data["lmdb_path"]) / "meta.json"
    attr_npz = Path(data["attr_npz"])
    recon_summary = out / "reconstruction" / "recon_summary.json"
    latents = out / "latent_probing" / "latents.npz"
    probes = out / "latent_probing" / "probes"
    probe_metrics = probes / "probe_metrics.csv"
    attr_ckpt = out / "../finetuned_attr_classifier.pt"
    probe_analysis = out / "latent_probing" / "analysis"
    probe_analysis_summary = probe_analysis / "analysis_summary.json"
    swap_grid = out / "latent_probing" / "swap_null_grid.png"
    abduct_grid = out / "latent_probing" / "abduct_xt_z_grid.png"
    cf_out = out / "counterfactuals" / args.attribute
    cf_summary = cf_out / "summary.json"
    modeled_attrs = ",".join(raw.get("encoder", {}).get("conditioning_attrs", []))
    if not modeled_attrs:
        raise ValueError("No modeled attributes found; set encoder.conditioning_attrs in the model config.")
    cohorts = Path(cohort_raw["output"])
    cohort_weights = cohorts.with_name(cohorts.stem + "_intervention_weights.csv")
    cf1_out = out / "counterfactuals" / "cf1"
    cf1_per_intervention = cf1_out / "cf1_per_intervention.csv"
    cf1_aggregate = cf1_out / "cf1_aggregate.csv"
    cf1_grid = cf1_out / "cf1_experiments_grid.png"
    model_name = raw.get("model_name") or raw.get("name") or Path(raw["output_dir"]).name
    safe_model_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_name)
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
    run([py, "experiments/hdae/latent_probing/extract_latents.py", "--config", args.config, "--ckpt", str(ckpt),
         "--output", str(latents)],
        outputs=[latents], force=args.force)
    run([py, "experiments/hdae/latent_probing/train_linear_probes.py", "--latents", str(latents), "--output-dir",
         str(probes)],
        outputs=[probe_metrics], force=args.force)
    run([py, "experiments/hdae/latent_probing/analyze_probe_results.py", "--probe-metrics", str(probe_metrics),
         "--output-dir", str(probe_analysis)],
        outputs=[probe_analysis_summary, probe_analysis / "probe_heatmap.png",
                 probe_analysis / "best_level_counts.png"], force=args.force)
    run([py, "experiments/hdae/latent_probing/swap_null_grid.py", "--config", args.config, "--ckpt", str(ckpt),
         "--output", str(swap_grid)],
        outputs=[swap_grid, swap_grid.with_suffix(".json")], force=args.force)
    run([py, "experiments/hdae/latent_probing/abduct_xt_z_grid.py", "--config", args.config, "--ckpt", str(ckpt),
         "--output", str(abduct_grid)],
        outputs=[abduct_grid, abduct_grid.with_suffix(".json")], force=args.force)
    # run([py, "experiments/hdae/counterfactuals/train_attr_classifier.py", "--config", args.config, "--output", str(attr_ckpt)],
    #     outputs=[attr_ckpt], force=args.force, skip=args.skip_attr_classifier and attr_ckpt.exists(), reason="requested and checkpoint exists")
    run([py, "experiments/hdae/build_cohorts.py", "--attr-npz", str(attr_npz), "--attributes", modeled_attrs,
         "--num-images", str(cohort_raw["num_images"]), "--seed", str(cohort_raw["seed"]), "--output", str(cohorts)],
        outputs=[cohorts, cohort_weights], force=args.force, skip=args.skip_cf1, reason="CF1 skipped")
    run([py, "experiments/hdae/causal/train_scm.py", "--causal-graph", args.causal_graph, "--attr-npz", str(attr_npz)],
        outputs=[scm_ckpt], force=args.force, skip=args.skip_cf1, reason="CF1 skipped")
    run([py, "experiments/hdae/counterfactuals/run_counterfactual_eval.py", "--config", args.config, "--ckpt",
         str(ckpt),
         "--attr-classifier", str(attr_ckpt), "--attribute", args.attribute,
         "--num-images", str(args.num_cf_images), "--output-dir", str(cf_out)],
        outputs=[cf_summary], force=args.force)
    run([py, "experiments/hdae/counterfactuals/run_cf1_eval.py", "--model-type", "hdae", "--config", args.config,
         "--ckpt", str(ckpt), "--attr-classifier", str(attr_ckpt), "--cohorts", str(cohorts),
         "--lmdb-path", str(data["lmdb_path"]), "--causal-graph", args.causal_graph,
         "--output-dir", str(cf1_out), "--model-name", model_name, "--edit-strength", str(args.cf1_edit_strength)],
        outputs=[cf1_per_intervention, cf1_aggregate, cf1_grid, cf1_out / f"frontier_{safe_model_name}.png"],
        force=True, skip=args.skip_cf1, reason="CF1 skipped")
    logging.info("Pipeline complete. Outputs are under %s", out)
    logging.info("CF1 outputs: per-intervention=%s aggregate=%s cohort-weights=%s", cf1_per_intervention, cf1_aggregate,
                 cohort_weights)


if __name__ == "__main__":
    main()
