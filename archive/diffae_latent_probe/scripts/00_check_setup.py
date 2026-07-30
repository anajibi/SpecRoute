from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from diffae_tools.config_io import git_commit_hash, load_config, resolve_path


def fail(message: str):
    raise SystemExit(f"[setup-check] {message}")


def main():
    parser = argparse.ArgumentParser(description="Validate the DiffAE experimental setup.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    repo_root = resolve_path(cfg["repo_root"], base_dir=config_dir)
    checkpoint_path = resolve_path(cfg["checkpoint_path"], base_dir=config_dir)
    image_dir = resolve_path(cfg["image_dir"], base_dir=config_dir)
    aligned_image_dir = resolve_path(cfg["aligned_image_dir"], base_dir=config_dir)
    output_dir = resolve_path(cfg["output_dir"], base_dir=config_dir)
    checkpoint_dir = checkpoint_path.parent

    print("[setup-check] resolved paths:")
    for name, value in [
        ("repo_root", repo_root),
        ("checkpoint_dir", checkpoint_dir),
        ("checkpoint_path", checkpoint_path),
        ("image_dir", image_dir),
        ("aligned_image_dir", aligned_image_dir),
        ("output_dir", output_dir),
    ]:
        print(f"  {name}: {value}")

    if not repo_root.exists():
        fail(f"DiffAE repo_root does not exist: {repo_root}")
    if not (repo_root / "templates.py").exists() or not (repo_root / "experiment.py").exists():
        fail(f"repo_root does not look like the official DiffAE repository: {repo_root}")
    if git_commit_hash(repo_root) is None:
        fail(f"repo_root is not a git repository or commit hash could not be resolved: {repo_root}")
    if not torch.cuda.is_available():
        fail("CUDA is not visible to torch on this machine.")
    if not checkpoint_dir.exists():
        fail(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_path.exists():
        fail(f"FFHQ256 checkpoint file does not exist: {checkpoint_path}")

    print("[setup-check] torch.cuda is available:", torch.cuda.is_available())
    print("[setup-check] DiffAE git commit:", git_commit_hash(repo_root))
    print("[setup-check] setup looks valid")
    print(json.dumps({"repo_root": str(repo_root), "checkpoint_path": str(checkpoint_path)}, indent=2))


if __name__ == "__main__":
    main()

