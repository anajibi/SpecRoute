from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import subprocess

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    return (base / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit_hash(repo_root: str | Path) -> str | None:
    repo_root = Path(repo_root)
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def add_repo_to_path(repo_root: str | Path):
    import sys

    repo_root = str(Path(repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

