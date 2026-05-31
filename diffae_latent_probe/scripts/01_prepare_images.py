from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from diffae_tools.config_io import ensure_dir, load_config, resolve_path
from diffae_tools.image_io import align_image_with_fallback, list_image_files, sanitize_image_id

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare aligned FFHQ-style images for DiffAE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_dir", default=None, help="Override config.image_dir")
    parser.add_argument("--output_dir", default=None, help="Override config.aligned_image_dir")
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no_official_align", action="store_true", help="Disable official dlib-based alignment")
    parser.add_argument("--output_size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent
    repo_root = resolve_path(cfg["repo_root"], base_dir=config_dir)
    input_dir = resolve_path(args.input_dir or cfg["image_dir"], base_dir=config_dir)
    output_dir = ensure_dir(resolve_path(args.output_dir or cfg["aligned_image_dir"], base_dir=config_dir))

    if not input_dir.exists():
        raise SystemExit(f"[prepare-images] input directory does not exist: {input_dir}")

    rows = []
    image_files = list(list_image_files(input_dir, recursive=args.recursive))
    if not image_files:
        raise SystemExit(f"[prepare-images] no images found in: {input_dir}")

    LOGGER.info("Preparing %d images -> %s", len(image_files), output_dir)
    for src_path in image_files:
        image_id = sanitize_image_id(src_path, base_dir=input_dir)
        dst_path = output_dir / image_id
        result = align_image_with_fallback(
            src_path,
            dst_path,
            repo_root=repo_root,
            output_size=args.output_size,
            prefer_official=not args.no_official_align,
        )
        rows.append(
            {
                "image_id": image_id,
                "original_path": str(src_path),
                "aligned_path": str(dst_path),
                "success": bool(result.success),
                "failure_reason": result.failure_reason or "",
                "strategy": result.strategy,
            }
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    LOGGER.info("Saved metadata.csv with %d rows", len(metadata))


if __name__ == "__main__":
    main()

