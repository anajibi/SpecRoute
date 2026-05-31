from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffae_tools.config_io import ensure_dir, git_commit_hash, load_config, resolve_path
from diffae_tools.image_io import load_image_tensor
from diffae_tools.latent_codec import save_latent_bundle
from diffae_tools.model_loader import DiffAEModelWrapper

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def make_image_id(path: Path, base_dir: Path) -> str:
    """
    Create a stable image_id from a raw image path.

    Example:
        raw_images/ffhq256/00001.png -> 00001.png
        raw_images/ffhq256/folder/00001.png -> folder__00001.png
    """
    rel = path.relative_to(base_dir)
    return "__".join(rel.parts)


def is_valid_image(path: Path, min_bytes: int = 1024) -> bool:
    """
    Reject corrupted, empty, or truncated images before DataLoader starts.

    This avoids a single bad image crashing encoding inside a worker process.
    """
    try:
        if not path.exists():
            return False

        if path.stat().st_size < min_bytes:
            return False

        with Image.open(path) as img:
            img.verify()

        return True
    except Exception:
        return False


def list_raw_images(raw_image_dir: Path, recursive: bool = True) -> list[Path]:
    if recursive:
        candidates = raw_image_dir.rglob("*")
    else:
        candidates = raw_image_dir.glob("*")

    image_files = sorted(
        p for p in candidates
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    return image_files


class RawImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        image = load_image_tensor(
            row["image_path"],
            image_size=self.image_size,
            normalize=True,
        )

        return {
            "image": image,
            "image_id": row["image_id"],
            "image_path": row["image_path"],
        }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Encode already-aligned raw FFHQ256 images with DiffAE into "
            "semantic and stochastic latents."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Only search the top level of raw_image_dir.",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip image validity checks before encoding.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    repo_root = resolve_path(cfg["repo_root"], base_dir=config_dir)
    checkpoint_path = resolve_path(cfg["checkpoint_path"], base_dir=config_dir)

    # IMPORTANT:
    # This script assumes images are already aligned.
    # It does NOT use aligned_image_dir or metadata.csv.
    raw_image_dir = resolve_path(
        cfg.get("raw_image_dir", cfg.get("image_dir")),
        base_dir=config_dir,
    )

    output_dir = ensure_dir(
        resolve_path(Path(cfg["output_dir"]) / "latents", base_dir=config_dir)
    )

    image_size = int(cfg.get("image_size", 256))
    batch_size = int(args.batch_size or cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 4))
    device = cfg.get("device", "cuda")

    if raw_image_dir is None:
        raise SystemExit(
            "[encode] config must contain either raw_image_dir or image_dir."
        )

    if not raw_image_dir.exists():
        raise SystemExit(f"[encode] raw image directory does not exist: {raw_image_dir}")

    image_files = list_raw_images(
        raw_image_dir,
        recursive=not args.no_recursive,
    )

    if not image_files:
        raise SystemExit(f"[encode] no raw images found in {raw_image_dir}")

    LOGGER.info("Found %d candidate raw images in %s", len(image_files), raw_image_dir)

    rows = []
    bad_rows = []

    for p in image_files:
        image_id = make_image_id(p, raw_image_dir)

        row = {
            "image_id": image_id,
            "image_path": str(p),
            "original_path": str(p),
            "already_aligned": True,
            "strategy": "raw_ffhq256_already_aligned",
        }

        if args.skip_validation or is_valid_image(p):
            rows.append(row)
        else:
            bad_rows.append(row)

    if bad_rows:
        bad_df = pd.DataFrame(bad_rows)
        bad_path = output_dir / "bad_images.csv"
        bad_df.to_csv(bad_path, index=False)
        LOGGER.warning(
            "Skipped %d invalid/corrupted images. Saved list to %s",
            len(bad_df),
            bad_path,
        )

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise SystemExit("[encode] all candidate images were invalid.")

    frame = frame.sort_values("image_id").reset_index(drop=True)

    if args.max_images is not None:
        frame = frame.iloc[: args.max_images].reset_index(drop=True)

    LOGGER.info("Encoding %d images", len(frame))

    dataset = RawImageDataset(frame, image_size=image_size)

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    torch_device = torch.device(device if use_cuda else "cpu")

    LOGGER.info("Requested device from config: %s", device)
    LOGGER.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    LOGGER.info("Using torch device: %s", torch_device)

    if torch_device.type == "cuda":
        LOGGER.info("CUDA device name: %s", torch.cuda.get_device_name(torch_device))
        torch.backends.cudnn.benchmark = True


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )

    wrapper = DiffAEModelWrapper(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        device=str(torch_device),
    ).load()

    semantic_chunks = []
    stochastic_chunks = []
    image_ids = []
    image_paths = []

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using torch device: %s", torch_device)

    if torch_device.type == "cuda":
        LOGGER.info("CUDA device name: %s", torch.cuda.get_device_name(torch_device))
        torch.backends.cudnn.benchmark = True

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(
            torch_device,
            non_blocking=use_cuda,
        )

        with torch.inference_mode():
            z_sem = wrapper.encode_semantic(images, return_cpu=False)
            stochastic = wrapper.encode_stochastic(images, z_sem=z_sem, return_cpu=False)

        if batch_idx == 0:
            LOGGER.info("Input image tensor device: %s", images.device)
            LOGGER.info("z_sem device before saving: %s", z_sem.device)
            LOGGER.info("stochastic device before saving: %s", stochastic.device)

        semantic_chunks.append(z_sem.detach().cpu().numpy())
        stochastic_chunks.append(stochastic.detach().cpu().numpy())

        image_ids.extend(batch["image_id"])
        image_paths.extend(batch["image_path"])

        if batch_idx % 25 == 0:
            LOGGER.info("Encoded batch %d / %d", batch_idx + 1, len(loader))

    z_sem = np.concatenate(semantic_chunks, axis=0)
    stochastic = np.concatenate(stochastic_chunks, axis=0)

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "repo_root": str(repo_root),
        "git_commit": git_commit_hash(repo_root),
        "raw_image_dir": str(raw_image_dir),
        "image_size": image_size,
        "num_images": int(len(image_ids)),
        "z_sem_shape": list(z_sem.shape),
        "stochastic_shape": list(stochastic.shape),
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "config_path": str(Path(args.config).resolve()),
        "already_aligned": True,
        "uses_aligned_image_dir": False,
    }

    save_latent_bundle(
        output_dir,
        z_sem,
        stochastic,
        image_ids,
        metadata=metadata,
    )

    pd.DataFrame(
        {
            "image_id": image_ids,
            "image_path": image_paths,
            # Kept for backward compatibility with later scripts.
            "aligned_path": image_paths,
            "already_aligned": True,
        }
    ).to_csv(output_dir / "image_ids.csv", index=False)

    if args.save_images:
        example_count = min(16, len(dataset))
        previews = torch.stack([dataset[i]["image"] for i in range(example_count)])

        from torchvision.utils import make_grid, save_image

        grid = make_grid(
            previews,
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )
        save_image(grid, output_dir / "raw_ffhq256_preview.png")

    LOGGER.info("Saved semantic/stochastic latents to %s", output_dir)


if __name__ == "__main__":
    main()