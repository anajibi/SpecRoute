from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffae_tools.config_io import ensure_dir, load_config, resolve_path
from diffae_tools.image_io import denormalize_from_diffae, load_image_tensor
from diffae_tools.latent_codec import load_latent_bundle
from diffae_tools.model_loader import DiffAEModelWrapper
from diffae_tools.plotting import save_reconstruction_panel

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


class ReconstructionDataset(Dataset):
    """
    Dataset for reconstruction evaluation.

    This version uses explicit image paths saved during encoding:
        outputs/latents/image_ids.csv

    It does NOT assume aligned_image_dir / image_id.
    """

    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = row["image_path"]

        return {
            "image_id": row["image_id"],
            "image": load_image_tensor(
                image_path,
                image_size=self.image_size,
                normalize=True,
            ),
            "image_path": image_path,
        }


def _compute_ssim(original_01: np.ndarray, recon_01: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim

        return float(
            ssim(
                original_01,
                recon_01,
                channel_axis=2,
                data_range=1.0,
            )
        )
    except Exception:
        return float("nan")


def _make_lpips_metric(device: torch.device):
    try:
        import lpips

        metric = lpips.LPIPS(net="alex").to(device)
        metric.eval()
        return metric
    except Exception as exc:
        LOGGER.warning("LPIPS unavailable; skipping. Reason: %s", exc)
        return None


def _compute_lpips_batch(
    metric,
    device: torch.device,
    original_01: torch.Tensor,
    recon_01: torch.Tensor,
) -> list[float]:
    """
    original_01 and recon_01 are expected in [0, 1].
    LPIPS expects [-1, 1], so we map them inside this function.
    """
    if metric is None:
        return [float("nan")] * original_01.shape[0]

    with torch.inference_mode():
        original_m11 = original_01.to(device, non_blocking=True) * 2.0 - 1.0
        recon_m11 = recon_01.to(device, non_blocking=True) * 2.0 - 1.0
        score = metric(original_m11, recon_m11)

    return score.detach().cpu().view(-1).numpy().astype(float).tolist()


def _build_reconstruction_frame(
    latent_dir: Path,
    bundle,
    cfg: dict,
    config_dir: Path,
) -> pd.DataFrame:
    """
    Prefer the image paths saved by the new encoder.

    Expected:
        outputs/latents/image_ids.csv
        columns: image_id, image_path, aligned_path, already_aligned

    Fallback:
        raw_image_dir / image_id
    Legacy fallback:
        aligned_image_dir / image_id
    """
    image_ids_path = latent_dir / "image_ids.csv"

    if image_ids_path.exists():
        frame = pd.read_csv(image_ids_path)

        if "image_path" not in frame.columns:
            if "aligned_path" in frame.columns:
                LOGGER.warning(
                    "image_ids.csv has no image_path column; using aligned_path as fallback."
                )
                frame["image_path"] = frame["aligned_path"]
            else:
                raise ValueError(
                    f"{image_ids_path} must contain image_path or aligned_path."
                )

        required = {"image_id", "image_path"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{image_ids_path} missing columns: {missing}")

        frame = frame[["image_id", "image_path"]].copy()

    else:
        LOGGER.warning(
            "No %s found. Falling back to config paths. "
            "This is weaker; rerun 02_encode_dataset.py if possible.",
            image_ids_path,
        )

        if "raw_image_dir" in cfg or "image_dir" in cfg:
            raw_image_dir = resolve_path(
                cfg.get("raw_image_dir", cfg.get("image_dir")),
                base_dir=config_dir,
            )
            frame = pd.DataFrame(
                {
                    "image_id": bundle.image_ids,
                    "image_path": [str(raw_image_dir / image_id) for image_id in bundle.image_ids],
                }
            )
        elif "aligned_image_dir" in cfg:
            aligned_image_dir = resolve_path(cfg["aligned_image_dir"], base_dir=config_dir)
            frame = pd.DataFrame(
                {
                    "image_id": bundle.image_ids,
                    "image_path": [str(aligned_image_dir / image_id) for image_id in bundle.image_ids],
                }
            )
        else:
            raise ValueError(
                "Cannot reconstruct image paths. Need outputs/latents/image_ids.csv, "
                "or config raw_image_dir/image_dir/aligned_image_dir."
            )

    # Keep same ordering as the latent bundle.
    latent_ids = list(bundle.image_ids)
    order = pd.DataFrame(
        {
            "image_id": latent_ids,
            "_latent_index": list(range(len(latent_ids))),
        }
    )

    frame = order.merge(frame, on="image_id", how="left")

    missing_paths = frame["image_path"].isna().sum()
    if missing_paths:
        raise ValueError(f"{missing_paths} latent image_ids have no image_path.")

    bad_paths = []
    for p in frame["image_path"]:
        if not Path(p).exists():
            bad_paths.append(p)

    if bad_paths:
        bad_report = latent_dir / "missing_reconstruction_paths.txt"
        with open(bad_report, "w") as f:
            for p in bad_paths:
                f.write(str(p) + "\n")
        raise FileNotFoundError(
            f"{len(bad_paths)} image paths do not exist. "
            f"Saved list to {bad_report}"
        )

    return frame[["image_id", "image_path", "_latent_index"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct raw already-aligned FFHQ256 images from saved DiffAE latents."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--no_lpips",
        action="store_true",
        help="Disable LPIPS computation.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    repo_root = resolve_path(cfg["repo_root"], base_dir=config_dir)
    checkpoint_path = resolve_path(cfg["checkpoint_path"], base_dir=config_dir)

    latent_dir = resolve_path(Path(cfg["output_dir"]) / "latents", base_dir=config_dir)
    output_dir = ensure_dir(
        resolve_path(Path(cfg["output_dir"]) / "reconstructions", base_dir=config_dir)
    )

    image_size = int(cfg.get("image_size", 256))
    batch_size = int(args.batch_size or cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 4))
    requested_device = str(cfg.get("device", "cuda"))

    use_cuda = requested_device.startswith("cuda") and torch.cuda.is_available()
    torch_device = torch.device(requested_device if use_cuda else "cpu")

    LOGGER.info("Requested device: %s", requested_device)
    LOGGER.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    LOGGER.info("Using device: %s", torch_device)

    if torch_device.type == "cuda":
        LOGGER.info("CUDA device name: %s", torch.cuda.get_device_name(torch_device))
        torch.backends.cudnn.benchmark = True

    bundle = load_latent_bundle(latent_dir)

    frame = _build_reconstruction_frame(
        latent_dir=latent_dir,
        bundle=bundle,
        cfg=cfg,
        config_dir=config_dir,
    )

    if args.max_images is not None:
        frame = frame.iloc[: args.max_images].reset_index(drop=True)

    if frame.empty:
        raise SystemExit("[reconstruct] no images to reconstruct.")

    LOGGER.info("Reconstructing %d images", len(frame))

    semantic = torch.as_tensor(bundle.semantic).float()
    stochastic = torch.as_tensor(bundle.stochastic).float()

    wrapper = DiffAEModelWrapper(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        device=str(torch_device),
    ).load()

    dataset = ReconstructionDataset(frame, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )

    lpips_metric = None if args.no_lpips else _make_lpips_metric(torch_device)

    metrics_rows = []

    # Only keep a small number for visualization.
    panel_originals_01 = []
    panel_recons_01 = []
    max_panel_items = 32

    for batch_idx, batch in enumerate(loader):
        batch_originals_m11 = batch["image"]
        bsz = batch_originals_m11.shape[0]

        latent_indices = frame.iloc[
            batch_idx * batch_size : batch_idx * batch_size + bsz
        ]["_latent_index"].to_numpy()

        z_sem = semantic[latent_indices].to(torch_device, non_blocking=use_cuda)
        xT = stochastic[latent_indices].to(torch_device, non_blocking=use_cuda)

        with torch.inference_mode():
            recon_m11 = wrapper.decode_from_latents(z_sem, xT)

        # Move to CPU for metrics and convert both original/recon to [0, 1].
        original_01 = denormalize_from_diffae(batch_originals_m11).detach().cpu()
        recon_01 = recon_m11.detach().cpu().clamp(0.0, 1.0)

        if batch_idx == 0:
            LOGGER.info("original batch device: %s", batch_originals_m11.device)
            LOGGER.info("z_sem device: %s", z_sem.device)
            LOGGER.info("xT device: %s", xT.device)
            LOGGER.info("recon device before CPU conversion: %s", recon_m11.device)

        original_np = original_01.permute(0, 2, 3, 1).numpy()
        recon_np = recon_01.permute(0, 2, 3, 1).numpy()

        lpips_scores = _compute_lpips_batch(
            lpips_metric,
            torch_device,
            original_01,
            recon_01,
        )

        for i in range(bsz):
            mse = float(np.mean((original_np[i] - recon_np[i]) ** 2))
            l1 = float(np.mean(np.abs(original_np[i] - recon_np[i])))
            psnr = float("inf") if mse == 0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))

            metrics_rows.append(
                {
                    "image_id": batch["image_id"][i],
                    "image_path": batch["image_path"][i],
                    "l1": l1,
                    "mse": mse,
                    "l2": mse,
                    "psnr": psnr,
                    "ssim": _compute_ssim(original_np[i], recon_np[i]),
                    "lpips": float(lpips_scores[i]),
                }
            )

        remaining_panel_slots = max_panel_items - len(panel_originals_01)
        if remaining_panel_slots > 0:
            take = min(remaining_panel_slots, bsz)
            panel_originals_01.extend([original_01[i] for i in range(take)])
            panel_recons_01.extend([recon_01[i] for i in range(take)])

        if batch_idx % 25 == 0:
            LOGGER.info("Reconstructed batch %d / %d", batch_idx + 1, len(loader))

    metrics_df = pd.DataFrame(metrics_rows)

    summary = {"image_id": "__mean__", "image_path": ""}
    for col in ["l1", "mse", "l2", "psnr", "ssim", "lpips"]:
        summary[col] = float(metrics_df[col].replace([np.inf, -np.inf], np.nan).mean())

    metrics_df = pd.concat([metrics_df, pd.DataFrame([summary])], ignore_index=True)
    metrics_df.to_csv(output_dir / "reconstruction_metrics.csv", index=False)

    if panel_originals_01 and panel_recons_01:
        originals_panel = torch.stack(panel_originals_01, dim=0)
        recons_panel = torch.stack(panel_recons_01, dim=0)

        save_reconstruction_panel(
            originals_panel,
            recons_panel,
            output_dir / "recon_grid.png",
            max_items=min(max_panel_items, len(panel_originals_01)),
        )

    LOGGER.info("Saved reconstructions and metrics to %s", output_dir)


if __name__ == "__main__":
    main()