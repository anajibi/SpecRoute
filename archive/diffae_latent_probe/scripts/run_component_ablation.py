from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffae_tools.config_io import ensure_dir, load_config, resolve_path
from diffae_tools.image_io import (
    denormalize_from_diffae,
    load_image_tensor,
    save_tensor_image,
    tensor_to_pil,
)
from diffae_tools.latent_codec import load_latent_bundle
from diffae_tools.model_loader import DiffAEModelWrapper

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


class AblationDataset(Dataset):
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
            "image": load_image_tensor(image_path, image_size=self.image_size, normalize=True),
            "image_path": image_path,
        }


def _safe_id(image_id: str) -> str:
    return image_id.replace(os.sep, "__")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/Library/Fonts/Arial.ttf", size)
        except OSError:
            return ImageFont.load_default()


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


def _compute_lpips_batch(metric, device: torch.device, original_01: torch.Tensor, recon_01: torch.Tensor) -> list[float]:
    if metric is None:
        return [float("nan")] * original_01.shape[0]

    with torch.inference_mode():
        original_m11 = original_01.to(device, non_blocking=True) * 2.0 - 1.0
        recon_m11 = recon_01.to(device, non_blocking=True) * 2.0 - 1.0
        score = metric(original_m11, recon_m11)

    return score.detach().cpu().view(-1).numpy().astype(float).tolist()


def _gaussian_blur(batch: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return batch
    radius = max(1, int(3 * sigma + 0.5))
    size = radius * 2 + 1
    device = batch.device
    dtype = batch.dtype
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, size, size)
    kernel_2d = kernel_2d.repeat(batch.shape[1], 1, 1, 1)
    batch_pad = F.pad(batch, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(batch_pad, kernel_2d, padding=0, groups=batch.shape[1])


def _highpass(batch_01: torch.Tensor, sigma: float) -> torch.Tensor:
    blurred = _gaussian_blur(batch_01, sigma=sigma)
    return batch_01 - blurred


def _vis_residual(x: torch.Tensor, scale: float = 4.0) -> torch.Tensor:
    return (0.5 + scale * x).clamp(0.0, 1.0)


def _normalize_recon_01(recon: torch.Tensor) -> torch.Tensor:
    recon_cpu = recon.detach().cpu()
    if recon_cpu.min() < -0.1:
        recon_01 = denormalize_from_diffae(recon_cpu)
    else:
        recon_01 = recon_cpu
    return recon_01.clamp(0.0, 1.0)


def _save_image_01(tensor_01: torch.Tensor, path: Path) -> None:
    save_tensor_image(tensor_01, path, denormalize=False)


def _make_grid_canvas(
    grid_tensor: torch.Tensor,
    col_labels: list[str],
    image_size: int,
    padding: int,
    row_labels: list[str] | None = None,
    title: str | None = None,
) -> Image.Image:
    grid_pil = tensor_to_pil(grid_tensor, denormalize=False)
    font = _get_font(24)
    header_height = 60
    left_margin = 220 if row_labels else 20
    top_margin = header_height + (40 if title else 0)

    canvas = Image.new(
        "RGB",
        (grid_pil.width + left_margin, grid_pil.height + top_margin),
        (255, 255, 255),
    )
    canvas.paste(grid_pil, (left_margin, top_margin))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((left_margin + grid_pil.width // 2, 20), title, fill=(0, 0, 0), font=font, anchor="mm")

    for i, label in enumerate(col_labels):
        x_center = left_margin + padding + (image_size + padding) * i + image_size // 2
        draw.text((x_center, top_margin - 30), label, fill=(0, 0, 0), font=font, anchor="mm", align="center")

    if row_labels:
        for i, label in enumerate(row_labels):
            y_center = top_margin + padding + (image_size + padding) * i + image_size // 2
            draw.text((left_margin // 2, y_center), label, fill=(0, 0, 0), font=font, anchor="mm", align="center")

    return canvas


def _save_labeled_grid(
    images_01: list[torch.Tensor],
    col_labels: list[str],
    out_path: Path,
    image_size: int,
    padding: int = 4,
    row_labels: list[str] | None = None,
    title: str | None = None,
    nrow: int | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = [img.detach().cpu().float().clamp(0.0, 1.0) for img in images_01]
    nrow = nrow or len(col_labels)
    grid_tensor = make_grid(torch.stack(tensors), nrow=nrow, padding=padding, normalize=False, value_range=(0, 1))
    canvas = _make_grid_canvas(
        grid_tensor,
        col_labels=col_labels,
        image_size=image_size,
        padding=padding,
        row_labels=row_labels,
        title=title,
    )
    canvas.save(out_path)


def _build_reconstruction_frame(latent_dir: Path, bundle, cfg: dict, config_dir: Path) -> pd.DataFrame:
    image_ids_path = latent_dir / "image_ids.csv"

    if image_ids_path.exists():
        frame = pd.read_csv(image_ids_path)

        if "image_path" not in frame.columns:
            if "aligned_path" in frame.columns:
                LOGGER.warning("image_ids.csv has no image_path column; using aligned_path as fallback.")
                frame["image_path"] = frame["aligned_path"]
            else:
                raise ValueError(f"{image_ids_path} must contain image_path or aligned_path.")

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
            raw_image_dir = resolve_path(cfg.get("raw_image_dir", cfg.get("image_dir")), base_dir=config_dir)
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

    latent_ids = list(bundle.image_ids)
    order = pd.DataFrame({"image_id": latent_ids, "_latent_index": list(range(len(latent_ids)))})
    frame = order.merge(frame, on="image_id", how="left")

    missing_paths = frame["image_path"].isna().sum()
    if missing_paths:
        raise ValueError(f"{missing_paths} latent image_ids have no image_path.")

    bad_paths = [p for p in frame["image_path"] if not Path(p).exists()]
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


def _decode_in_chunks(wrapper: DiffAEModelWrapper, z: torch.Tensor, x_t: torch.Tensor, chunk_size: int) -> torch.Tensor:
    outputs = []
    for start in range(0, z.shape[0], chunk_size):
        end = min(start + chunk_size, z.shape[0])
        outputs.append(wrapper.decode_from_latents(z[start:end], x_t[start:end]))
    return torch.cat(outputs, dim=0)


def _make_perm(num_items: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_items).tolist()
    fixed = []
    for i, j in enumerate(perm):
        if j == i:
            j = perm[(i + 1) % num_items]
        fixed.append(int(j))
    return fixed


def _load_original_01(path: str, image_size: int) -> torch.Tensor:
    img_m11 = load_image_tensor(path, image_size=image_size, normalize=True)
    img_01 = denormalize_from_diffae(img_m11.unsqueeze(0))[0]
    return img_01.clamp(0.0, 1.0)


def _compute_metrics_row(
    original_01: torch.Tensor,
    recon_01: torch.Tensor,
    lpips_metric,
    device: torch.device,
    compute_lpips: bool,
    compute_ssim: bool,
    compute_mse: bool,
    compute_arcface: bool,
    compute_attributes: bool,
    highpass_sigma: float,
) -> dict:
    original_np = original_01.permute(1, 2, 0).numpy()
    recon_np = recon_01.permute(1, 2, 0).numpy()

    metrics = {
        "lpips": float("nan"),
        "ssim": float("nan"),
        "mse": float("nan"),
        "arcface_sim": float("nan"),
        "attribute_consistency": float("nan"),
        "hf_mse": float("nan"),
        "hf_l1": float("nan"),
    }

    if compute_mse:
        metrics["mse"] = float(np.mean((original_np - recon_np) ** 2))

    if compute_ssim:
        metrics["ssim"] = _compute_ssim(original_np, recon_np)

    if compute_lpips:
        lpips_score = _compute_lpips_batch(lpips_metric, device, original_01.unsqueeze(0), recon_01.unsqueeze(0))[0]
        metrics["lpips"] = float(lpips_score)

    if compute_arcface:
        LOGGER.warning("ArcFace requested, but no ArcFace integration is available. Using NaN values.")

    if compute_attributes:
        LOGGER.warning("Attribute consistency requested, but no attribute model is available. Using NaN values.")

    if highpass_sigma > 0:
        orig_hp = _highpass(original_01.unsqueeze(0), sigma=highpass_sigma)[0]
        recon_hp = _highpass(recon_01.unsqueeze(0), sigma=highpass_sigma)[0]
        diff = orig_hp - recon_hp
        metrics["hf_mse"] = float(torch.mean(diff ** 2).item())
        metrics["hf_l1"] = float(torch.mean(diff.abs()).item())

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffAE component ablation experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/component_ablation")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_xt_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute_lpips", action="store_true", default=True)
    parser.add_argument("--compute_ssim", action="store_true", default=True)
    parser.add_argument("--compute_mse", action="store_true", default=True)
    parser.add_argument("--compute_arcface", action="store_true", default=False)
    parser.add_argument("--compute_attributes", action="store_true", default=False)
    parser.add_argument("--save_individual_images", action="store_true", default=False)
    parser.add_argument("--save_grids", action="store_true", default=True)
    parser.add_argument("--highpass_method", default="gaussian_residual")
    parser.add_argument("--highpass_sigma", type=float, default=2.0)
    args = parser.parse_args()

    _set_seed(args.seed)
    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    repo_root = resolve_path(cfg["repo_root"], base_dir=config_dir)
    checkpoint_path = resolve_path(cfg["checkpoint_path"], base_dir=config_dir)
    latent_dir = resolve_path(Path(cfg["output_dir"]) / "latents", base_dir=config_dir)
    output_dir = resolve_path(args.output_dir, base_dir=config_dir)

    image_size = int(cfg.get("image_size", 256))
    num_workers = int(cfg.get("num_workers", 4))

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    torch_device = torch.device(args.device if use_cuda else "cpu")

    LOGGER.info("Requested device: %s", args.device)
    LOGGER.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    LOGGER.info("Using device: %s", torch_device)

    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    bundle = load_latent_bundle(latent_dir)
    frame = _build_reconstruction_frame(latent_dir, bundle, cfg, config_dir)
    if args.num_images is not None:
        frame = frame.iloc[: args.num_images].reset_index(drop=True)

    if frame.empty:
        raise SystemExit("[component_ablation] no images to process.")
    if len(frame) < 2:
        raise SystemExit("[component_ablation] need at least two images for swap/mismatch experiments.")
    if args.num_xt_samples < 1:
        raise SystemExit("[component_ablation] num_xt_samples must be >= 1.")

    z_sem_all = torch.as_tensor(bundle.semantic).float()
    x_t_all = torch.as_tensor(bundle.stochastic).float()

    # Mean semantic code over the images included in this experiment.
    # This implements z_mean = avg_i z_i.
    selected_latent_indices = frame["_latent_index"].to_numpy()
    z_mean_cpu = z_sem_all[selected_latent_indices].mean(dim=0, keepdim=True)

    wrapper = DiffAEModelWrapper(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        device=str(torch_device),
    ).load()

    dataset = AblationDataset(frame, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )

    output_root = ensure_dir(output_dir)
    outputs = {
        "full": ensure_dir(output_root / "full_recon"),
        "z_only": ensure_dir(output_root / "z_only"),
        "z_only_marginal": ensure_dir(output_root / "z_only_marginal"),
        "xt_mean": ensure_dir(output_root / "xt_only_mean"),
        "xt_zero": ensure_dir(output_root / "xt_only_zero"),
        "xt_mismatch": ensure_dir(output_root / "xt_only_mismatch"),
        "grids": ensure_dir(output_root / "grids"),
        "results": ensure_dir(output_root / "results"),
    }

    perm = _make_perm(len(frame), args.seed)

    pair_rows = []
    for i, j in enumerate(perm):
        pair_rows.append(
            {
                "image_id": frame.iloc[i]["image_id"],
                "image_path": frame.iloc[i]["image_path"],
                "donor_id": frame.iloc[j]["image_id"],
                "donor_path": frame.iloc[j]["image_path"],
            }
        )

    pd.DataFrame(pair_rows).to_csv(outputs["results"] / "pairings.csv", index=False)

    config_payload = {
        "config": cfg,
        "args": vars(args),
        "num_images": len(frame),
    }
    (outputs["results"] / "config.json").write_text(json.dumps(config_payload, indent=2))

    if args.highpass_method != "gaussian_residual":
        LOGGER.warning("Unsupported highpass_method=%s; using gaussian_residual.", args.highpass_method)

    lpips_metric = _make_lpips_metric(torch_device) if args.compute_lpips else None

    metrics_rows = []
    grid_cache = {}
    donor_cache = {}

    torch_gen = torch.Generator(device=torch_device)
    torch_gen.manual_seed(args.seed)



    for batch_idx, batch in enumerate(loader):
        batch_originals_m11 = batch["image"]
        bsz = batch_originals_m11.shape[0]
        start = batch_idx * args.batch_size
        end = start + bsz
        frame_indices = list(range(start, end))

        latent_indices = frame.iloc[start:end]["_latent_index"].to_numpy()
        z_sem = z_sem_all[latent_indices].to(torch_device, non_blocking=use_cuda)
        x_t = x_t_all[latent_indices].to(torch_device, non_blocking=use_cuda)

        # Batch version of z_mean for D(z_mean, xT_i).
        z_mean_batch = z_mean_cpu.to(torch_device, non_blocking=use_cuda).expand_as(z_sem)

        donor_frame_indices = [perm[i] for i in frame_indices]
        donor_latent_indices = frame.iloc[donor_frame_indices]["_latent_index"].to_numpy()
        z_donor = z_sem_all[donor_latent_indices].to(torch_device, non_blocking=use_cuda)

        with torch.inference_mode():
            recon_full = wrapper.decode_from_latents(z_sem, x_t)

            eps_single = torch.randn(x_t.shape, device=x_t.device, generator=torch_gen)
            recon_z_only = wrapper.decode_from_latents(z_sem, eps_single)

            # Correct xT-mean condition:
            # D(z_mean, xT_i)
            recon_xt_mean = wrapper.decode_from_latents(z_mean_batch, x_t)

            # Other xT-only probes:
            # D(0, xT_i) and D(z_j, xT_i)
            recon_xt_zero = wrapper.decode_from_latents(torch.zeros_like(z_sem), x_t)
            recon_xt_mismatch = wrapper.decode_from_latents(z_donor, x_t)

        recon_xt_mean_01 = _normalize_recon_01(recon_xt_mean)

        original_01 = denormalize_from_diffae(batch_originals_m11).detach().cpu().clamp(0.0, 1.0)
        recon_full_01 = _normalize_recon_01(recon_full)
        recon_z_only_01 = _normalize_recon_01(recon_z_only)
        recon_xt_zero_01 = _normalize_recon_01(recon_xt_zero)
        recon_xt_mismatch_01 = _normalize_recon_01(recon_xt_mismatch)

        z_rep = z_sem.repeat_interleave(args.num_xt_samples, dim=0)
        x_t_rep = x_t.repeat_interleave(args.num_xt_samples, dim=0)
        eps_marginal = torch.randn(x_t_rep.shape, device=x_t_rep.device, generator=torch_gen)
        recon_z_only_marginal = _decode_in_chunks(wrapper, z_rep, eps_marginal, args.batch_size)
        recon_z_only_marginal_01 = _normalize_recon_01(recon_z_only_marginal)
        recon_z_only_marginal_avg_01 = recon_z_only_marginal_01.view(
            bsz,
            args.num_xt_samples,
            recon_z_only_marginal_01.shape[1],
            recon_z_only_marginal_01.shape[2],
            recon_z_only_marginal_01.shape[3],
        ).mean(dim=1)

        for i in range(bsz):
            image_id = batch["image_id"][i]
            safe_id = _safe_id(image_id)
            donor_idx = donor_frame_indices[i]
            donor_id = frame.iloc[donor_idx]["image_id"]
            donor_safe = _safe_id(donor_id)

            if donor_id not in donor_cache:
                donor_cache[donor_id] = _load_original_01(
                    frame.iloc[donor_idx]["image_path"], image_size=image_size
                )
            donor_img = donor_cache[donor_id]

            input_01 = original_01[i]

            if args.save_individual_images:
                # Skip per-image full recon outputs; only keep the grid for full reconstruction.
                _save_image_01(input_01, outputs["z_only"] / f"{safe_id}_input.png")
                _save_image_01(recon_z_only_01[i], outputs["z_only"] / f"{safe_id}_zonly.png")
                _save_image_01(input_01, outputs["xt_mean"] / f"{safe_id}_input.png")
                _save_image_01(recon_xt_mean_01[i], outputs["xt_mean"] / f"{safe_id}_xtonly_mean.png")
                _save_image_01(input_01, outputs["xt_zero"] / f"{safe_id}_input.png")
                _save_image_01(recon_xt_zero_01[i], outputs["xt_zero"] / f"{safe_id}_xtonly_zero.png")
                _save_image_01(input_01, outputs["xt_mismatch"] / f"{safe_id}_input.png")
                _save_image_01(
                    recon_xt_mismatch_01[i],
                    outputs["xt_mismatch"] / f"{safe_id}_z_from_{donor_safe}_xT_from_{safe_id}.png",
                )
                _save_image_01(donor_img, outputs["xt_mismatch"] / f"{safe_id}_semantic_donor_{donor_safe}.png")

            if args.save_individual_images:
                z_marg_dir = ensure_dir(outputs["z_only_marginal"] / safe_id)
                for k in range(args.num_xt_samples):
                    idx = i * args.num_xt_samples + k
                    _save_image_01(recon_z_only_marginal_01[idx], z_marg_dir / f"sample_{k + 1}.png")

            if args.save_grids:
                _save_labeled_grid(
                    [input_01, recon_full_01[i]],
                    ["Input", "Full"],
                    outputs["grids"] / f"full_recon_{safe_id}.png",
                    image_size=image_size,
                )
                _save_labeled_grid(
                    [input_01, recon_z_only_01[i]],
                    ["Input", "z-only"],
                    outputs["grids"] / f"z_only_{safe_id}.png",
                    image_size=image_size,
                )
                _save_labeled_grid(
                    [input_01, recon_xt_mean_01[i]],
                    ["Input", "xT-only mean"],
                    outputs["grids"] / f"xt_only_mean_{safe_id}.png",
                    image_size=image_size,
                )
                _save_labeled_grid(
                    [input_01, recon_xt_zero_01[i]],
                    ["Input", "xT-only zero"],
                    outputs["grids"] / f"xt_only_zero_{safe_id}.png",
                    image_size=image_size,
                )
                _save_labeled_grid(
                    [input_01, recon_xt_mismatch_01[i], donor_img],
                    ["Input", "xT-only mismatch", f"Semantic donor {donor_safe}"],
                    outputs["grids"] / f"xt_only_mismatch_{safe_id}.png",
                    image_size=image_size,
                )

                z_marg_images = [input_01]
                z_marg_labels = ["Input"]
                for k in range(args.num_xt_samples):
                    idx = i * args.num_xt_samples + k
                    z_marg_images.append(recon_z_only_marginal_01[idx])
                    z_marg_labels.append(f"Sample {k + 1}")
                _save_labeled_grid(
                    z_marg_images,
                    z_marg_labels,
                    outputs["grids"] / f"z_only_marginal_{safe_id}.png",
                    image_size=image_size,
                )

                z_marg_avg_hp_full = _highpass(recon_full_01[i].unsqueeze(0), sigma=args.highpass_sigma)[0]
                z_marg_avg_hp_avg = _highpass(
                    recon_z_only_marginal_avg_01[i].unsqueeze(0), sigma=args.highpass_sigma
                )[0]
                z_marg_avg_hp_diff = z_marg_avg_hp_full - z_marg_avg_hp_avg
                _save_labeled_grid(
                    [
                        recon_full_01[i],
                        recon_z_only_marginal_avg_01[i],
                        _vis_residual(z_marg_avg_hp_full),
                        _vis_residual(z_marg_avg_hp_avg),
                        _vis_residual(z_marg_avg_hp_diff),
                    ],
                    [
                        "Full",
                        "Avg z-only marginal",
                        "H(Full)",
                        "H(Avg)",
                        "H(Full)-H(Avg)",
                    ],
                    outputs["grids"] / f"z_only_marginal_avg_vs_full_{safe_id}.png",
                    image_size=image_size,
                )

            if args.save_grids:
                orig_hp = _highpass(input_01.unsqueeze(0), sigma=args.highpass_sigma)[0]
                full_hp = _highpass(recon_full_01[i].unsqueeze(0), sigma=args.highpass_sigma)[0]
                z_hp = _highpass(recon_z_only_01[i].unsqueeze(0), sigma=args.highpass_sigma)[0]
                xt_hp = _highpass(recon_xt_mean_01[i].unsqueeze(0), sigma=args.highpass_sigma)[0]
                full_hp_diff = orig_hp - full_hp
                z_hp_diff = orig_hp - z_hp
                xt_hp_diff = orig_hp - xt_hp
                _save_labeled_grid(
                    [
                        input_01,
                        orig_hp,
                        recon_full_01[i],
                        full_hp,
                        full_hp_diff,
                        recon_z_only_01[i],
                        z_hp,
                        z_hp_diff,
                        recon_xt_mean_01[i],
                        xt_hp,
                        xt_hp_diff,
                    ],
                    [
                        "Original",
                        "H(Original)",
                        "Full",
                        "H(Full)",
                        "H(Orig)-H(Full)",
                        "z-only",
                        "H(z-only)",
                        "H(Orig)-H(z-only)",
                        "xT-mean",
                        "H(xT-mean)",
                        "H(Orig)-H(xT-mean)",
                    ],
                    outputs["grids"] / f"high_frequency_{safe_id}.png",
                    image_size=image_size,
                )

            for setting, recon_01 in [
                ("full", recon_full_01[i]),
                ("z_only", recon_z_only_01[i]),
                ("xt_only_mean", recon_xt_mean_01[i]),
                ("xt_only_zero", recon_xt_zero_01[i]),
                ("xt_only_mismatch", recon_xt_mismatch_01[i]),
            ]:
                metrics = _compute_metrics_row(
                    input_01,
                    recon_01,
                    lpips_metric,
                    torch_device,
                    compute_lpips=args.compute_lpips,
                    compute_ssim=args.compute_ssim,
                    compute_mse=args.compute_mse,
                    compute_arcface=args.compute_arcface,
                    compute_attributes=args.compute_attributes,
                    highpass_sigma=args.highpass_sigma,
                )
                semantic_source = image_id
                xt_source = image_id
                if setting == "z_only":
                    xt_source = "epsilon"
                elif setting == "xt_only_mean":
                    semantic_source = "__mean__"
                elif setting == "xt_only_zero":
                    semantic_source = "__zero__"
                elif setting == "xt_only_mismatch":
                    semantic_source = donor_id

                metrics_rows.append(
                    {
                        "image_id": image_id,
                        "setting": setting,
                        "semantic_source_id": semantic_source,
                        "xT_source_id": xt_source,
                        "sample_id": 0,
                        **metrics,
                    }
                )

            for k in range(args.num_xt_samples):
                idx = i * args.num_xt_samples + k
                metrics = _compute_metrics_row(
                    input_01,
                    recon_z_only_marginal_01[idx],
                    lpips_metric,
                    torch_device,
                    compute_lpips=args.compute_lpips,
                    compute_ssim=args.compute_ssim,
                    compute_mse=args.compute_mse,
                    compute_arcface=args.compute_arcface,
                    compute_attributes=args.compute_attributes,
                    highpass_sigma=args.highpass_sigma,
                )
                metrics_rows.append(
                    {
                        "image_id": image_id,
                        "setting": "z_only_marginal",
                        "semantic_source_id": image_id,
                        "xT_source_id": "epsilon",
                        "sample_id": k + 1,
                        **metrics,
                    }
                )

            if image_id not in grid_cache:
                grid_cache[image_id] = {
                    "input": input_01,
                    "full": recon_full_01[i],
                    "z_only": recon_z_only_01[i],
                    "z_only_marginal": [
                        recon_z_only_marginal_01[i * args.num_xt_samples + k]
                        for k in range(args.num_xt_samples)
                    ],
                    "z_only_marginal_avg": recon_z_only_marginal_avg_01[i],
                    "xt_mean": recon_xt_mean_01[i],
                    "xt_zero": recon_xt_zero_01[i],
                    "xt_mismatch": recon_xt_mismatch_01[i],
                    "donor": donor_img,
                    "donor_id": donor_id,
                }

        if batch_idx % 10 == 0:
            LOGGER.info("Processed batch %d / %d", batch_idx + 1, len(loader))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = outputs["results"] / "component_ablation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary_rows = []
    for setting, group in metrics_df.groupby("setting"):
        row = {"setting": setting}
        for col in ["lpips", "ssim", "mse", "arcface_sim", "attribute_consistency", "hf_mse", "hf_l1"]:
            values = group[col].replace([np.inf, -np.inf], np.nan)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outputs["results"] / "component_ablation_summary.csv", index=False)

    if args.save_grids and grid_cache:
        selected_ids = list(grid_cache.keys())[: min(8, len(grid_cache))]
        main_cols = [
            "Original",
            "Full",
            "z-only",
            *[f"z-only s{k + 1}" for k in range(args.num_xt_samples)],
            "xT-mean",
            "xT-zero",
            "xT-mismatch",
            "Semantic donor",
        ]

        main_images = []
        row_labels = []
        for image_id in selected_ids:
            entry = grid_cache[image_id]
            row_labels.append(_safe_id(image_id)[:18])
            main_images.extend(
                [
                    entry["input"],
                    entry["full"],
                    entry["z_only"],
                    *entry["z_only_marginal"],
                    entry["xt_mean"],
                    entry["xt_zero"],
                    entry["xt_mismatch"],
                    entry["donor"],
                ]
            )

        _save_labeled_grid(
            main_images,
            main_cols,
            outputs["grids"] / "main_attribution_grid.png",
            image_size=image_size,
            row_labels=row_labels,
            title="Component Ablation Grid",
            nrow=len(main_cols),
        )

        fixed_id = selected_ids[0]
        fixed_entry = grid_cache[fixed_id]
        fixed_frame_idx = int(frame.index[frame["image_id"] == fixed_id][0])
        fixed_latent_idx = int(frame.iloc[fixed_frame_idx]["_latent_index"])
        z_fixed = z_sem_all[fixed_latent_idx : fixed_latent_idx + 1].to(torch_device)
        x_t_fixed = x_t_all[fixed_latent_idx : fixed_latent_idx + 1].to(torch_device)

        rng = np.random.RandomState(args.seed)
        donor_candidates = [idx for idx in range(len(frame)) if frame.iloc[idx]["image_id"] != fixed_id]
        rng.shuffle(donor_candidates)
        donor_candidates = donor_candidates[:3]

        fixed_z_images = [fixed_entry["input"], fixed_entry["full"]]
        fixed_z_labels = ["Original", "Full"]
        for donor_idx in donor_candidates[:2]:
            donor_latent = int(frame.iloc[donor_idx]["_latent_index"])
            donor_id = frame.iloc[donor_idx]["image_id"]
            x_t_d = x_t_all[donor_latent : donor_latent + 1].to(torch_device)
            with torch.inference_mode():
                swap_xt = wrapper.decode_from_latents(z_fixed, x_t_d)
            fixed_z_images.append(_normalize_recon_01(swap_xt)[0])
            fixed_z_labels.append(f"xT donor {_safe_id(donor_id)[:10]}")

        eps_samples = []
        for _ in range(3):
            eps = torch.randn(x_t_fixed.shape, device=x_t_fixed.device, generator=torch_gen)
            with torch.inference_mode():
                eps_recon = wrapper.decode_from_latents(z_fixed, eps)
            eps_samples.append(_normalize_recon_01(eps_recon)[0])
        fixed_z_images.extend(eps_samples)
        fixed_z_labels.extend(["epsilon 1", "epsilon 2", "epsilon 3"])

        _save_labeled_grid(
            fixed_z_images,
            fixed_z_labels,
            outputs["grids"] / "fixed_z_vary_xt_grid.png",
            image_size=image_size,
            title="Fixed z_sem, varying x_T",
        )

        fixed_xt_images = [fixed_entry["input"], fixed_entry["full"], fixed_entry["xt_mean"], fixed_entry["xt_zero"]]
        fixed_xt_labels = ["Original", "Full", "z_mean", "z_zero"]
        for donor_idx in donor_candidates:
            donor_latent = int(frame.iloc[donor_idx]["_latent_index"])
            donor_id = frame.iloc[donor_idx]["image_id"]
            z_d = z_sem_all[donor_latent : donor_latent + 1].to(torch_device)
            with torch.inference_mode():
                recon = wrapper.decode_from_latents(z_d, x_t_fixed)
            fixed_xt_images.append(_normalize_recon_01(recon)[0])
            fixed_xt_labels.append(f"z donor {_safe_id(donor_id)[:10]}")

        _save_labeled_grid(
            fixed_xt_images,
            fixed_xt_labels,
            outputs["grids"] / "fixed_xt_vary_z_grid.png",
            image_size=image_size,
            title="Fixed x_T, varying z_sem",
        )

        hf_images = []
        hf_labels = [
            "Original",
            "H(Original)",
            "Full",
            "H(Full)",
            "H(Orig)-H(Full)",
            "z-only",
            "H(z-only)",
            "H(Orig)-H(z-only)",
            "xT-mean",
            "H(xT-mean)",
            "H(Orig)-H(xT-mean)",
        ]
        row_labels = []
        for image_id in selected_ids:
            entry = grid_cache[image_id]
            orig_hp = _highpass(entry["input"].unsqueeze(0), sigma=args.highpass_sigma)[0]
            full_hp = _highpass(entry["full"].unsqueeze(0), sigma=args.highpass_sigma)[0]
            z_hp = _highpass(entry["z_only"].unsqueeze(0), sigma=args.highpass_sigma)[0]
            xt_hp = _highpass(entry["xt_mean"].unsqueeze(0), sigma=args.highpass_sigma)[0]
            full_hp_diff = orig_hp - full_hp
            z_hp_diff = orig_hp - z_hp
            xt_hp_diff = orig_hp - xt_hp
            hf_images.extend([
                entry["input"],
                _vis_residual(orig_hp),
                entry["full"],
                _vis_residual(full_hp),
                _vis_residual(full_hp_diff),
                entry["z_only"],
                _vis_residual(z_hp),
                _vis_residual(z_hp_diff),
                entry["xt_mean"],
                _vis_residual(xt_hp),
                _vis_residual(xt_hp_diff),
            ])
            row_labels.append(_safe_id(image_id)[:18])

        _save_labeled_grid(
            hf_images,
            hf_labels,
            outputs["grids"] / "high_frequency_examples.png",
            image_size=image_size,
            row_labels=row_labels,
            title="High-Frequency Detail",
            nrow=len(hf_labels),
        )

        z_marg_avg_images = []
        z_marg_avg_labels = [
            "Full",
            "Avg z-only marginal",
            "H(Full)",
            "H(Avg)",
            "H(Full)-H(Avg)",
        ]
        row_labels = []
        for image_id in selected_ids:
            entry = grid_cache[image_id]
            full = entry["full"]
            avg = entry["z_only_marginal_avg"]
            full_hp = _highpass(full.unsqueeze(0), sigma=args.highpass_sigma)[0]
            avg_hp = _highpass(avg.unsqueeze(0), sigma=args.highpass_sigma)[0]
            hp_diff = full_hp - avg_hp

            z_marg_avg_images.extend(
                [
                    full,
                    avg,
                    _vis_residual(full_hp),
                    _vis_residual(avg_hp),
                    _vis_residual(hp_diff),
                ]
            )
            row_labels.append(_safe_id(image_id)[:18])

        _save_labeled_grid(
            z_marg_avg_images,
            z_marg_avg_labels,
            outputs["grids"] / "z_only_marginal_average_vs_full_grid.png",
            image_size=image_size,
            row_labels=row_labels,
            title="Averaged z-only marginal vs full reconstruction",
            nrow=len(z_marg_avg_labels),
        )

    LOGGER.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()

