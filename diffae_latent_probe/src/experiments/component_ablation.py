from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import logging
import time

from torch.utils.data import DataLoader, Dataset

from src.metrics.high_frequency import highpass
from src.metrics.reconstruction import ssim
from src.visualization.grids import save_labeled_grid
from src.visualization.highpass import vis_residual
from torchvision.utils import save_image


def lpips_batch(lpips_metric, original_01: torch.Tensor, recon_01: torch.Tensor) -> torch.Tensor:
    x = original_01 * 2.0 - 1.0
    y = recon_01 * 2.0 - 1.0
    with torch.inference_mode():
        vals = lpips_metric(x, y)
    return vals.view(-1)


class ComponentAblationDataset(Dataset):
    def __init__(self, image_ids: list[str], image_dir: str | Path, image_size: int):
        self.image_ids = list(image_ids)
        self.image_dir = Path(image_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        image_path = self.image_dir / image_id
        image = _load_image_01(image_path, self.image_size)
        return {
            "idx": idx,
            "image_id": image_id,
            "image": image,
        }


def decode_in_chunks(model, z: torch.Tensor, x_t: torch.Tensor, chunk_size: int) -> torch.Tensor:
    outs = []
    for start in range(0, z.shape[0], chunk_size):
        end = min(start + chunk_size, z.shape[0])
        outs.append(model.decode(z[start:end], x_t[start:end]))
    return torch.cat(outs, dim=0)


def batch_to_01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if x.min() < 0.0 or x.max() > 1.0:
        x = (x.clamp(-1, 1) + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def mse_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).flatten(1).mean(dim=1)


def hf_metrics_batch(x: torch.Tensor, y: torch.Tensor, sigma: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    x_hp = highpass(x, sigma=sigma)
    y_hp = highpass(y, sigma=sigma)
    diff = x_hp - y_hp
    hf_mse_values = (diff ** 2).flatten(1).mean(dim=1)
    hf_l1_values = diff.abs().flatten(1).mean(dim=1)
    return hf_mse_values, hf_l1_values


def ssim_batch_cpu(original: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    vals = []
    original_np = original.permute(0, 2, 3, 1).numpy()
    recon_np = recon.permute(0, 2, 3, 1).numpy()
    for i in range(original_np.shape[0]):
        vals.append(ssim(original_np[i], recon_np[i]))
    return torch.tensor(vals)


def append_metric_rows(metrics_rows: list[dict], image_ids: list[str], setting: str, values: dict) -> None:
    for i, image_id in enumerate(image_ids):
        metrics_rows.append(
            {
                "image_id": image_id,
                "setting": setting,
                "lpips": float(values["lpips"][i]) if values["lpips"] is not None else float("nan"),
                "ssim": float(values["ssim"][i]) if values["ssim"] is not None else float("nan"),
                "mse": float(values["mse"][i]) if values["mse"] is not None else float("nan"),
                "hf_mse": float(values["hf_mse"][i]) if values["hf_mse"] is not None else float("nan"),
                "hf_l1": float(values["hf_l1"][i]) if values["hf_l1"] is not None else float("nan"),
            }
        )


def save_component_visualizations(
    image_ids: list[str],
    originals: torch.Tensor,
    recons: dict[str, torch.Tensor],
    grids_dir: Path,
    images_dir: Path,
    image_size: int,
    compute_high_frequency: bool,
    summary_grid_images: list[torch.Tensor],
    summary_grid_labels: list[str],
) -> None:
    for i, image_id in enumerate(image_ids):
        grid_images = [
            originals[i],
            recons["full"][i],
            recons["z_only"][i],
            recons["z_only_marginal_avg"][i],
            recons["xt_only_mean"][i],
            recons["xt_only_zero"][i],
            recons["z_swap"][i],
            recons["xt_swap"][i],
        ]
        save_labeled_grid(
            images=grid_images,
            col_labels=[
                "original",
                "full",
                "z-only",
                "z-only avg",
                "xT-mean",
                "xT-zero",
                "xT-mismatch",
                "xT-swap",
            ],
            out_path=grids_dir / f"component_ablation_{image_id}.png",
            image_size=image_size,
        )

        sample_dir = images_dir / image_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_image(recons["full"][i], sample_dir / "full.png")
        save_image(recons["z_only"][i], sample_dir / "z_only.png")
        save_image(recons["z_only_marginal_avg"][i], sample_dir / "z_only_avg.png")
        save_image(recons["xt_only_mean"][i], sample_dir / "xt_mean.png")
        save_image(recons["xt_only_zero"][i], sample_dir / "xt_zero.png")
        save_image(recons["z_swap"][i], sample_dir / "z_swap.png")
        save_image(recons["xt_swap"][i], sample_dir / "xt_swap.png")
        save_image(originals[i], sample_dir / "original.png")

        summary_grid_images.extend(grid_images)
        summary_grid_labels.append(image_id)

        if compute_high_frequency:
            orig_hp = highpass(originals[i].unsqueeze(0), sigma=2.0)[0]
            full_hp = highpass(recons["full"][i].unsqueeze(0), sigma=2.0)[0]
            z_hp = highpass(recons["z_only"][i].unsqueeze(0), sigma=2.0)[0]
            z_avg_hp = highpass(recons["z_only_marginal_avg"][i].unsqueeze(0), sigma=2.0)[0]
            xt_hp = highpass(recons["xt_only_mean"][i].unsqueeze(0), sigma=2.0)[0]
            save_labeled_grid(
                images=[
                    originals[i],
                    vis_residual(orig_hp),
                    recons["full"][i],
                    vis_residual(full_hp),
                    recons["z_only"][i],
                    vis_residual(z_hp),
                    recons["z_only_marginal_avg"][i],
                    vis_residual(z_avg_hp),
                    recons["xt_only_mean"][i],
                    vis_residual(xt_hp),
                ],
                col_labels=[
                    "original",
                    "H(orig)",
                    "full",
                    "H(full)",
                    "z-only",
                    "H(z-only)",
                    "z-only avg",
                    "H(z-only avg)",
                    "xT-mean",
                    "H(xT-mean)",
                ],
                out_path=grids_dir / f"high_frequency_{image_id}.png",
                image_size=image_size,
            )


def run_component_ablation(
    model,
    bundle,
    image_dir: str | Path,
    output_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_xt_samples: int,
    device: torch.device,
    compute_lpips: bool,
    compute_ssim: bool,
    compute_mse: bool,
    compute_high_frequency: bool,
    lpips_metric=None,
    max_visualization_images: int | None = None,
    save_grids: bool = True,
    save_per_image_metrics: bool = False,
    num_workers: int = 4,
    decode_chunk_size: int | None = None,
    use_amp: bool = True,
) -> None:
    output_dir = Path(output_dir)
    grids_dir = output_dir / "grids"
    images_dir = output_dir / "images"
    grids_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = logging.getLogger(__name__)

    LOGGER.info("Component ablation output: %s", output_dir)
    LOGGER.info("Total images: %d", len(bundle.image_ids))
    LOGGER.info("Batch size: %d | num_xt_samples: %d", batch_size, num_xt_samples)
    if max_visualization_images is None:
        LOGGER.info("Visualization cap: none")
    else:
        LOGGER.info("Visualization cap: %d", max_visualization_images)
    LOGGER.info("Save per-image metrics: %s", save_per_image_metrics)
    LOGGER.info("Decode chunk size: %s", decode_chunk_size or batch_size)
    LOGGER.info("Use AMP: %s", use_amp)

    dataset = ComponentAblationDataset(bundle.image_ids, image_dir, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    z_sem_all_cpu = bundle.z_sem.float().cpu()
    x_t_all_cpu = bundle.x_t.float().cpu()
    z_mean_cpu = z_sem_all_cpu.mean(dim=0, keepdim=True)

    perm = np.random.RandomState(0).permutation(len(z_sem_all_cpu))

    metrics_rows = []
    summary_grid_images = []
    summary_grid_labels = []

    if max_visualization_images is None or max_visualization_images >= len(bundle.image_ids):
        visualization_indices = set(range(len(bundle.image_ids)))
    else:
        visualization_indices = set(range(min(max_visualization_images, len(bundle.image_ids))))

    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)
    except Exception:
        ssim_metric = None

    decode_chunk_size = decode_chunk_size or batch_size
    amp_enabled = use_amp and device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        t0 = time.perf_counter()
        batch_indices = batch["idx"].long()
        image_ids = batch["image_id"]
        original_cpu = batch["image"]

        z_batch = z_sem_all_cpu[batch_indices].to(device, non_blocking=True)
        x_t_batch = x_t_all_cpu[batch_indices].to(device, non_blocking=True)

        perm_indices = torch.as_tensor([perm[int(i)] for i in batch_indices], dtype=torch.long)
        z_mismatch = z_sem_all_cpu[perm_indices].to(device, non_blocking=True)
        x_t_swap = x_t_all_cpu[perm_indices].to(device, non_blocking=True)
        z_mean_batch = z_mean_cpu.to(device, non_blocking=True).expand_as(z_batch)

        decode_start = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                recon_full = decode_in_chunks(model, z_batch, x_t_batch, decode_chunk_size)
                eps = torch.randn_like(x_t_batch)
                recon_z_only = decode_in_chunks(model, z_batch, eps, decode_chunk_size)
                recon_xt_mean = decode_in_chunks(model, z_mean_batch, x_t_batch, decode_chunk_size)
                recon_xt_zero = decode_in_chunks(model, torch.zeros_like(z_batch), x_t_batch, decode_chunk_size)
                recon_xt_mismatch = decode_in_chunks(model, z_mismatch, x_t_batch, decode_chunk_size)
                recon_xt_swap = decode_in_chunks(model, z_batch, x_t_swap, decode_chunk_size)

                eps_shape = (len(batch_indices) * num_xt_samples, *x_t_batch.shape[1:])
                eps_marginal = torch.randn(eps_shape, device=device)
                z_rep = z_batch.repeat_interleave(num_xt_samples, dim=0)
                recon_z_only_marginal = decode_in_chunks(model, z_rep, eps_marginal, decode_chunk_size)

        recon_z_only_marginal_avg = recon_z_only_marginal.view(
            len(batch_indices),
            num_xt_samples,
            *recon_full.shape[1:],
        ).mean(dim=1)
        decode_time = time.perf_counter() - decode_start

        metric_start = time.perf_counter()
        original_gpu = original_cpu.to(device, non_blocking=True)
        original_gpu = batch_to_01(original_gpu).float()

        recons_gpu = {
            "full": batch_to_01(recon_full).float(),
            "z_only": batch_to_01(recon_z_only).float(),
            "z_only_marginal_avg": batch_to_01(recon_z_only_marginal_avg).float(),
            "xt_only_mean": batch_to_01(recon_xt_mean).float(),
            "xt_only_zero": batch_to_01(recon_xt_zero).float(),
            "z_swap": batch_to_01(recon_xt_mismatch).float(),
            "xt_swap": batch_to_01(recon_xt_swap).float(),
        }

        for setting, recon_gpu in recons_gpu.items():
            values = {
                "lpips": None,
                "ssim": None,
                "mse": None,
                "hf_mse": None,
                "hf_l1": None,
            }
            if compute_mse:
                values["mse"] = mse_batch(original_gpu, recon_gpu).cpu()
            if compute_high_frequency:
                hf_mse_vals, hf_l1_vals = hf_metrics_batch(original_gpu, recon_gpu, sigma=2.0)
                values["hf_mse"] = hf_mse_vals.cpu()
                values["hf_l1"] = hf_l1_vals.cpu()
            if compute_lpips and lpips_metric is not None:
                values["lpips"] = lpips_batch(lpips_metric, original_gpu, recon_gpu).cpu()
            if compute_ssim:
                if ssim_metric is not None:
                    with torch.inference_mode():
                        ssim_vals = ssim_metric(recon_gpu, original_gpu)
                    if ssim_vals.ndim == 0:
                        ssim_vals = ssim_vals.repeat(recon_gpu.shape[0])
                    values["ssim"] = ssim_vals.detach().cpu()
                else:
                    values["ssim"] = ssim_batch_cpu(original_cpu, recon_gpu.cpu())

            append_metric_rows(metrics_rows, list(image_ids), setting, values)

        metric_time = time.perf_counter() - metric_start

        viz_start = time.perf_counter()
        if save_grids and visualization_indices:
            vis_mask = [int(idx) in visualization_indices for idx in batch_indices]
            if any(vis_mask):
                vis_ids = [image_ids[i] for i, keep in enumerate(vis_mask) if keep]
                originals_vis = original_cpu[vis_mask]
                recons_vis = {k: v[vis_mask].cpu() for k, v in recons_gpu.items()}
                save_component_visualizations(
                    vis_ids,
                    originals_vis,
                    recons_vis,
                    grids_dir,
                    images_dir,
                    image_size,
                    compute_high_frequency,
                    summary_grid_images,
                    summary_grid_labels,
                )
        viz_time = time.perf_counter() - viz_start

        total_time = time.perf_counter() - t0
        LOGGER.info(
            "batch %d / %d | decode %.2fs | metrics %.2fs | viz %.2fs | total %.2fs",
            batch_idx + 1,
            (len(bundle.image_ids) + batch_size - 1) // batch_size,
            decode_time,
            metric_time,
            viz_time,
            total_time,
        )

    if save_grids and summary_grid_images:
        save_labeled_grid(
            images=summary_grid_images,
            col_labels=[
                "original",
                "full",
                "z-only",
                "z-only avg",
                "xT-mean",
                "xT-zero",
                "xT-mismatch",
                "xT-swap",
            ],
            out_path=grids_dir / "component_ablation_summary.png",
            image_size=image_size,
            row_labels=summary_grid_labels,
        )

    if save_per_image_metrics:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        LOGGER.info("Saved metrics to %s", output_dir / "metrics.csv")
    else:
        metrics_df = pd.DataFrame(metrics_rows)

    summary_rows = []
    for setting, group in metrics_df.groupby("setting"):
        summary_rows.append(
            {
                "setting": setting,
                "lpips_mean": float(group["lpips"].mean()),
                "lpips_std": float(group["lpips"].std(ddof=0)),
                "ssim_mean": float(group["ssim"].mean()),
                "ssim_std": float(group["ssim"].std(ddof=0)),
                "mse_mean": float(group["mse"].mean()),
                "mse_std": float(group["mse"].std(ddof=0)),
                "hf_mse_mean": float(group["hf_mse"].mean()),
                "hf_mse_std": float(group["hf_mse"].std(ddof=0)),
                "hf_l1_mean": float(group["hf_l1"].mean()),
                "hf_l1_std": float(group["hf_l1"].std(ddof=0)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    LOGGER.info("Saved summary to %s", output_dir / "summary.csv")


def _load_image_01(image_path: str | Path, image_size: int) -> torch.Tensor:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    return tensor.permute(2, 0, 1).clamp(0.0, 1.0)
