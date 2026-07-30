from __future__ import annotations

from pathlib import Path

import torch
import logging

from src.visualization.edit_grids import save_attribute_edit_grid


LOGGER = logging.getLogger(__name__)


def _to_01_gpu(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float()
    if image.min() < 0.0 or image.max() > 1.0:
        image = (image.clamp(-1, 1) + 1.0) / 2.0
    return image.clamp(0.0, 1.0)


def _to_01_cpu(image: torch.Tensor) -> torch.Tensor:
    return _to_01_gpu(image).cpu()


def generate_pseudo_counterfactuals(
    model,
    bundle,
    directions: dict,
    target_attributes: list[str],
    alpha_values: list[int],
    output_dir: str | Path,
    image_size: int,
    device: torch.device,
    batch_size: int,
    max_visualization_images: int | None = None,
    save_images: bool = True,
    save_grids: bool = True,
    xt_source: str = "encoded",
    gaussian_generator: torch.Generator | None = None,
    use_amp: bool = True,
) -> dict[str, dict[str, str]]:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    grids_dir = output_dir / "grids"
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    if save_grids:
        grids_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Pseudo-counterfactual outputs: %s", output_dir)
    LOGGER.info("Attributes: %s", ", ".join(target_attributes))
    LOGGER.info("Alpha values: %s", alpha_values)
    LOGGER.info("Batch size: %d", batch_size)
    if max_visualization_images is None:
        LOGGER.info("Visualization cap: none")
    else:
        LOGGER.info("Visualization cap: %d", max_visualization_images)
    LOGGER.info("x_T source: %s", xt_source)

    z_sem_cpu = bundle.z_sem.float().cpu()
    x_t_cpu = bundle.x_t.float().cpu()

    class _IndexDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(bundle.image_ids)

        def __getitem__(self, index: int) -> int:
            return index

    loader = torch.utils.data.DataLoader(_IndexDataset(), batch_size=batch_size, shuffle=False)

    def _broadcast_dir(direction: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if direction.dim() == ref.dim():
            return direction
        if direction.dim() == 1 and ref.dim() == 2:
            return direction.view(1, -1)
        view_shape = [1] * ref.dim()
        view_shape[-1] = direction.shape[-1]
        return direction.view(*view_shape)

    results = {}
    amp_enabled = use_amp and device.type == "cuda"

    for attr_idx, attr in enumerate(target_attributes):
        LOGGER.info("Editing attribute %d / %d: %s", attr_idx + 1, len(target_attributes), attr)
        attr_dir = images_dir / attr
        if save_images:
            attr_dir.mkdir(parents=True, exist_ok=True)
        w = directions[attr]["w"].to(device)
        random_dir = torch.randn_like(w)
        random_dir = random_dir / (random_dir.norm() + 1e-8)

        records = []
        grid_rows = []
        row_labels = []
        col_labels = [f"alpha_{a}" for a in alpha_values]
        saved_rows = 0

        for batch_idx, batch_indices in enumerate(loader):
            idx = batch_indices.tolist()
            ids = [bundle.image_ids[i] for i in idx]
            z_batch = z_sem_cpu[idx].to(device, non_blocking=True)
            if xt_source == "gaussian":
                if gaussian_generator is None:
                    x_t_batch = torch.randn(x_t_cpu[idx].shape, device=device)
                else:
                    x_t_batch = torch.randn(x_t_cpu[idx].shape, device=device, generator=gaussian_generator)
            else:
                x_t_batch = x_t_cpu[idx].to(device, non_blocking=True)

            w_b = _broadcast_dir(w, z_batch)
            rand_b = _broadcast_dir(random_dir, z_batch)
            alpha_t = torch.tensor(alpha_values, device=w.device, dtype=w.dtype)
            alpha_view = alpha_t.view(len(alpha_values), *([1] * z_batch.dim()))

            z_edit = z_batch.unsqueeze(0) + alpha_view * w_b.unsqueeze(0)
            z_edit_flat = z_edit.view(-1, *z_batch.shape[1:])
            x_t_rep = x_t_batch.unsqueeze(0).expand(len(alpha_values), *x_t_batch.shape)
            x_t_rep = x_t_rep.reshape(-1, *x_t_batch.shape[1:])

            z_rand = z_batch.unsqueeze(0) + alpha_view * rand_b.unsqueeze(0)
            z_rand_flat = z_rand.view(-1, *z_batch.shape[1:])

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    recon = model.decode(z_edit_flat, x_t_rep)
                    rand_recon = model.decode(z_rand_flat, x_t_rep)

            recon_01 = _to_01_gpu(recon).view(len(alpha_values), len(idx), *recon.shape[1:])
            rand_01 = _to_01_gpu(rand_recon).view(len(alpha_values), len(idx), *rand_recon.shape[1:])

            for i, image_id in enumerate(ids):
                for alpha_idx, alpha in enumerate(alpha_values):
                    out_path = attr_dir / image_id / f"alpha_{alpha}.png"
                    rand_path = attr_dir / image_id / f"alpha_{alpha}_random.png"
                    if save_images:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        from torchvision.utils import save_image

                        save_image(recon_01[alpha_idx, i].cpu(), out_path)
                        save_image(rand_01[alpha_idx, i].cpu(), rand_path)

                    records.append(
                        {
                            "attribute": attr,
                            "image_id": image_id,
                            "alpha": int(alpha),
                            "kind": "edit",
                            "path": str(out_path) if save_images else "",
                        }
                    )
                    records.append(
                        {
                            "attribute": attr,
                            "image_id": image_id,
                            "alpha": int(alpha),
                            "kind": "random_control",
                            "path": str(rand_path) if save_images else "",
                        }
                    )

                if save_grids and (max_visualization_images is None or saved_rows < max_visualization_images):
                    grid_rows.extend([recon_01[a, i].cpu() for a in range(len(alpha_values))])
                    row_labels.append(image_id)
                    saved_rows += 1

            if (batch_idx + 1) % 10 == 0:
                if device.type == "cuda":
                    gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
                    LOGGER.info(
                        "attr=%s batch=%d/%d gpu_mem=%.2fGB",
                        attr,
                        batch_idx + 1,
                        len(loader),
                        gpu_mem,
                    )

            del z_edit, z_edit_flat, z_rand, z_rand_flat
            del x_t_rep, recon, rand_recon, recon_01, rand_01
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if save_grids and grid_rows:
            save_attribute_edit_grid(
                images=grid_rows,
                col_labels=col_labels,
                out_path=grids_dir / f"{attr}_edit_grid.png",
                image_size=image_size,
                row_labels=row_labels,
                title=f"{attr} edits",
            )
            LOGGER.info("Saved %s edit grid", attr)

        records_path = output_dir / f"{attr}_records.csv"
        import pandas as pd

        pd.DataFrame(records).to_csv(records_path, index=False)
        results[attr] = {"records_csv": str(records_path)}

    return results
