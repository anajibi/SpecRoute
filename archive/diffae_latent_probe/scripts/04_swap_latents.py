from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffae_tools.config_io import ensure_dir, load_config, resolve_path
from diffae_tools.image_io import denormalize_from_diffae, load_image_tensor, tensor_to_pil
from diffae_tools.latent_codec import load_latent_bundle
from diffae_tools.model_loader import DiffAEModelWrapper

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def _find_index(image_ids, target: str) -> int:
    image_ids = list(image_ids)
    try:
        return image_ids.index(target)
    except ValueError:
        available = "\n".join(image_ids[:20])
        raise SystemExit(
            f"[swap] image_id not found: {target}\n\n"
            f"First available image_ids:\n{available}"
        )


def _pair_indices(image_ids, image_id_a=None, image_id_b=None, num_pairs=1, seed=0):
    image_ids = list(image_ids)

    if image_id_a is not None and image_id_b is not None:
        return [(_find_index(image_ids, image_id_a), _find_index(image_ids, image_id_b))]

    rng = random.Random(seed)
    indices = list(range(len(image_ids)))

    if len(indices) < 2:
        raise SystemExit("[swap] need at least two images for swapping.")

    pairs = []
    for _ in range(num_pairs):
        a, b = rng.sample(indices, 2)
        pairs.append((a, b))

    return pairs


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/Library/Fonts/Arial.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _load_latent_image_frame(latent_dir: Path, bundle, cfg: dict, config_dir: Path) -> pd.DataFrame:
    image_ids_path = latent_dir / "image_ids.csv"

    if image_ids_path.exists():
        frame = pd.read_csv(image_ids_path)
        if "image_path" not in frame.columns:
            if "aligned_path" in frame.columns:
                frame["image_path"] = frame["aligned_path"]
            else:
                raise ValueError(f"{image_ids_path} must contain either image_path or aligned_path.")
        if "image_id" not in frame.columns:
            raise ValueError(f"{image_ids_path} must contain image_id.")
        frame = frame[["image_id", "image_path"]].copy()
    else:
        if "raw_image_dir" in cfg or "image_dir" in cfg:
            raw_image_dir = resolve_path(cfg.get("raw_image_dir", cfg.get("image_dir")), base_dir=config_dir)
            frame = pd.DataFrame({
                "image_id": bundle.image_ids,
                "image_path": [str(raw_image_dir / image_id) for image_id in bundle.image_ids],
            })
        else:
            raise ValueError("Cannot infer image paths. Rerun latent encoding.")

    order = pd.DataFrame({"image_id": list(bundle.image_ids), "_latent_index": list(range(len(bundle.image_ids)))})
    frame = order.merge(frame, on="image_id", how="left")

    bad_paths = [p for p in frame["image_path"] if not Path(p).exists()]
    if bad_paths:
        raise FileNotFoundError(f"{len(bad_paths)} image paths do not exist.")

    return frame[["image_id", "image_path", "_latent_index"]].reset_index(drop=True)


def _load_original_01(path: str, image_size: int) -> torch.Tensor:
    img_m11 = load_image_tensor(path, image_size=image_size, normalize=True)
    img_01 = denormalize_from_diffae(img_m11.unsqueeze(0))[0]
    return img_01.clamp(0.0, 1.0)


def _decode_01(wrapper: DiffAEModelWrapper, z_sem: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    """
    Decode one latent pair and dynamically check the output range
    to prevent double-denormalization (which causes pale images).
    """
    with torch.inference_mode():
        out_raw = wrapper.decode_from_latents(z_sem, x_t)

    out_detached = out_raw.detach().cpu()

    # Scrutinize the tensor's minimum value to determine its current mathematical range
    if out_detached.min() < -0.1:
        # Tensor contains negative values; it is in [-1, 1] and requires denormalization
        out_01 = denormalize_from_diffae(out_detached)[0]
    else:
        # Tensor minimum is ~0.0; it is already in [0, 1] and denormalizing it will destroy black levels
        out_01 = out_detached[0]

    return out_01.clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Generate pairwise and reference swaps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image_id_a", default=None)
    parser.add_argument("--image_id_b", default=None)
    parser.add_argument("--num_pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    # NEW ARGUMENTS FOR GRID 2
    parser.add_argument("--reference_id", default=None, help="Explicit ID for the reference image in Grid 2.")
    parser.add_argument("--num_reference_swaps", type=int, default=32,
                        help="How many 'other' images to swap with the reference.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent
    repo_root = resolve_path(cfg["repo_root"], base_dir=config_dir)
    checkpoint_path = resolve_path(cfg["checkpoint_path"], base_dir=config_dir)
    latent_dir = resolve_path(Path(cfg["output_dir"]) / "latents", base_dir=config_dir)
    output_dir = ensure_dir(resolve_path(Path(cfg["output_dir"]) / "swaps", base_dir=config_dir))
    image_size = int(cfg.get("image_size", 256))

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch_device.type == "cuda"
    LOGGER.info("Using device: %s", torch_device)

    bundle = load_latent_bundle(latent_dir)
    frame = _load_latent_image_frame(latent_dir, bundle, cfg, config_dir)

    semantic = torch.as_tensor(bundle.semantic).float()
    stochastic = torch.as_tensor(bundle.stochastic).float()

    wrapper = DiffAEModelWrapper(repo_root=repo_root, checkpoint_path=checkpoint_path, device=str(torch_device)).load()
    font = _get_font(28)
    font_bold = _get_font(24)
    padding_val = 4

    # ==========================================
    # GRID 1: PAIRWISE RANDOM SWAPS (Original)
    # ==========================================
    pairs = _pair_indices(bundle.image_ids, args.image_id_a, args.image_id_b, num_pairs=args.num_pairs, seed=args.seed)
    LOGGER.info("Generating pairwise swaps for %d pairs...", len(pairs))

    grid1_images, pair_rows = [], []
    for pair_idx, (a_idx, b_idx) in enumerate(pairs):
        row_a, row_b = frame.iloc[a_idx], frame.iloc[b_idx]
        img_a = _load_original_01(row_a["image_path"], image_size)
        img_b = _load_original_01(row_b["image_path"], image_size)
        z_a, x_a = semantic[a_idx:a_idx + 1].to(torch_device), stochastic[a_idx:a_idx + 1].to(torch_device)
        z_b, x_b = semantic[b_idx:b_idx + 1].to(torch_device), stochastic[b_idx:b_idx + 1].to(torch_device)

        recon_a, recon_b = _decode_01(wrapper, z_a, x_a), _decode_01(wrapper, z_b, x_b)
        swap_ab, swap_ba = _decode_01(wrapper, z_a, x_b), _decode_01(wrapper, z_b, x_a)

        grid1_images.extend([img_a, recon_a, img_b, recon_b, swap_ab, swap_ba])
        pair_rows.append({"image_id_a": row_a["image_id"], "image_id_b": row_b["image_id"]})

    grid1_tensor = make_grid(
        torch.stack(grid1_images),
        nrow=6,
        padding=padding_val,
        normalize=False,
        value_range=(0, 1)
    )
    grid1_pil = tensor_to_pil(grid1_tensor, denormalize=False)

    top_margin_1, left_margin_1 = 60, 180
    canvas1 = Image.new("RGB", (grid1_pil.width + left_margin_1, grid1_pil.height + top_margin_1), (255, 255, 255))
    canvas1.paste(grid1_pil, (left_margin_1, top_margin_1))
    draw1 = ImageDraw.Draw(canvas1)

    for i, label in enumerate(["A (Orig)", "A (Recon)", "B (Orig)", "B (Recon)", "Swap (zA + xB)", "Swap (zB + xA)"]):
        x_center = left_margin_1 + padding_val + (image_size + padding_val) * i + (image_size // 2)
        draw1.text((x_center, top_margin_1 // 2), label, fill=(0, 0, 0), font=font, anchor="mm")

    for i, row in enumerate(pair_rows):
        y_center = top_margin_1 + padding_val + (image_size + padding_val) * i + (image_size // 2)
        draw1.text((left_margin_1 // 2, y_center),
                   f"Pair {i + 1}\nA: {row['image_id_a'][:12]}\nB: {row['image_id_b'][:12]}", fill=(0, 0, 0), font=font,
                   anchor="mm", align="center")

    out_path_1 = output_dir / f"grid1_pairwise_{len(pairs)}pairs.png"
    canvas1.save(out_path_1)
    LOGGER.info("Saved pairwise grid to %s", out_path_1)

    # ==========================================
    # GRID 2: ONE-TO-MANY REFERENCE SWAPS (New)
    # ==========================================
    # Determine reference ID
    if args.reference_id is not None:
        ref_idx = _find_index(bundle.image_ids, args.reference_id)
    else:
        ref_idx = pairs[0][0]  # Default to the first image 'A' from the pair generation

    ref_row = frame.iloc[ref_idx]
    ref_img = _load_original_01(ref_row["image_path"], image_size)
    z_ref = semantic[ref_idx:ref_idx + 1].to(torch_device)
    x_ref = stochastic[ref_idx:ref_idx + 1].to(torch_device)

    # Gather 'B' images, ensuring we don't pick the reference itself
    rng = random.Random(args.seed + 1)
    available_indices = [i for i in range(len(bundle.image_ids)) if i != ref_idx]
    num_to_sample = min(args.num_reference_swaps, len(available_indices))
    b_indices = rng.sample(available_indices, num_to_sample)

    LOGGER.info("Generating reference swaps for Ref=%s against %d images...", ref_row["image_id"], num_to_sample)

    grid2_images = []
    ref_rows = []

    for idx in b_indices:
        b_row = frame.iloc[idx]
        img_b = _load_original_01(b_row["image_path"], image_size)
        z_b = semantic[idx:idx + 1].to(torch_device)
        x_b = stochastic[idx:idx + 1].to(torch_device)

        # Col 2: Reference Semantic + B Stochastic
        swap_zref_xb = _decode_01(wrapper, z_ref, x_b)

        # Col 3: B Semantic + Reference Stochastic
        swap_zb_xref = _decode_01(wrapper, z_b, x_ref)

        grid2_images.extend([img_b, swap_zref_xb, swap_zb_xref])
        ref_rows.append({"image_id_b": b_row["image_id"]})

    grid2_tensor = make_grid(
        torch.stack(grid2_images),
        nrow=3,
        padding=padding_val,
        normalize=False,
        value_range=(0, 1)
    )
    grid2_pil = tensor_to_pil(grid2_tensor, denormalize=False)

    # Layout dimensions for Grid 2
    top_header = image_size + 140  # Ample space for the top reference image and headers
    left_margin_2 = 180

    # Calculate exact placement for the top reference image so it centers nicely above the 3 columns
    canvas2_width = max(grid2_pil.width + left_margin_2, left_margin_2 + image_size + 40)
    canvas2_height = grid2_pil.height + top_header
    canvas2 = Image.new("RGB", (canvas2_width, canvas2_height), (255, 255, 255))

    # Paste the grid
    canvas2.paste(grid2_pil, (left_margin_2, top_header))
    draw2 = ImageDraw.Draw(canvas2)

    # Draw the Reference Image on top
    ref_pil = tensor_to_pil(ref_img, denormalize=False)
    ref_x = left_margin_2 + (grid2_pil.width // 2) - (image_size // 2)
    ref_y = 40
    canvas2.paste(ref_pil, (ref_x, ref_y))

    draw2.text((ref_x + image_size // 2, ref_y - 20), f"Reference Image: {ref_row['image_id'][:16]}", fill=(0, 0, 0),
               font=font_bold, anchor="mm")

    # Column Labels for Grid 2
    col2_labels = [
        "Other Image (B)\nOriginal",
        "Swap Z\n(z_Ref + x_B)",
        "Swap X\n(z_B + x_Ref)"
    ]
    for i, label in enumerate(col2_labels):
        x_center = left_margin_2 + padding_val + (image_size + padding_val) * i + (image_size // 2)
        draw2.text((x_center, top_header - 30), label, fill=(0, 0, 0), font=font, anchor="mm", align="center")

    # Row Labels for Grid 2
    for i, row in enumerate(ref_rows):
        y_center = top_header + padding_val + (image_size + padding_val) * i + (image_size // 2)
        draw2.text((left_margin_2 // 2, y_center), f"Target B:\n{row['image_id_b'][:16]}", fill=(0, 0, 0), font=font,
                   anchor="mm", align="center")

    out_path_2 = output_dir / f"grid2_reference_swaps.png"
    canvas2.save(out_path_2)
    LOGGER.info("Saved reference one-to-many grid to %s", out_path_2)


if __name__ == "__main__":
    main()