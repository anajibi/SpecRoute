from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
import sys
for candidate in (REPO_ROOT, REPO_ROOT / "diffae_latent_probe", EXPERIMENT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.train_attribute_classifier import train_or_load_classifier  # noqa: E402
from src.classifier_guidance import classifier_guided_ddim_sample  # noqa: E402
from src.datasets import CelebAAttributeDataset, build_subset, write_dicts_csv  # noqa: E402
from src.ddim_inversion import ddim_invert, ddim_reconstruct  # noqa: E402
from src.diffusion_backbone import load_unconditional_celebahq_backbone  # noqa: E402
from src.evaluation import compute_edit_metrics, predict_paths_to_csv  # noqa: E402
from src.utils import ensure_dir, image_file_stem, load_yaml, resolve_device, save_tensor_image, set_seed  # noqa: E402
from src.visualization import save_guidance_grid  # noqa: E402


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(cfg)
    if args.num_images is not None:
        cfg["experiment"]["num_images"] = args.num_images
    if args.target_attributes:
        cfg["editing"]["target_attributes"] = args.target_attributes
    if args.guidance_scales:
        cfg["editing"]["guidance_scales"] = [float(value) for value in args.guidance_scales]
    return cfg


def _safe_attribute_name(attribute: str) -> str:
    return attribute.replace("/", "_").replace(" ", "_")


def _merge_single_attribute_predictions(
    predictions_by_attr: dict[str, list[dict[str, object]]],
    image_paths: list[str],
) -> list[dict[str, object]]:
    merged = [{"row_id": row_idx, "image_path": str(path)} for row_idx, path in enumerate(image_paths)]
    by_path = {row["image_path"]: row for row in merged}
    for attr_name, rows in predictions_by_attr.items():
        for row in rows:
            by_path[str(row["image_path"])][attr_name] = row[attr_name]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-Z classifier-guided diffusion editing POC.")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config" / "no_z_classifier_guided_poc.yaml"))
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--target-attributes", nargs="*", default=None)
    parser.add_argument("--guidance-scales", nargs="*", type=float, default=None)
    args = parser.parse_args()

    cfg = _apply_overrides(load_yaml(args.config), args)
    exp_cfg = cfg["experiment"]
    dataset_cfg = cfg["dataset"]
    diffusion_cfg = cfg["diffusion"]
    editing_cfg = cfg["editing"]
    evaluation_cfg = cfg["evaluation"]
    visualization_cfg = cfg["visualization"]

    set_seed(int(exp_cfg.get("seed", 42)))
    device = resolve_device(exp_cfg.get("device", "cuda"))
    output_root = ensure_dir(exp_cfg.get("output_root", EXPERIMENT_DIR / "outputs" / "poc"))
    image_dir = ensure_dir(output_root / "images")
    recon_dir = ensure_dir(output_root / "reconstructions")
    grid_dir = ensure_dir(output_root / "grids")

    dataset = CelebAAttributeDataset(
        image_dir=dataset_cfg["image_dir"],
        attr_path=dataset_cfg["attr_path"],
        partition_path=dataset_cfg.get("partition_path"),
        split=dataset_cfg.get("split", "test"),
        image_size=int(dataset_cfg.get("image_size", 256)),
    )
    subset = build_subset(dataset, int(exp_cfg.get("num_images", 16)))

    target_attributes = list(editing_cfg["target_attributes"])
    target_classifiers: dict[str, torch.nn.Module] = {}
    attr_names: list[str] = []
    for target_attr in target_attributes:
        classifier, classifier_attr_names, classifier_checkpoint, classifier_loaded = train_or_load_classifier(
            cfg, device, target_attribute=target_attr
        )
        if classifier_attr_names != [target_attr]:
            raise ValueError(f"Expected a single-output classifier for {target_attr}, got {classifier_attr_names}")
        target_classifiers[target_attr] = classifier
        attr_names.append(target_attr)
        print(f"Using {target_attr} classifier checkpoint: {classifier_checkpoint} (loaded={classifier_loaded})")

    backbone = load_unconditional_celebahq_backbone(
        model_id=diffusion_cfg.get("model_id", "google/ddpm-celebahq-256"),
        device=device,
        use_fp16=bool(diffusion_cfg.get("use_fp16", True)),
    )
    print(f"Loaded unconditional Diffusers model: {diffusion_cfg.get('model_id', 'google/ddpm-celebahq-256')}")

    records: list[dict[str, object]] = []
    original_paths: list[str] = []
    guidance_scales = [float(value) for value in editing_cfg["guidance_scales"]]
    target_values = editing_cfg["target_values"]
    image_size = int(dataset_cfg.get("image_size", 256))

    for local_index in range(len(subset)):
        batch = subset[local_index]
        image = batch["image"].unsqueeze(0).to(device)
        image_id = str(batch["image_id"])
        original_path = str(batch["image_path"])
        original_paths.append(original_path)
        stem = image_file_stem(original_path, local_index)

        x_t = ddim_invert(
            backbone.unet,
            copy.deepcopy(backbone.inverse_scheduler),
            image,
            num_inversion_steps=int(diffusion_cfg.get("num_inversion_steps", 50)),
            device=device,
        )
        reconstruction = ddim_reconstruct(
            backbone.unet,
            copy.deepcopy(backbone.scheduler),
            x_t,
            num_inference_steps=int(diffusion_cfg.get("num_inference_steps", 50)),
            device=device,
        )
        reconstruction_path = save_tensor_image(reconstruction[0], recon_dir / f"{stem}_ddim_reconstruction.png")

        for target_attr in target_attributes:
            classifier = target_classifiers[target_attr]
            target_value = int(target_values[target_attr])
            target_idx = 0
            target_records_start = len(records)
            for guidance_scale in guidance_scales:
                sample = classifier_guided_ddim_sample(
                    unet=backbone.unet,
                    scheduler=copy.deepcopy(backbone.scheduler),
                    classifier=classifier,
                    x_T=x_t,
                    target_attribute_index=target_idx,
                    target_value=target_value,
                    guidance_scale=guidance_scale,
                    num_guidance_steps_per_timestep=int(editing_cfg.get("num_guidance_steps_per_timestep", 1)),
                    guidance_start_step=int(editing_cfg.get("guidance_start_step", 0)),
                    guidance_end_step=int(editing_cfg.get("guidance_end_step", diffusion_cfg.get("num_inference_steps", 50))),
                    device=device,
                    clamp_x0=bool(editing_cfg.get("clamp_x0", True)),
                    guidance_on_x0_pred=bool(editing_cfg.get("guidance_on_x0_pred", True)),
                    use_amp=bool(diffusion_cfg.get("use_amp", True)),
                    num_inference_steps=int(diffusion_cfg.get("num_inference_steps", 50)),
                )
                edited_path = save_tensor_image(
                    sample[0],
                    image_dir / f"{stem}_{target_attr}_value{target_value}_guidance{guidance_scale:g}.png",
                )
                records.append(
                    {
                        "image_id": image_id,
                        "original_path": original_path,
                        "reconstruction_path": str(reconstruction_path),
                        "edited_path": str(edited_path),
                        "target_attribute": target_attr,
                        "target_value": target_value,
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": int(diffusion_cfg.get("num_inference_steps", 50)),
                        "num_inversion_steps": int(diffusion_cfg.get("num_inversion_steps", 50)),
                    }
                )
            if bool(visualization_cfg.get("save_grids", True)) and local_index < int(visualization_cfg.get("max_visualization_images", 16)):
                target_records = records[target_records_start:]
                save_guidance_grid(
                    original_path=original_path,
                    reconstruction_path=reconstruction_path,
                    edited_records=target_records,
                    guidance_scales=guidance_scales,
                    output_path=grid_dir / f"{stem}_{target_attr}_guidance_grid.png",
                    tile_size=image_size,
                )

    records_csv = output_root / "edited_image_records.csv"
    write_dicts_csv(records, records_csv)

    if bool(evaluation_cfg.get("compute_attribute_predictions", True)):
        unique_original_paths = list(dict.fromkeys(original_paths))
        edited_paths = [str(record["edited_path"]) for record in records]
        original_by_attr: dict[str, list[dict[str, object]]] = {}
        edited_by_attr: dict[str, list[dict[str, object]]] = {}
        for target_attr, classifier in target_classifiers.items():
            safe_attr = _safe_attribute_name(target_attr)
            original_by_attr[target_attr] = predict_paths_to_csv(
                classifier,
                unique_original_paths,
                [target_attr],
                output_root / f"attribute_predictions_original_{safe_attr}.csv",
                device,
                image_size=image_size,
                batch_size=int(exp_cfg.get("batch_size", 4)),
            )
            edited_by_attr[target_attr] = predict_paths_to_csv(
                classifier,
                edited_paths,
                [target_attr],
                output_root / f"attribute_predictions_edited_{safe_attr}.csv",
                device,
                image_size=image_size,
                batch_size=int(exp_cfg.get("batch_size", 4)),
            )
        original_predictions = _merge_single_attribute_predictions(original_by_attr, unique_original_paths)
        edited_predictions = _merge_single_attribute_predictions(edited_by_attr, edited_paths)
        write_dicts_csv(original_predictions, output_root / "attribute_predictions_original.csv")
        write_dicts_csv(edited_predictions, output_root / "attribute_predictions_edited.csv")
        compute_edit_metrics(
            records,
            original_predictions,
            edited_predictions,
            attr_names,
            output_root / "edit_metrics.csv",
            output_root / "preservation_summary.csv",
            image_size=image_size,
            compute_mse=bool(evaluation_cfg.get("compute_mse", True)),
        )

    print(f"Saved edited image records: {records_csv}")
    print(f"Saved outputs under: {output_root}")


if __name__ == "__main__":
    main()
