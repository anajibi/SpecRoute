from __future__ import annotations

import argparse
from pathlib import Path
import sys
import logging
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data.celeba_dataset import CelebADataset
from src.data.celeba_attributes import DEFAULT_TARGET_ATTRIBUTES, compute_attribute_prevalence, filter_attributes_by_prevalence
from src.experiments.attribute_directions import train_attribute_directions
from src.experiments.component_ablation import run_component_ablation
from src.experiments.pseudo_counterfactuals import generate_pseudo_counterfactuals
from src.experiments.preservation_analysis import run_preservation_analysis
from src.metrics.high_frequency import hf_l1, hf_mse
from src.metrics.reconstruction import lpips as lpips_metric_fn
from src.metrics.reconstruction import mse, ssim
from src.models.attribute_classifier import AttributeClassifier, AttributeClassifierConfig, predict_attribute_probabilities, train_attribute_classifier
from src.models.diffae_wrapper import DiffAEWrapper
from src.models.latent_bundle import LatentBundle
from src.utils.io import ensure_dir, read_yaml, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.visualization.grids import save_labeled_grid
from src.experiments.other_attribute_classifier import (
    AttributeSubsetDataset,
    build_other_attribute_list,
    evaluate_other_attribute_classifier,
    train_or_load_other_attribute_classifier,
)


def _make_lpips_metric(device: torch.device):
    try:
        import lpips

        metric = lpips.LPIPS(net="alex").to(device)
        metric.eval()
        return metric
    except Exception:
        return None


def _denorm_to_01(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.detach().cpu().clamp(-1, 1) + 1.0) / 2.0


def _write_experiment_summary(output_path: Path, summary: dict) -> None:
    lines = [
        "# Experiment Summary",
        f"- Timestamp: {summary.get('timestamp', 'n/a')}",
        f"- Config: {summary.get('config_path', 'n/a')}",
        f"- Output root: {summary.get('output_root', 'n/a')}",
        f"- Continue mode: {summary.get('continue_mode', False)}",
        "",
        "## Dataset",
        f"- Image dir: {summary.get('dataset', {}).get('image_dir', 'n/a')}",
        f"- Attr path: {summary.get('dataset', {}).get('attr_path', 'n/a')}",
        f"- Split: {summary.get('dataset', {}).get('split', 'n/a')}",
        f"- Partition path: {summary.get('dataset', {}).get('partition_path', 'n/a')}",
        f"- Dataset size: {summary.get('dataset', {}).get('size', 'n/a')}",
        f"- Subset size: {summary.get('dataset', {}).get('subset_size', 'n/a')}",
        "",
        "## Attributes",
        f"- Target attributes kept: {summary.get('attributes', {}).get('target_kept_count', 0)}",
        f"- Target attributes removed: {summary.get('attributes', {}).get('target_removed_count', 0)}",
        f"- Other attributes kept: {summary.get('attributes', {}).get('other_count', 0)}",
        "",
        "## Steps",
        f"- Latents: {summary.get('latents', {}).get('status', 'n/a')} ({summary.get('latents', {}).get('path', 'n/a')})",
        f"- Component ablation: {summary.get('component_ablation', {}).get('status', 'n/a')} ({summary.get('component_ablation', {}).get('summary_path', 'n/a')})",
        f"- Attribute directions: {summary.get('attribute_directions', {}).get('status', 'n/a')} ({summary.get('attribute_directions', {}).get('path', 'n/a')})",
        "",
        "## Pseudo-counterfactual runs",
    ]
    pseudo_runs = summary.get("pseudo_runs", [])
    if not pseudo_runs:
        lines.append("- None")
    else:
        for run in pseudo_runs:
            lines.append(
                f"- {run.get('name', 'run')}: {run.get('status', 'n/a')} ({run.get('output_dir', 'n/a')})"
            )
    lines.extend(
        [
            "",
            "## Classifiers",
            f"- Other attribute classifier: {summary.get('other_classifier', {}).get('status', 'n/a')}"
            f"; mean acc={summary.get('other_classifier', {}).get('val_accuracy_mean', 'n/a')}",
            f"- Attribute classifier: {summary.get('attribute_classifier', {}).get('status', 'n/a')}"
            f"; target mean acc={summary.get('attribute_classifier', {}).get('target_accuracy_mean', 'n/a')}",
            "",
            "## Metrics",
            f"- Compute attribute predictions: {summary.get('metrics', {}).get('compute_attribute_predictions', False)}",
            f"- Compute MSE: {summary.get('metrics', {}).get('compute_mse', False)}",
            f"- Compute SSIM: {summary.get('metrics', {}).get('compute_ssim', False)}",
            f"- Compute LPIPS: {summary.get('metrics', {}).get('compute_lpips', False)}",
            f"- Compute high frequency: {summary.get('metrics', {}).get('compute_high_frequency', False)}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def _build_subset(dataset: CelebADataset, num_images: int | None) -> tuple[Subset, list[int]]:
    if num_images is None or num_images >= len(dataset):
        indices = list(range(len(dataset)))
        return Subset(dataset, indices), indices
    indices = list(range(num_images))
    return Subset(dataset, indices), indices


def _compute_attribute_accuracy(
    probs_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    target_attributes: list[str],
) -> pd.DataFrame:
    rows = []
    for attr in target_attributes:
        preds = probs_df[["image_id", attr]].rename(columns={attr: "pred"})
        truth = gt_df[["image_id", attr]].rename(columns={attr: "gt"})
        merged = preds.merge(truth, on="image_id", how="inner")
        if merged.empty:
            continue
        pred_labels = (merged["pred"] >= 0.5).astype(int)
        gt_labels = merged["gt"].astype(int)
        accuracy = float((pred_labels == gt_labels).mean())
        rows.append({"attribute": attr, "accuracy": accuracy, "num_samples": int(len(merged))})
    return pd.DataFrame(rows)


def _compute_edited_accuracy(
    edited_probs_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    target_attributes: list[str],
    alpha_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cumulative_rows = []
    for alpha in alpha_values:
        total_correct = 0
        total_samples = 0
        for attr in target_attributes:
            subset = edited_probs_df[
                (edited_probs_df["attribute"] == attr)
                & (edited_probs_df["alpha"] == alpha)
                & (edited_probs_df["edit_type"] == "attribute")
            ]
            if subset.empty:
                continue
            preds = subset[["image_id", attr]].rename(columns={attr: "pred"})
            truth = gt_df[["image_id", attr]].rename(columns={attr: "gt"})
            merged = preds.merge(truth, on="image_id", how="inner")
            if merged.empty:
                continue
            pred_labels = (merged["pred"] >= 0.5).astype(int)
            gt_labels = merged["gt"].astype(int)
            correct = int((pred_labels == gt_labels).sum())
            accuracy = float(correct / len(merged))
            rows.append(
                {
                    "edited_attribute": attr,
                    "alpha": alpha,
                    "accuracy": accuracy,
                    "num_samples": int(len(merged)),
                }
            )
            total_correct += correct
            total_samples += int(len(merged))
        if total_samples:
            cumulative_rows.append(
                {
                    "alpha": alpha,
                    "accuracy": float(total_correct / total_samples),
                    "num_samples": int(total_samples),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(cumulative_rows)


def _evaluate_pseudo_results(
    pseudo_dir: Path,
    pseudo_results: dict[str, dict[str, str]],
    alpha_values: list[int],
    classifier: AttributeClassifier | None,
    device: torch.device,
    batch_size: int,
    attr_names: list[str],
    target_attributes: list[str],
    original_probs_df: pd.DataFrame | None,
    gt_df: pd.DataFrame | None,
    image_lookup_01: dict[str, torch.Tensor],
    bundle: LatentBundle,
    compute_mse: bool,
    compute_ssim: bool,
    compute_lpips: bool,
    compute_high_frequency: bool,
    lpips_metric,
    dataset_cfg: dict,
    cfg: dict,
    logger: logging.Logger,
    other_classifier: AttributeClassifier | None = None,
    other_attr_names: list[str] | None = None,
    other_original_probs_df: pd.DataFrame | None = None,
) -> None:
    import pandas as pd
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    def _predict_records_streaming(
        classifier: AttributeClassifier,
        loader: DataLoader,
        attr_names: list[str],
        device: torch.device,
        logger: logging.Logger,
        progress_label: str,
    ) -> pd.DataFrame:
        classifier.eval()
        rows: list[dict[str, object]] = []
        total_batches = len(loader)
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                images = (batch["image"].to(device, non_blocking=True) - 0.5) / 0.5
                logits = classifier(images)
                probs = torch.sigmoid(logits).detach().cpu()

                batch_size_local = probs.shape[0]
                for i in range(batch_size_local):
                    row = {
                        "image_id": batch["image_id"][i],
                        "attribute": batch["attribute"][i],
                            "alpha": batch["alpha"][i],
                        "edit_type": batch["edit_type"][i],
                    }
                    for j, attr_name in enumerate(attr_names):
                        row[attr_name] = float(probs[i, j])
                    rows.append(row)

                del images, logits, probs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                if batch_idx % max(1, total_batches // 5) == 0 and batch_idx > 0:
                    logger.info("%s: batch %d/%d", progress_label, batch_idx, total_batches)

        result = pd.DataFrame(rows)
        logger.info("%s complete: %d rows", progress_label, len(result))
        return result

    records_frames = []
    for attr, payload in pseudo_results.items():
        records_path = payload.get("records_csv") if isinstance(payload, dict) else None
        if records_path and Path(records_path).exists():
            records_frames.append(pd.read_csv(records_path))
    if not records_frames:
        logger.warning("No pseudo-counterfactual records found for evaluation.")
        return

    records_df = pd.concat(records_frames, ignore_index=True)
    records_df["edit_type"] = records_df["kind"].map({"edit": "attribute", "random_control": "random"})
    logger.info(
        "Loaded pseudo-counterfactual records: %d rows, %d attributes, output=%s",
        len(records_df),
        records_df["attribute"].nunique(),
        pseudo_dir,
    )

    class _RecordsDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.df = df.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> dict:
            row = self.df.iloc[idx]
            if not row["path"]:
                raise ValueError("Missing image path for pseudo-counterfactual record.")
            with Image.open(row["path"]) as img:
                img = img.convert("RGB")
                image = torch.from_numpy(np.array(img)).float() / 255.0
                image = image.permute(2, 0, 1).clamp(0.0, 1.0)
            return {
                "image_id": row["image_id"],
                "attribute": row["attribute"],
                "alpha": row["alpha"],
                "edit_type": row["edit_type"],
                "path": row["path"],
                "image": image,
            }

    dataset = _RecordsDataset(records_df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logger.info("Prepared edited records for evaluation: %d", len(dataset))

    if classifier is not None and original_probs_df is not None and gt_df is not None:
        if len(dataset):
            logger.info("Running attribute predictions for edited images: %d", len(dataset))
            edited_probs_df = _predict_records_streaming(
                classifier=classifier,
                loader=loader,
                attr_names=attr_names,
                device=device,
                logger=logger,
                progress_label="Edited attribute prediction",
            )
            edited_probs_df.to_csv(pseudo_dir / "attribute_predictions.csv", index=False)
            logger.info("Saved edited attribute predictions.")

            edited_only = edited_probs_df[edited_probs_df["edit_type"] == "attribute"].copy()
            logger.info("Edited-only records for evaluation: %d", len(edited_only))
            edited_accuracy_df, cumulative_accuracy_df = _compute_edited_accuracy(
                edited_only,
                gt_df,
                target_attributes,
                alpha_values,
            )
            edited_accuracy_df.to_csv(pseudo_dir / "attribute_accuracy_edited.csv", index=False)
            cumulative_accuracy_df.to_csv(pseudo_dir / "attribute_accuracy_edited_cumulative.csv", index=False)
            logger.info("Saved edited attribute accuracies.")

            logger.info("Running preservation analysis (filtered).")
            run_preservation_analysis(
                original_probs=original_probs_df,
                edited_probs=edited_only,
                attributes_df=bundle.attributes,
                target_attributes=target_attributes,
                attr_names=attr_names,
                alpha_values=alpha_values,
                output_dir=pseudo_dir,
                prevalence_min=cfg["attribute_filtering"]["prevalence_min"],
                prevalence_max=cfg["attribute_filtering"]["prevalence_max"],
                use_semantic_exclusion_groups=False,
                stats_output_path=Path(cfg["outputs"]["root"]) / "attribute_stats.csv",
                summary_output_path=pseudo_dir / "preservation_summary_all_filtered.csv",
            )
            logger.info("Running preservation analysis (semantic exclusions).")
            run_preservation_analysis(
                original_probs=original_probs_df,
                edited_probs=edited_only,
                attributes_df=bundle.attributes,
                target_attributes=target_attributes,
                attr_names=attr_names,
                alpha_values=alpha_values,
                output_dir=pseudo_dir,
                prevalence_min=cfg["attribute_filtering"]["prevalence_min"],
                prevalence_max=cfg["attribute_filtering"]["prevalence_max"],
                use_semantic_exclusion_groups=True,
                stats_output_path=Path(cfg["outputs"]["root"]) / "attribute_stats.csv",
                summary_output_path=pseudo_dir / "preservation_summary_semantic_excluded.csv",
            )

            metrics_path = pseudo_dir / "preservation_metrics.csv"
            if metrics_path.exists() and cfg["visualization"]["save_failure_grids"]:
                logger.info("Building failure/good case grids.")
                metrics_df = pd.read_csv(metrics_path)
                success_mask = ((metrics_df["alpha"] >= 0) & (metrics_df["success_score"] > 0)) | (
                    (metrics_df["alpha"] < 0) & (metrics_df["success_score"] > 0)
                )
                failure_mask = success_mask & (metrics_df["non_target_mean_abs_change"] > 0.2)
                good_mask = success_mask & (metrics_df["non_target_mean_abs_change"] < 0.05)

                failure_cases = metrics_df[failure_mask].head(cfg["visualization"]["num_visualization_images"])
                good_cases = metrics_df[good_mask].head(cfg["visualization"]["num_visualization_images"])
                logger.info(
                    "Failure/good case selection: failure=%d good=%d",
                    len(failure_cases),
                    len(good_cases),
                )

                def _load_image(path: str):
                    with Image.open(path) as img:
                        img = img.convert("RGB")
                        tensor = torch.from_numpy(np.array(img)).float() / 255.0
                        return tensor.permute(2, 0, 1).clamp(0.0, 1.0)

                def _save_cases(cases, out_path, title):
                    images = []
                    labels = []
                    for _, row in cases.iterrows():
                        key = (row["image_id"], row["edited_attribute"], row["alpha"], "attribute")
                        path = None
                        for rec in records_df.itertuples():
                            if rec.image_id == row["image_id"] and rec.attribute == row["edited_attribute"] and rec.alpha == row["alpha"] and rec.kind == "edit":
                                path = rec.path
                                break
                        if not path:
                            continue
                        images.append(_load_image(path))
                        labels.append(f"{row['image_id']}|{row['edited_attribute']}|{row['alpha']}")
                    if images:
                        save_labeled_grid(
                            images=images,
                            col_labels=["edit"],
                            out_path=out_path,
                            image_size=dataset_cfg["image_size"],
                            row_labels=labels,
                            title=title,
                            nrow=1,
                        )

                _save_cases(failure_cases, pseudo_dir / "failure_cases.png", "Failure cases")
                _save_cases(good_cases, pseudo_dir / "good_cases.png", "Good cases")
                logger.info("Saved failure/good case grids.")

    if other_classifier is not None and other_attr_names and other_original_probs_df is not None and len(dataset):
        other_probs_df = _predict_records_streaming(
            classifier=other_classifier,
            loader=loader,
            attr_names=other_attr_names,
            device=device,
            logger=logger,
            progress_label="Other-attribute prediction",
        )
        other_probs_df.to_csv(pseudo_dir / "other_attribute_predictions.csv", index=False)
        logger.info("Saved other-attribute predictions: %s", pseudo_dir / "other_attribute_predictions.csv")

        edited_other = other_probs_df[other_probs_df["edit_type"] == "attribute"].copy().reset_index(drop=True)
        original_other = other_original_probs_df.set_index("image_id")

        edited_vals = edited_other[other_attr_names].to_numpy()
        original_vals = original_other.loc[edited_other["image_id"]][other_attr_names].to_numpy()
        delta = edited_vals - original_vals
        abs_delta = abs(delta)
        flip = (edited_vals >= 0.5) != (original_vals >= 0.5)

        edited_other["other_mean_abs_delta"] = abs_delta.mean(axis=1)
        edited_other["other_mean_signed_delta"] = delta.mean(axis=1)
        edited_other["other_flip_rate"] = flip.mean(axis=1)

        summary_rows = []
        grouped = edited_other.groupby(["attribute", "alpha"])
        for (attr, alpha), group in grouped:
            summary_rows.append(
                {
                    "edited_attribute": attr,
                    "alpha": alpha,
                    "mean_abs_delta": float(group["other_mean_abs_delta"].mean()),
                    "mean_signed_delta": float(group["other_mean_signed_delta"].mean()),
                    "mean_flip_rate": float(group["other_mean_flip_rate"].mean()),
                }
            )
        pd.DataFrame(summary_rows).to_csv(pseudo_dir / "other_attribute_skew_summary.csv", index=False)
        logger.info("Saved other-attribute skew summary: %s", pseudo_dir / "other_attribute_skew_summary.csv")

        per_attr_rows = []
        for j, other_attr in enumerate(other_attr_names):
            vals_abs = abs_delta[:, j]
            vals_signed = delta[:, j]
            vals_flip = flip[:, j].astype(float)
            for (attr, alpha), group in grouped:
                idx = group.index.to_numpy()
                per_attr_rows.append(
                    {
                        "edited_attribute": attr,
                        "alpha": alpha,
                        "other_attribute": other_attr,
                        "mean_abs_delta": float(vals_abs[idx].mean()),
                        "mean_signed_delta": float(vals_signed[idx].mean()),
                        "mean_flip_rate": float(vals_flip[idx].mean()),
                    }
                )
        pd.DataFrame(per_attr_rows).to_csv(pseudo_dir / "other_attribute_skew_by_attr.csv", index=False)
        logger.info("Saved other-attribute skew per-attribute: %s", pseudo_dir / "other_attribute_skew_by_attr.csv")

    if compute_mse or compute_ssim:
        logger.info("Computing image similarity metrics for edited images.")
        similarity_rows = []
        for record in records_df.itertuples():
            if not record.path:
                continue
            image_id = record.image_id
            original_01 = image_lookup_01.get(image_id)
            if original_01 is None:
                continue
            with Image.open(record.path) as img:
                img = img.convert("RGB")
                recon_01 = torch.from_numpy(np.array(img)).float() / 255.0
                recon_01 = recon_01.permute(2, 0, 1).clamp(0.0, 1.0)
            row = {
                "image_id": image_id,
                "attribute": record.attribute,
                "alpha": record.alpha,
                "edit_type": "random" if record.kind == "random_control" else "attribute",
                "lpips": float("nan"),
                "ssim": float("nan"),
                "mse": float("nan"),
                "hf_mse": float("nan"),
                "hf_l1": float("nan"),
            }
            if compute_mse:
                row["mse"] = mse(original_01, recon_01)
            if compute_ssim:
                row["ssim"] = ssim(original_01, recon_01)
            if compute_lpips:
                row["lpips"] = lpips_metric_fn(lpips_metric, device, original_01, recon_01)
            if compute_high_frequency:
                row["hf_mse"] = hf_mse(original_01, recon_01, sigma=2.0)
                row["hf_l1"] = hf_l1(original_01, recon_01, sigma=2.0)
            similarity_rows.append(row)
        pd.DataFrame(similarity_rows).to_csv(pseudo_dir / "image_similarity_metrics.csv", index=False)
        logger.info("Saved image similarity metrics.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CelebA pseudo-counterfactual experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--continue", dest="continue_experiment", action="store_true")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    logger = setup_logging()
    logging.getLogger("PIL").setLevel(logging.WARNING)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": args.config,
        "continue_mode": args.continue_experiment,
        "metrics": cfg.get("metrics", {}),
    }

    logger.info("Config path: %s", args.config)
    logger.info("Output root: %s", cfg["outputs"]["root"])
    logger.info("Continue mode: %s", args.continue_experiment)

    seed = cfg["experiment"]["seed"]
    set_seed(seed)
    logger.info("Seed: %d", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    output_root = ensure_dir(cfg["outputs"]["root"])
    write_json(Path(output_root) / "config.json", cfg)
    summary["output_root"] = str(output_root)

    dataset_cfg = cfg["dataset"]
    logger.info("Dataset image_dir: %s", dataset_cfg["image_dir"])
    logger.info("Dataset attr_path: %s", dataset_cfg["attr_path"])
    logger.info("Dataset partition_path: %s", dataset_cfg.get("partition_path"))
    logger.info("Dataset split: %s", dataset_cfg.get("split"))

    dataset = CelebADataset(
        image_dir=dataset_cfg["image_dir"],
        attr_path=dataset_cfg["attr_path"],
        image_size=dataset_cfg["image_size"],
        split=dataset_cfg.get("split"),
        partition_path=dataset_cfg.get("partition_path"),
    )
    logger.info("Dataset size: %d", len(dataset))

    subset, subset_indices = _build_subset(dataset, cfg["experiment"]["num_images"])
    logger.info("Subset size: %d", len(subset))
    dataloader = DataLoader(subset, batch_size=cfg["experiment"]["batch_size"], shuffle=False)
    logger.info("Batch size: %d", cfg["experiment"]["batch_size"])

    image_lookup_01 = {}
    for batch_idx, batch in enumerate(dataloader):
        for image_id, image in zip(batch["image_id"], batch["image"]):
            image_lookup_01[image_id] = _denorm_to_01(image)
        if batch_idx % 10 == 0:
            logger.info("Cached originals batch %d / %d", batch_idx + 1, len(dataloader))

    latents_dir = ensure_dir(output_root / "latents")
    z_sem_path = latents_dir / "z_sem.pt"
    x_t_path = latents_dir / "x_T.pt"

    latents_recomputed = True
    if cfg["latents"]["recompute"] or not z_sem_path.exists() or not x_t_path.exists():
        logger.info("Computing latents for CelebA subset.")
        model = DiffAEWrapper(
            repo_root=cfg["model"]["repo_root"],
            checkpoint_path=cfg["model"]["checkpoint_path"],
            device=str(device),
        )
        z_sem_list = []
        x_t_list = []
        image_ids = []
        attributes_rows = []
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device, non_blocking=True)
            z_sem = model.encode_semantic(images)
            x_t = model.encode_stochastic(images, z_sem)
            z_sem_list.append(z_sem.detach().cpu())
            x_t_list.append(x_t.detach().cpu())
            image_ids.extend(batch["image_id"])
            attributes_rows.append(batch["attributes"].detach().cpu())
            if batch_idx % 10 == 0:
                logger.info("Encoded latents batch %d / %d", batch_idx + 1, len(dataloader))
        z_sem_tensor = torch.cat(z_sem_list, dim=0)
        x_t_tensor = torch.cat(x_t_list, dim=0)
        attributes_tensor = torch.cat(attributes_rows, dim=0)
        attr_names = dataset.get_attr_names()
        attributes_df = pd.DataFrame(attributes_tensor.numpy(), columns=attr_names)
        bundle = LatentBundle(
            z_sem=z_sem_tensor,
            x_t=x_t_tensor,
            image_ids=image_ids,
            attributes=attributes_df,
            metadata={"image_size": dataset_cfg["image_size"]},
        )
        bundle.save(latents_dir)
        logger.info("Saved latents to %s", latents_dir)
    else:
        latents_recomputed = False
        logger.info("Loading cached latents from %s", latents_dir)
        bundle = LatentBundle.load(latents_dir)

    summary["latents"] = {
        "status": "recomputed" if latents_recomputed else "loaded",
        "path": str(latents_dir),
    }

    target_attributes = cfg.get("target_attributes", DEFAULT_TARGET_ATTRIBUTES)
    logger.info("Target attributes (raw): %s", ", ".join(target_attributes))

    prevalence_min = cfg["attribute_filtering"]["prevalence_min"]
    prevalence_max = cfg["attribute_filtering"]["prevalence_max"]

    all_attr_names = dataset.get_attr_names()
    other_attr_names, other_prevalence_df = build_other_attribute_list(
        dataset.get_attribute_frame(),
        all_attr_names,
        target_attributes,
        prevalence_min,
        prevalence_max,
    )
    other_prevalence_df.to_csv(Path(output_root) / "other_attribute_prevalence.csv", index=False)
    pd.DataFrame({"attribute": other_attr_names}).to_csv(Path(output_root) / "other_attribute_list.csv", index=False)
    logger.info("Other attributes kept after filtering: %d", len(other_attr_names))

    prevalence_df = compute_attribute_prevalence(bundle.attributes, target_attributes)
    prevalence_df["kept"] = prevalence_df["prevalence"].between(prevalence_min, prevalence_max)
    prevalence_df.to_csv(Path(output_root) / "target_attribute_prevalence.csv", index=False)
    target_attributes = filter_attributes_by_prevalence(bundle.attributes, target_attributes, prevalence_min, prevalence_max)
    removed = [attr for attr in cfg.get("target_attributes", DEFAULT_TARGET_ATTRIBUTES) if attr not in target_attributes]
    logger.info(
        "Target attributes filtered by prevalence %.3f-%.3f: kept %d / %d",
        prevalence_min,
        prevalence_max,
        len(target_attributes),
        len(cfg.get("target_attributes", DEFAULT_TARGET_ATTRIBUTES)),
    )
    if removed:
        logger.info("Removed/ignored attributes: %s", ", ".join(removed))

    summary["attributes"] = {
        "target_kept_count": len(target_attributes),
        "target_removed_count": len(removed),
        "other_count": len(other_attr_names),
    }

    lpips_metric = _make_lpips_metric(device) if cfg["metrics"]["compute_lpips"] else None
    logger.info("LPIPS enabled: %s", cfg["metrics"]["compute_lpips"])

    component_dir = ensure_dir(output_root / "component_ablation")
    model = DiffAEWrapper(
        repo_root=cfg["model"]["repo_root"],
        checkpoint_path=cfg["model"]["checkpoint_path"],
        device=str(device),
    )
    logger.info("Running component ablation.")
    ablation_cfg = cfg.get("component_ablation", {})
    summary_path = Path(component_dir) / "summary.csv"
    component_ablation_ran = not summary_path.exists()
    if summary_path.exists():
        logger.info("Skipping component ablation; found existing summary at %s", summary_path)
    else:
        run_component_ablation(
            model=model,
            bundle=bundle,
            image_dir=dataset_cfg["image_dir"],
            output_dir=component_dir,
            image_size=dataset_cfg["image_size"],
            batch_size=cfg["experiment"]["batch_size"],
            num_xt_samples=cfg["experiment"]["num_xt_samples"],
            device=device,
            compute_lpips=cfg["metrics"]["compute_lpips"],
            compute_ssim=cfg["metrics"]["compute_ssim"],
            compute_mse=cfg["metrics"]["compute_mse"],
            compute_high_frequency=cfg["metrics"]["compute_high_frequency"],
            lpips_metric=lpips_metric,
            max_visualization_images=cfg["visualization"]["max_visualization_images"],
            save_grids=cfg["visualization"]["save_component_ablation_grids"],
            save_per_image_metrics=cfg["visualization"]["save_ablation_per_image_metrics"],
            num_workers=ablation_cfg.get("num_workers", 4),
            decode_chunk_size=ablation_cfg.get("decode_chunk_size", cfg["experiment"]["batch_size"]),
            use_amp=ablation_cfg.get("use_amp", True),
        )
        logger.info("Component ablation complete: %s", component_dir)

    summary["component_ablation"] = {
        "status": "ran" if component_ablation_ran else "skipped",
        "summary_path": str(summary_path),
    }

    other_classifier = None
    other_original_probs_df = None
    other_classifier_loaded = False
    other_classifier_val_mean = None
    if other_attr_names:
        logger.info("Preparing other-attribute classifier (%d attributes).", len(other_attr_names))
        other_cfg_dict = cfg.get("other_attribute_classifier", cfg.get("attribute_classifier", {}))
        other_cfg = AttributeClassifierConfig(
            num_attributes=len(other_attr_names),
            lr=other_cfg_dict.get("lr", 1e-3),
            epochs=other_cfg_dict.get("epochs", 3),
            backbone=other_cfg_dict.get("backbone", "resnet18"),
            pretrained=other_cfg_dict.get("pretrained", False),
        )

        train_dataset = CelebADataset(
            image_dir=dataset_cfg["image_dir"],
            attr_path=dataset_cfg["attr_path"],
            image_size=dataset_cfg["image_size"],
            split="train" if dataset_cfg.get("partition_path") else dataset_cfg.get("split"),
            partition_path=dataset_cfg.get("partition_path"),
        )
        val_split = "val" if dataset_cfg.get("partition_path") else dataset_cfg.get("split")
        val_dataset = CelebADataset(
            image_dir=dataset_cfg["image_dir"],
            attr_path=dataset_cfg["attr_path"],
            image_size=dataset_cfg["image_size"],
            split=val_split,
            partition_path=dataset_cfg.get("partition_path"),
        )

        train_attr_names = train_dataset.get_attr_names()
        other_attr_indices = [train_attr_names.index(name) for name in other_attr_names]
        train_subset = AttributeSubsetDataset(train_dataset, other_attr_indices, other_attr_names)
        val_subset = AttributeSubsetDataset(val_dataset, other_attr_indices, other_attr_names)
        train_loader = DataLoader(train_subset, batch_size=cfg["experiment"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=cfg["experiment"]["batch_size"], shuffle=False)

        other_classifier, other_attr_names, other_classifier_loaded = train_or_load_other_attribute_classifier(
            output_dir=Path(output_root) / "other_attribute_classifier",
            train_loader=train_loader,
            val_loader=val_loader,
            attr_names=other_attr_names,
            device=device,
            cfg=other_cfg,
            force_retrain=other_cfg_dict.get("retrain", False),
        )

        if other_classifier_loaded:
            logger.info("Loaded existing other-attribute classifier from %s", Path(output_root) / "other_attribute_classifier")
        else:
            logger.info("Trained new other-attribute classifier.")

        val_frame = val_dataset.get_attribute_frame()[["image_id", *other_attr_names]].reset_index(drop=True)
        val_accuracy = evaluate_other_attribute_classifier(
            other_classifier,
            val_loader,
            val_frame,
            other_attr_names,
            device,
        )
        val_accuracy.to_csv(Path(output_root) / "other_attribute_classifier_val_accuracy.csv", index=False)
        if val_accuracy.empty:
            logger.warning("Other-attribute classifier validation accuracy is empty.")
        else:
            other_classifier_val_mean = float(val_accuracy["accuracy"].mean())
            logger.info(
                "Other-attribute classifier validation accuracy: mean=%.4f min=%.4f max=%.4f",
                other_classifier_val_mean,
                float(val_accuracy["accuracy"].min()),
                float(val_accuracy["accuracy"].max()),
            )
        logger.info("Saved other-attribute classifier validation accuracy.")

        subset_other = AttributeSubsetDataset(subset, other_attr_indices, other_attr_names)
        subset_other_loader = DataLoader(subset_other, batch_size=cfg["experiment"]["batch_size"], shuffle=False)
        other_probs = predict_attribute_probabilities(other_classifier, subset_other_loader, device)
        other_original_probs_df = pd.DataFrame(other_probs.numpy(), columns=other_attr_names)
        other_original_probs_df.insert(0, "image_id", bundle.image_ids)
        other_original_probs_df.to_csv(Path(output_root) / "other_attribute_predictions_original.csv", index=False)

    summary["other_classifier"] = {
        "status": "loaded" if other_classifier_loaded else ("trained" if other_classifier is not None else "skipped"),
        "val_accuracy_mean": other_classifier_val_mean,
        "val_accuracy_path": str(Path(output_root) / "other_attribute_classifier_val_accuracy.csv"),
    }

    directions_dir = Path(output_root) / "attribute_directions"
    if not target_attributes:
        logger.warning("No target attributes remain after filtering; skipping attribute directions and pseudo-counterfactuals.")
        summary["attribute_directions"] = {"status": "skipped", "path": str(directions_dir)}
        summary["pseudo_runs"] = []
        summary_path = Path(output_root) / "experiment_summary.md"
        _write_experiment_summary(summary_path, summary)
        return

    directions_dir = ensure_dir(directions_dir)
    directions_path = Path(directions_dir) / "directions.pt"
    directions_loaded = bool(args.continue_experiment and directions_path.exists())
    if directions_loaded:
        logger.info("Loading cached attribute directions from %s", directions_path)
    else:
        logger.info("Training attribute directions.")
        train_attribute_directions(
            z_sem=bundle.z_sem,
            attributes=bundle.attributes,
            target_attributes=target_attributes,
            output_dir=directions_dir,
            class_weight=cfg["attribute_editing"]["class_weight"],
            normalize_direction=cfg["attribute_editing"]["normalize_direction"],
        )
        logger.info("Saved directions to %s", directions_dir)

    directions = torch.load(directions_path, map_location="cpu")

    summary["attribute_directions"] = {
        "status": "loaded" if directions_loaded else "trained",
        "path": str(directions_path),
    }

    alpha_values = cfg["attribute_editing"]["alpha_values"]
    logger.info("Alpha values: %s", alpha_values)

    def _load_pseudo_records(root_dir: Path, attributes: list[str]) -> dict[str, dict[str, str]]:
        results = {}
        for attr in attributes:
            records_path = root_dir / f"{attr}_records.csv"
            if not records_path.exists():
                return {}
            results[attr] = {"records_csv": str(records_path)}
        return results

    pseudo_runs = []
    pseudo_dir = ensure_dir(output_root / "pseudo_counterfactuals")
    logger.info("Generating pseudo-counterfactuals (encoded x_T).")
    pseudo_results = {}
    pseudo_reused = False
    if args.continue_experiment:
        pseudo_results = _load_pseudo_records(pseudo_dir, target_attributes)
        if pseudo_results:
            pseudo_reused = True
            logger.info("Found existing pseudo-counterfactual records under %s", pseudo_dir)
    if not pseudo_results:
        pseudo_results = generate_pseudo_counterfactuals(
            model=model,
            bundle=bundle,
            directions=directions,
            target_attributes=target_attributes,
            alpha_values=alpha_values,
            output_dir=pseudo_dir,
            image_size=dataset_cfg["image_size"],
            device=device,
            batch_size=cfg["pseudo_counterfactuals"]["batch_size"],
            max_visualization_images=cfg["visualization"]["max_visualization_images"],
            save_images=cfg["visualization"]["save_attribute_images"],
            save_grids=cfg["visualization"]["save_attribute_edit_grids"],
            xt_source="encoded",
            use_amp=cfg["pseudo_counterfactuals"].get("use_amp", True),
        )
    pseudo_runs.append((pseudo_dir, pseudo_results))
    summary["pseudo_runs"] = [
        {"name": "encoded_xt", "status": "loaded" if pseudo_reused else "generated", "output_dir": str(pseudo_dir)}
    ]
    logger.info("Pseudo-counterfactuals ready under %s", pseudo_dir)

    if cfg["pseudo_counterfactuals"]["run_gaussian_xt_ablation"]:
        gaussian_dir = ensure_dir(output_root / "pseudo_counterfactuals_gaussian_xt")
        logger.info("Generating pseudo-counterfactuals (gaussian x_T ablation).")
        gaussian_results = {}
        gaussian_reused = False
        if args.continue_experiment:
            gaussian_results = _load_pseudo_records(gaussian_dir, target_attributes)
            if gaussian_results:
                gaussian_reused = True
                logger.info("Found existing gaussian pseudo-counterfactual records under %s", gaussian_dir)
        if not gaussian_results:
            generator = torch.Generator(device=device).manual_seed(cfg["experiment"]["seed"])
            gaussian_results = generate_pseudo_counterfactuals(
                model=model,
                bundle=bundle,
                directions=directions,
                target_attributes=target_attributes,
                alpha_values=alpha_values,
                output_dir=gaussian_dir,
                image_size=dataset_cfg["image_size"],
                device=device,
                batch_size=cfg["pseudo_counterfactuals"]["batch_size"],
                max_visualization_images=cfg["visualization"]["max_visualization_images"],
                save_images=cfg["visualization"]["save_attribute_images"],
                save_grids=cfg["visualization"]["save_attribute_edit_grids"],
                xt_source="gaussian",
                gaussian_generator=generator,
                use_amp=cfg["pseudo_counterfactuals"].get("use_amp", True),
            )
        pseudo_runs.append((gaussian_dir, gaussian_results))
        summary["pseudo_runs"].append(
            {
                "name": "gaussian_xt",
                "status": "loaded" if gaussian_reused else "generated",
                "output_dir": str(gaussian_dir),
            }
        )
        logger.info("Pseudo-counterfactuals (gaussian x_T) ready under %s", gaussian_dir)

    attribute_classifier_summary = {"status": "skipped", "target_accuracy_mean": None}
    if cfg["metrics"]["compute_attribute_predictions"]:
        logger.info("Preparing attribute classifier.")
        train_dataset = CelebADataset(
            image_dir=dataset_cfg["image_dir"],
            attr_path=dataset_cfg["attr_path"],
            image_size=dataset_cfg["image_size"],
            split="train" if dataset_cfg.get("partition_path") else dataset_cfg.get("split"),
            partition_path=dataset_cfg.get("partition_path"),
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg["experiment"]["batch_size"], shuffle=True)
        classifier_cfg_dict = cfg.get("attribute_classifier", {})
        classifier_dir = ensure_dir(output_root / "attribute_classifier")
        model_path = Path(classifier_dir) / "model.pt"
        meta_path = Path(classifier_dir) / "metadata.json"
        force_retrain = classifier_cfg_dict.get("retrain", False)
        if force_retrain:
            logger.info("Attribute classifier retrain requested; skipping cached model load.")
        classifier_cfg = AttributeClassifierConfig(
            num_attributes=len(train_dataset.get_attr_names()),
            lr=classifier_cfg_dict.get("lr", 1e-3),
            epochs=classifier_cfg_dict.get("epochs", 3),
            backbone=classifier_cfg_dict.get("backbone", "resnet18"),
            pretrained=classifier_cfg_dict.get("pretrained", False),
        )
        logger.info(
            "Attribute classifier: backbone=%s pretrained=%s attributes=%d epochs=%d",
            classifier_cfg.backbone,
            classifier_cfg.pretrained,
            classifier_cfg.num_attributes,
            classifier_cfg.epochs,
        )
        attr_names = train_dataset.get_attr_names()
        classifier_loaded = False
        classifier = None
        if model_path.exists() and meta_path.exists() and not force_retrain:
            try:
                metadata = json.loads(meta_path.read_text())
                meta_num_attrs = int(metadata.get("num_attributes", -1))
                meta_backbone = metadata.get("backbone", classifier_cfg.backbone)
                meta_pretrained = bool(metadata.get("pretrained", classifier_cfg.pretrained))
                meta_attr_names = metadata.get("attr_names", attr_names)
                if meta_num_attrs != len(attr_names) or meta_attr_names != attr_names:
                    logger.warning(
                        "Attribute classifier metadata mismatch (num_attrs=%s expected=%s). Retraining.",
                        meta_num_attrs,
                        len(attr_names),
                    )
                else:
                    classifier = AttributeClassifier(
                        num_attributes=meta_num_attrs,
                        backbone=meta_backbone,
                        pretrained=meta_pretrained,
                    )
                    classifier.load_state_dict(torch.load(model_path, map_location="cpu"))
                    classifier.to(device)
                    classifier_loaded = True
                    attr_names = meta_attr_names
                    logger.info("Loaded attribute classifier from %s", classifier_dir)
            except Exception as exc:
                logger.warning("Failed to load attribute classifier (%s). Retraining.", exc)
        elif not force_retrain:
            logger.info("No cached attribute classifier found at %s", classifier_dir)

        if not classifier_loaded:
            logger.info("Training attribute classifier.")
            classifier = AttributeClassifier(
                num_attributes=classifier_cfg.num_attributes,
                backbone=classifier_cfg.backbone,
                pretrained=classifier_cfg.pretrained,
            )
            classifier = train_attribute_classifier(classifier, train_loader, None, device, classifier_cfg)
            torch.save(classifier.state_dict(), model_path)
            meta_payload = {
                "num_attributes": classifier_cfg.num_attributes,
                "backbone": classifier_cfg.backbone,
                "pretrained": classifier_cfg.pretrained,
                "attr_names": attr_names,
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            meta_path.write_text(json.dumps(meta_payload, indent=2))
            logger.info("Saved attribute classifier to %s", classifier_dir)

        logger.info("Predicting attributes for originals.")
        original_loader = DataLoader(subset, batch_size=cfg["experiment"]["batch_size"], shuffle=False)
        original_probs = predict_attribute_probabilities(classifier, original_loader, device)
        original_probs_df = pd.DataFrame(original_probs.numpy(), columns=attr_names)
        original_probs_df.insert(0, "image_id", bundle.image_ids)
        original_probs_df.to_csv(Path(output_root) / "attribute_predictions_original.csv", index=False)
        logger.info("Saved original attribute predictions.")

        gt_df = bundle.attributes.copy()
        gt_df.insert(0, "image_id", bundle.image_ids)
        original_accuracy_all_df = _compute_attribute_accuracy(
            original_probs_df,
            gt_df,
            attr_names,
        )
        original_accuracy_all_df.to_csv(Path(output_root) / "attribute_accuracy_original_all.csv", index=False)
        original_accuracy_df = _compute_attribute_accuracy(original_probs_df, gt_df, target_attributes)
        original_accuracy_df.to_csv(Path(output_root) / "attribute_accuracy_original.csv", index=False)
        logger.info("Saved original attribute accuracies.")

        if not original_accuracy_df.empty:
            attribute_classifier_summary = {
                "status": "loaded" if classifier_loaded else "trained",
                "target_accuracy_mean": float(original_accuracy_df["accuracy"].mean()),
                "target_accuracy_path": str(Path(output_root) / "attribute_accuracy_original.csv"),
            }
        else:
            attribute_classifier_summary = {
                "status": "loaded" if classifier_loaded else "trained",
                "target_accuracy_mean": None,
                "target_accuracy_path": str(Path(output_root) / "attribute_accuracy_original.csv"),
            }

        for pseudo_dir, pseudo_results in pseudo_runs:
            _evaluate_pseudo_results(
                pseudo_dir=pseudo_dir,
                pseudo_results=pseudo_results,
                alpha_values=alpha_values,
                classifier=classifier,
                device=device,
                batch_size=cfg["experiment"]["batch_size"],
                attr_names=attr_names,
                target_attributes=target_attributes,
                original_probs_df=original_probs_df,
                gt_df=gt_df,
                image_lookup_01=image_lookup_01,
                bundle=bundle,
                compute_mse=cfg["metrics"]["compute_mse"],
                compute_ssim=cfg["metrics"]["compute_ssim"],
                compute_lpips=cfg["metrics"]["compute_lpips"],
                compute_high_frequency=cfg["metrics"]["compute_high_frequency"],
                lpips_metric=lpips_metric,
                dataset_cfg=dataset_cfg,
                cfg=cfg,
                logger=logger,
                other_classifier=other_classifier,
                other_attr_names=other_attr_names,
                other_original_probs_df=other_original_probs_df,
            )
    else:
        for pseudo_dir, pseudo_results in pseudo_runs:
            _evaluate_pseudo_results(
                pseudo_dir=pseudo_dir,
                pseudo_results=pseudo_results,
                alpha_values=alpha_values,
                classifier=None,
                device=device,
                batch_size=cfg["experiment"]["batch_size"],
                attr_names=dataset.get_attr_names(),
                target_attributes=target_attributes,
                original_probs_df=None,
                gt_df=None,
                image_lookup_01=image_lookup_01,
                bundle=bundle,
                compute_mse=cfg["metrics"]["compute_mse"],
                compute_ssim=cfg["metrics"]["compute_ssim"],
                compute_lpips=cfg["metrics"]["compute_lpips"],
                compute_high_frequency=cfg["metrics"]["compute_high_frequency"],
                lpips_metric=lpips_metric,
                dataset_cfg=dataset_cfg,
                cfg=cfg,
                logger=logger,
                other_classifier=other_classifier,
                other_attr_names=other_attr_names,
                other_original_probs_df=other_original_probs_df,
            )

    summary["attribute_classifier"] = attribute_classifier_summary
    summary_path = Path(output_root) / "experiment_summary.md"
    _write_experiment_summary(summary_path, summary)

    logger.info("CelebA pseudo-counterfactual experiment complete.")


if __name__ == "__main__":
    main()
