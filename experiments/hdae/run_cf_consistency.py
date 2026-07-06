#!/usr/bin/env python
"""Run counterfactual/factual consistency metrics from cached HDAE encodings.

Model specs are NAME=CONFIG,CKPT,PROBE_METRICS,PROBE_WEIGHTS_DIR. The script is
HDAE-native; external DiffAE baselines should be adapted to the same cache/edit
interface before being passed here.
"""
import argparse, csv, json, logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification


class HuggingFaceResNetWrapper(torch.nn.Module):
    """
    Optimized for 64x64 inputs. Uses a ResNet backbone which natively handles
    variable spatial resolutions without requiring aggressive, blurry upscaling.
    """

    def __init__(self, model_name="microsoft/resnet-50", num_attributes=40):
        super().__init__()
        logging.info(f"Initializing CNN backbone: {model_name}")

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.register_buffer("mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1))

        self.cnn = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_attributes,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        self.logged_range = False # Tracking flag for one-time log

    def forward(self, x):
        # 1. Deterministically convert [-1, 1] diffusion range to [0, 1]
        # We unconditionally apply this because the pipeline strictly outputs [-1, 1].
        x = x.clamp(0.0, 1.0)

        # 2. Interpolate to 128x128 to protect spatial features from pooling collapse
        x_resized = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        # 3. Apply ImageNet Normalization
        x_norm = (x_resized - self.mean) / self.std

        # ABSOLUTE FINAL DATA RANGE VERIFICATION
        if not self.logged_range:
            logging.info(f"WRAPPER INTERNAL VERIFICATION: Final ImageNet normalized input min={x_norm.min().item():.4f}, max={x_norm.max().item():.4f}")
            self.logged_range = True

        outputs = self.cnn(pixel_values=x_norm)
        return outputs.logits

from experiments.hdae.counterfactuals.directions import (
    choose_probe_row,
    direction_from_probe_checkpoint,
    probe_weight_path,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def parse_csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_models(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=config,ckpt,probe_metrics,weights, got {item!r}")
        name, spec = item.split("=", 1)
        parts = [Path(x) for x in spec.split(",")]
        if len(parts) != 4:
            raise ValueError(f"model {name!r} needs 4 comma-separated paths")
        out[name] = {"config": parts[0], "ckpt": parts[1], "probe_metrics": parts[2], "linear_probe_weights": parts[3]}
    return out


def source_indices(cohorts, attr, direction):
    side = "neg_idx" if direction == "positive" else "pos_idx"
    return [int(x) for x in cohorts["attributes"][attr][side]]


def batched(seq, n):
    for start in range(0, len(seq), n):
        yield seq[start:start + n]

def classifier_probs(pretrained_eval_classifier, x):
    """Strictly utilizes the pretrained non-linear classifier for final evaluation."""
    with torch.inference_mode():
        return torch.sigmoid(pretrained_eval_classifier(x)).detach().cpu().numpy()


def non_target_flip_fraction(base_probs, edit_probs, target_idx, mask):
    if mask.sum() == 0:
        return float("nan")
    non = [i for i in range(base_probs.shape[1]) if i != target_idx]
    flips = (base_probs[mask][:, non] >= 0.5) != (edit_probs[mask][:, non] >= 0.5)
    return float(flips.mean())


def compute_consistency(base_probs, edit_probs, target_idx, direction):
    if direction == "positive":
        source_mask = base_probs[:, target_idx] < 0.5
        success = source_mask & (edit_probs[:, target_idx] >= 0.5)
    else:
        source_mask = base_probs[:, target_idx] >= 0.5
        success = source_mask & (edit_probs[:, target_idx] < 0.5)
    fail = source_mask & ~success
    n_source = int(source_mask.sum())
    n_success = int(success.sum())
    n_fail = int(fail.sum())

    metrics = {
        "counterfactual_consistency": float(n_success / n_source) if n_source else float("nan"),
        "factual_flip_success": non_target_flip_fraction(base_probs, edit_probs, target_idx, success),
        "factual_flip_fail": non_target_flip_fraction(base_probs, edit_probs, target_idx, fail),
        "n_source": n_source, "n_success": n_success, "n_fail": n_fail
    }
    return metrics


def load_directions(probe_metrics, linear_probe_weights, attributes, levels):
    """Strictly loads mathematical directions from the linear probe files."""
    logging.info("Loading linear probe directions for vector math...")
    dirs = {}
    for attr in attributes:
        for level in levels:
            try:
                row = choose_probe_row(str(probe_metrics), attr, level=level)
                direction, _state = direction_from_probe_checkpoint(probe_weight_path(str(linear_probe_weights), row))
                dirs[(attr, level)] = direction
            except Exception as exc:
                logging.warning(f"Skipped linear direction attr={attr} level={level}: {exc}")
    logging.info(f"Loaded {len(dirs)} total linear probe directions.")
    return dirs


def cache_path(cache_dir, model_name, index):
    return Path(cache_dir) / model_name / f"{int(index):08d}.pt"


def ensure_cached(module, dataset, indices, cache_dir, model_name, T, batch_size, device):
    model = module.ema_model

    missing = [idx for idx in indices if not cache_path(cache_dir, model_name, idx).exists()]
    logging.info(f"Cache check complete: {len(indices) - len(missing)} caches found, {len(missing)} missing.")

    if not missing:
        return

    logging.info(f"Generating caches for {len(missing)} missing items.")
    (Path(cache_dir) / model_name).mkdir(parents=True, exist_ok=True)

    for ids in batched(missing, batch_size):
        imgs = torch.stack([dataset[i]["img"] for i in ids]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            encoded = model.encode(imgs)
            zs = [z.detach().cpu().float() for z in encoded["zs"]]
            cond = model.merge(encoded["zs"])
            x_t = module.encode_stochastic(imgs, cond, T=T).detach().cpu().float()
        for local, idx in enumerate(ids):
            out_path = cache_path(cache_dir, model_name, idx)
            torch.save({"index": int(idx), "zs": [z[local].clone() for z in zs],
                        "x_t": x_t[local].clone()}, out_path)


def load_cached_batch(cache_dir, model_name, indices, device):
    states = [torch.load(cache_path(cache_dir, model_name, idx), map_location="cpu", weights_only=True) for idx in
              indices]
    num_levels = len(states[0]["zs"])
    zs = [torch.stack([state["zs"][level] for state in states]).to(device) for level in range(num_levels)]
    x_t = torch.stack([state["x_t"] for state in states]).to(device)
    return zs, x_t


def score_recon0(module, pretrained_eval_classifier, zs, x_t, T, device):
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        cond = module.ema_model.merge(zs)
        recon = module.render(x_t, cond, T=T)
    return classifier_probs(pretrained_eval_classifier, recon)


def tensor_to_pil(t):
    """Calculated translation from natively rendered [0, 1] space to [0, 255] RGB."""
    # The .float() cast protects against float16 rendering artifacts
    arr = t.detach().cpu().float().clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def evaluate_cohort_accuracy(cohorts, dataset, pretrained_eval_classifier, attr_names, batch_size, device, out_path):
    """Evaluates base dataset accuracy across ALL 40 CelebA attributes using native dataset ground truth."""
    logging.info("--- Evaluating Base Cohort Pretrained Classifier Accuracy for ALL Attributes ---")

    # 1. Define the "whole cohort" of images
    # Gather all unique indices mentioned ANYWHERE in the cohorts file
    all_indices = set()
    for attr_data in cohorts.get("attributes", {}).values():
        all_indices.update(attr_data.get("pos_idx", []))
        all_indices.update(attr_data.get("neg_idx", []))

    all_indices = sorted([int(x) for x in all_indices])
    logging.info(f"Total unique cohort images to classify: {len(all_indices)}")

    # 2. Track metrics for every single CelebA attribute recognized by the classifier
    # 2. Track metrics for every single CelebA attribute recognized by the classifier
    correct_counts = {attr: 0 for attr in attr_names}
    total_counts = len(all_indices)

    log_range_once = True  # Protection flag

    for ids in batched(all_indices, batch_size):
        imgs = torch.stack([dataset[i]["img"] for i in ids]).to(device)

        # Extract native dataset ground truth for all 40 attributes
        gt_attrs = torch.stack([torch.as_tensor(dataset[i]["attr"]) for i in ids]).to(device)

        imgs_input = ((imgs + 1.0) / 2.0)

        # DATA RANGE VERIFICATION
        if log_range_once:
            logging.info(
                f"DATA RANGE VERIFICATION: Input tensor to classifier min={imgs_input.min().item():.4f}, max={imgs_input.max().item():.4f}")
            log_range_once = False

        probs = classifier_probs(pretrained_eval_classifier, imgs_input)

        # Convert sigmoid probabilities to binary predictions
        preds = (probs >= 0.5).astype(int)

        # CRITICAL FIX: Convert CelebA [-1, 1] ground truth to [0, 1]
        gt_attrs_np = (gt_attrs.cpu().numpy() > 0).astype(int)

        for b, _ in enumerate(ids):
            for attr_idx, attr_name in enumerate(attr_names):
                if preds[b, attr_idx] == gt_attrs_np[b, attr_idx]:
                    correct_counts[attr_name] += 1

    # 3. Assemble and save the comprehensive report
    rows = []
    for attr_name in attr_names:
        acc = correct_counts[attr_name] / total_counts if total_counts > 0 else float('nan')
        rows.append({
            "attribute": attr_name,
            "accuracy": acc,
            "samples_evaluated": total_counts,
            "correct_predictions": correct_counts[attr_name]
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attribute", "accuracy", "samples_evaluated", "correct_predictions"])
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Comprehensive CelebA accuracy saved to {out_path}")

def build_master_grid(edit_image_store, target_indices_map, attributes, directions, levels, out_dir, model_name):
    logging.info("Assembling precise visual master grid...")

    sample_img = None
    for level_dict in edit_image_store.values():
        if level_dict:
            sample_img = next(iter(level_dict.values()))
            break

    if not sample_img:
        logging.warning("No images collected to build grid.")
        return

    img_w, img_h = sample_img.size
    images_per_row = 5
    label_w = 400
    row_w = (img_w * images_per_row) + label_w
    row_h = img_h

    rows = []
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except IOError:
        font = ImageFont.load_default()

    for attr in attributes:
        for direction in directions:
            target_list = target_indices_map.get((attr, direction), [])
            if not target_list:
                continue

            for level in levels:
                key = (attr, direction, level)
                if key not in edit_image_store:
                    continue

                row_canvas = Image.new('RGB', (row_w, row_h), color=(255, 255, 255))

                for i, target_idx in enumerate(target_list):
                    if target_idx in edit_image_store[key]:
                        row_canvas.paste(edit_image_store[key][target_idx], (i * img_w, 0))

                draw = ImageDraw.Draw(row_canvas)
                label_text = f"Attr: {attr}\nDir: {direction}\nLatent: {level}"
                draw.text((img_w * images_per_row + 20, row_h // 2 - 20), label_text, fill=(0, 0, 0), font=font)

                rows.append(row_canvas)

    if not rows:
        return

    master_h = len(rows) * row_h
    master_canvas = Image.new('RGB', (row_w, master_h))
    for y_idx, row_img in enumerate(rows):
        master_canvas.paste(row_img, (0, y_idx * row_h))

    grid_path = Path(out_dir) / f"{model_name}_samples_grid.png"
    master_canvas.save(grid_path)
    logging.info(f"Master image grid successfully saved to: {grid_path}")


def run_model(model_name, spec, module, dataset, pretrained_eval_classifier, attr_to_idx, cohorts, attributes,
              directions, strength, T_eval, batch_size, cache_dir, output_rows, out_dir):
    logging.info(f"--- Initializing run for model: {model_name} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    levels = list(range(len(module.ema_model.hdae_conf.encoder.level_dims)))
    # Explicitly pull from the linear probe paths
    dirs = load_directions(spec["probe_metrics"], spec["linear_probe_weights"], attributes, levels)

    target_indices_map = {}
    all_indices_set = set()

    for attr in attributes:
        for direction in directions:
            req_idx = source_indices(cohorts, attr, direction)
            target_indices_map[(attr, direction)] = req_idx[:5]
            all_indices_set.update(req_idx)

    all_indices = sorted(list(all_indices_set))
    ensure_cached(module, dataset, all_indices, cache_dir, model_name, T_eval, batch_size, device)

    logging.info("Pre-calculating base probabilities for all required indices...")
    base_prob_map = {}
    for ids in batched(all_indices, batch_size):
        zs, x_t = load_cached_batch(cache_dir, model_name, ids, device)
        base_probs = score_recon0(module, pretrained_eval_classifier, zs, x_t, T_eval, device)
        for i, idx in enumerate(ids):
            base_prob_map[idx] = base_probs[i]

    logging.info("Building vectorized flat execution queue...")
    execution_queue = []

    for attr in attributes:
        for direction in directions:
            req_indices = source_indices(cohorts, attr, direction)
            sign = 1.0 if direction == "positive" else -1.0

            for level in levels:
                if (attr, level) not in dirs:
                    continue

                for idx in req_indices:
                    execution_queue.append({
                        "idx": idx, "attr": attr, "direction": direction,
                        "level": level, "dvec": dirs[(attr, level)], "sign": sign
                    })

    total_jobs = len(execution_queue)
    logging.info(f"Queue built. Total render jobs: {total_jobs}")

    results_store = {
        (attr, direction, level): {"base_probs": [], "edit_probs": []}
        for attr in attributes for direction in directions for level in levels
    }

    edit_image_store = {}

    log_diffusion_once = True # Protection flag to prevent console spam

    for batch_start in range(0, total_jobs, batch_size):
        job_batch = execution_queue[batch_start:batch_start + batch_size]
        logging.info(f"Processing render jobs {batch_start} to {batch_start + len(job_batch)} / {total_jobs}")

        batch_indices = [job["idx"] for job in job_batch]
        zs, x_t = load_cached_batch(cache_dir, model_name, batch_indices, device)

        zs_edit = [z.clone() for z in zs]

        for local_i, job in enumerate(job_batch):
            level = job["level"]
            dvec_tensor = torch.as_tensor(job["dvec"], dtype=zs[0].dtype, device=device)
            zs_edit[level][local_i] += job["sign"] * float(strength) * dvec_tensor

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            cond = module.ema_model.merge(zs_edit)

            # ABSOLUTE DIFFUSION INPUT VERIFICATION
            if log_diffusion_once:
                logging.info(f"DIFFUSION INPUT VERIFICATION: x_t min={x_t.min().item():.4f}, max={x_t.max().item():.4f}, dtype={x_t.dtype}")

            edits = module.render(x_t, cond, T=T_eval)

            # ABSOLUTE DIFFUSION OUTPUT VERIFICATION
            if log_diffusion_once:
                logging.info(f"DIFFUSION OUTPUT VERIFICATION: edits min={edits.min().item():.4f}, max={edits.max().item():.4f}, dtype={edits.dtype}")
                log_diffusion_once = False

        # Explicitly evaluate edits using the pretrained non-linear classifier
        edit_probs = classifier_probs(pretrained_eval_classifier, edits)

        for local_i, job in enumerate(job_batch):
            attr, direction, level = job["attr"], job["direction"], job["level"]
            idx = job["idx"]

            results_store[(attr, direction, level)]["base_probs"].append(base_prob_map[idx])
            results_store[(attr, direction, level)]["edit_probs"].append(edit_probs[local_i])

            if idx in target_indices_map[(attr, direction)]:
                key = (attr, direction, level)
                if key not in edit_image_store:
                    edit_image_store[key] = {}
                edit_image_store[key][idx] = tensor_to_pil(edits[local_i])

    logging.info("Aggregating final metrics...")
    for attr in attributes:
        target_idx = attr_to_idx[attr]
        for direction in directions:
            for level in levels:
                if (attr, level) not in dirs:
                    continue

                store = results_store[(attr, direction, level)]
                if not store["base_probs"]:
                    continue

                base_all = np.stack(store["base_probs"], axis=0)
                edit_all = np.stack(store["edit_probs"], axis=0)

                rec = compute_consistency(base_all, edit_all, target_idx, direction)
                output_rows.append({
                    "model": model_name, "attribute": attr, "latent_used": level,
                    "direction": direction, **rec
                })

    build_master_grid(edit_image_store, target_indices_map, attributes, directions, levels, out_dir, model_name)


def write_rows(path, rows):
    fields = ["model", "attribute", "latent_used", "direction", "counterfactual_consistency",
              "factual_flip_success", "factual_flip_fail", "n_source", "n_success", "n_fail"]
    logging.info(f"Writing {len(rows)} rows to {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts", required=True)
    p.add_argument("--models", nargs="+", required=True,
                   help="NAME=config,ckpt,probe_metrics,probe_weights_dir entries")
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--attributes", default="Smiling,Eyeglasses,Male,Young")
    p.add_argument("--directions", default="positive,negative")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--T-eval", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from experiments.hdae.counterfactuals.attribute_classifier import load_classifier
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.lit_module import HDAELitModule

    logging.info("Starting execution. Parsing arguments...")
    cohorts = json.loads(Path(args.cohorts).read_text())
    models = parse_models(args.models)
    attributes = parse_csv_list(args.attributes)
    directions = parse_csv_list(args.directions)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Loading unified Dataset and Pretrained Classifier...")
    first_spec = list(models.values())[0]
    first_cfg = load_hdae_config(str(first_spec["config"]))

    dataset = CelebAHQPacked(first_cfg.raw["data"]["lmdb_path"], first_cfg.raw["data"]["attr_npz"], flip=False)

    logging.info(f"Loading fine-tuned ResNet evaluation classifier from: {args.attr_classifier}")
    ckpt = torch.load(args.attr_classifier, map_location=device)

    # Extract attribute names saved during fine-tuning
    attr_names = [str(x) for x in ckpt["attribute_names"]]
    attr_to_idx = {name: i for i, name in enumerate(attr_names)}

    # Instantiate the wrapper and load the fine-tuned weights
    pretrained_eval_classifier = HuggingFaceResNetWrapper(num_attributes=len(attr_names)).to(device)

    # Handle the specific dictionary key your fine-tuning script used to save the model state
    state_dict = ckpt.get("state_dict", ckpt.get("model_state", ckpt))

    # Remove 'module.' prefix if the model was saved using DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # CRITICAL: Enforce strict=True. If this crashes, it means your saved weights
    # do not match the wrapper architecture, and we need to map the keys.
    pretrained_eval_classifier.load_state_dict(state_dict, strict=True)

    # Lock the classifier into evaluation mode strictly
    pretrained_eval_classifier.eval()


    # PRE-MODEL EVALUATION
    cohort_accuracy_path = out_dir / "cohort_classifier_accuracy.csv"
    evaluate_cohort_accuracy(cohorts, dataset, pretrained_eval_classifier, attr_names, args.batch_size, device,
                             cohort_accuracy_path)

    rows = []

    for model_name, spec in models.items():
        cfg = load_hdae_config(str(spec["config"]))
        T_eval = args.T_eval or cfg.raw["train"]["T_eval"]

        module = HDAELitModule.load_from_checkpoint(str(spec["ckpt"]), conf=cfg.train_conf, map_location="cpu").to(
            device).eval()

        run_model(model_name, spec, module, dataset, pretrained_eval_classifier, attr_to_idx, cohorts, attributes,
                  directions,
                  args.strength, T_eval, args.batch_size, args.cache_dir, rows, out_dir)

        del module
        torch.cuda.empty_cache()

    write_rows(args.out, rows)

    out_meta = Path(args.out).with_suffix(".json")
    logging.info(f"Writing metadata to {out_meta}")
    out_meta.write_text(json.dumps({
        "rows": len(rows), "strength": args.strength,
        "T_eval": args.T_eval, "cache_dir": args.cache_dir
    }, indent=2))

    logging.info("Script execution complete.")


if __name__ == "__main__":
    main()