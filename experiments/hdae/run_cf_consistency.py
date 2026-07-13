#!/usr/bin/env python
"""Run counterfactual/factual consistency metrics from cached HDAE encodings.

Model specs are NAME=CONFIG,CKPT. Legacy NAME=CONFIG,CKPT,PROBE_METRICS,PROBE_WEIGHTS_DIR entries are accepted, but probe paths are ignored because CF generation toggles conditioning only.
"""
import argparse, csv, json, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

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

        # 2. Apply ImageNet normalization directly at native 64x64 resolution.
        # The ResNet backbone is convolutional/global-pooling based and supports
        # variable spatial sizes, so avoid blur/cost from upsampling.
        x_norm = (x - self.mean) / self.std

        # ABSOLUTE FINAL DATA RANGE VERIFICATION
        if not self.logged_range:
            logging.info(f"WRAPPER INTERNAL VERIFICATION: Final ImageNet normalized input min={x_norm.min().item():.4f}, max={x_norm.max().item():.4f}")
            self.logged_range = True

        outputs = self.cnn(pixel_values=x_norm)
        return outputs.logits


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def parse_csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_models(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=config,ckpt, got {item!r}")
        name, spec = item.split("=", 1)
        parts = [Path(x) for x in spec.split(",")]
        if len(parts) not in {2, 4}:
            raise ValueError(f"model {name!r} needs CONFIG,CKPT (probe paths are no longer used)")
        out[name] = {"config": parts[0], "ckpt": parts[1]}
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


def non_target_flip_fraction(base_probs, edit_probs, target_idx, mask, preservation_idx=None):
    if mask.sum() == 0:
        return float("nan")
    non = list(preservation_idx) if preservation_idx is not None else [i for i in range(base_probs.shape[1]) if i != target_idx]
    flips = (base_probs[mask][:, non] >= 0.5) != (edit_probs[mask][:, non] >= 0.5)
    return float(flips.mean())


def compute_consistency(base_probs, edit_probs, target_idx, direction, preservation_idx=None):
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
    return {"counterfactual_consistency": float(n_success / n_source) if n_source else float("nan"),
            "factual_flip_success": non_target_flip_fraction(base_probs, edit_probs, target_idx, success, preservation_idx),
            "factual_flip_fail": non_target_flip_fraction(base_probs, edit_probs, target_idx, fail, preservation_idx),
            "n_source": n_source, "n_success": n_success, "n_fail": n_fail}


def cache_path(cache_dir, model_name, index):
    return Path(cache_dir) / model_name / f"{int(index):08d}.pt"


def ensure_cached(module, dataset, indices, cache_dir, model_name, T, batch_size, device, cond_indices):
    import torch
    model = module.ema_model
    missing = [idx for idx in indices if not cache_path(cache_dir, model_name, idx).exists()]
    if not missing:
        return
    (Path(cache_dir) / model_name).mkdir(parents=True, exist_ok=True)
    for ids in batched(missing, batch_size):
        imgs = torch.stack([dataset[i]["img"] for i in ids]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            from experiments.hdae.hdae.attr_utils import to_index_space
            y_raw = torch.stack([dataset[i]["attr"][cond_indices] for i in ids]).to(device)
            y_idx = to_index_space(y_raw, model.hdae_conf.encoder.attr_input_range).to(device)
            zs_live = model.encode(imgs)
            zs = [z.detach().cpu().float() for z in zs_live]
            cond = model.make_cond(zs_live, y_idx)
            x_t = module.encode_stochastic(imgs, cond, T=T).detach().cpu().float()
        for local, idx in enumerate(ids):
            torch.save({"index": int(idx), "zs": [z[local].clone() for z in zs],
                        "x_t": x_t[local].clone(), "y_idx": y_idx[local].detach().cpu().clone()}, cache_path(cache_dir, model_name, idx))


def load_cached_batch(cache_dir, model_name, indices, device):
    import torch
    states = [torch.load(cache_path(cache_dir, model_name, idx), map_location="cuda") for idx in indices]
    num_levels = len(states[0]["zs"])
    zs = [torch.stack([state["zs"][level] for state in states]).to(device) for level in range(num_levels)]
    x_t = torch.stack([state["x_t"] for state in states]).to(device)
    y_idx = torch.stack([state["y_idx"] for state in states]).to(device)
    return zs, x_t, y_idx


def score_recon0(module, classifier, zs, x_t, y_idx, T, device):
    import torch
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        recon = module.render(x_t, {"zs": zs, "y_idx": y_idx}, T=T)
    return classifier_probs(classifier, recon)

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

def run_model(args, model_name, spec, cohorts, attributes, directions, strength, T_eval, batch_size, cache_dir, attr_classifier, output_rows):
    import torch
    from experiments.hdae.counterfactuals.attribute_classifier import load_classifier
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.lit_module import HDAELitModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_hdae_config(str(spec["config"]))
    data = cfg.raw["data"]
    T = T_eval or cfg.raw["train"]["T_eval"]
    module = HDAELitModule.load_from_checkpoint(str(spec["ckpt"]), conf=cfg.train_conf, map_location="cpu").to(device).eval()
    dataset = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)

    logging.info(f"Loading fine-tuned ResNet evaluation classifier from: {args.attr_classifier}")
    out_dir = Path(args.out).parent

    ckpt = torch.load(args.attr_classifier, map_location="cuda")

    pretrained_eval_classifier = HuggingFaceResNetWrapper(num_attributes=len(dataset.attribute_names)).to(device)

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
    evaluate_cohort_accuracy(cohorts, dataset, pretrained_eval_classifier, dataset.attribute_names, args.batch_size, device,
                             cohort_accuracy_path)


    attr_names = [str(x) for x in dataset.attribute_names]
    attr_to_idx = {name: i for i, name in enumerate(attr_names)}
    cond_attrs = list(module.ema_model.hdae_conf.encoder.conditioning_attrs)
    cond_indices = [dataset.attribute_names.index(a) for a in cond_attrs]
    preservation_idx = [i for i, name in enumerate(attr_names) if name not in set(cond_attrs)]

    all_indices = sorted({idx for attr in attributes for d in directions for idx in source_indices(cohorts, attr, d)})
    ensure_cached(module, dataset, all_indices, cache_dir, model_name, T, batch_size, device, cond_indices)

    for attr in attributes:
        target_idx = attr_to_idx[attr]
        for direction in directions:
            indices = source_indices(cohorts, attr, direction)
            if attr not in cond_attrs:
                raise ValueError(f"attribute {attr!r} is not in conditioning_attrs={cond_attrs}")
            target_cond_col = cond_attrs.index(attr)
            base_all, edit_all = [], []
            for ids in batched(indices, batch_size):
                zs, x_t, y_idx = load_cached_batch(cache_dir, model_name, ids, device)
                base_probs = score_recon0(module, pretrained_eval_classifier, zs, x_t, y_idx, T, device)
                import torch
                y_cf = y_idx.clone()
                y_cf[:, target_cond_col] = 1 if direction == "positive" else 0
                with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                    edit = module.render(x_t, {"zs": zs, "y_idx": y_cf}, T=T)
                edit_probs = classifier_probs(pretrained_eval_classifier, edit)
                base_all.append(base_probs); edit_all.append(edit_probs)
            base = np.concatenate(base_all, axis=0)
            edit = np.concatenate(edit_all, axis=0)
            rec = compute_consistency(base, edit, target_idx, direction, preservation_idx)
            output_rows.append({"model": model_name, "attribute": attr, "latent_used": "conditioning",
                                "direction": direction, **rec})


def write_rows(path, rows):
    fields = ["model", "attribute", "latent_used", "direction", "counterfactual_consistency",
              "factual_flip_success", "factual_flip_fail", "n_source", "n_success", "n_fail"]
    logging.info(f"Writing {len(rows)} rows to {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts", required=True)
    p.add_argument("--models", nargs="+", required=True,
                   help="NAME=config,ckpt entries; legacy four-field specs are accepted but probe paths are ignored")
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--attributes", default="Smiling,Eyeglasses,Male,Young")
    p.add_argument("--directions", default="positive,negative")
    p.add_argument("--strength", type=float, default=1.0, help="Ignored: conditioning-only CF has no strength.")
    p.add_argument("--T-eval", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cohorts = json.loads(Path(args.cohorts).read_text())
    models = parse_models(args.models)
    rows = []


    for model_name, spec in models.items():
        run_model(args, model_name, spec, cohorts, parse_csv_list(args.attributes), parse_csv_list(args.directions),
                  args.strength, args.T_eval, args.batch_size, args.cache_dir, args.attr_classifier, rows)
    write_rows(args.out, rows)
    Path(args.out).with_suffix(".json").write_text(json.dumps({"rows": len(rows), "edit_mechanism": "conditioning_signal_only_fixed_latents",
                                                                 "T_eval": args.T_eval, "cache_dir": args.cache_dir}, indent=2))


if __name__ == "__main__":
    main()
