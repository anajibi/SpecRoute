#!/usr/bin/env python
"""Preservation-Consistency (PCF) evaluation for conditioned HDAE counterfactuals."""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3];
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.hdae.counterfactuals.attr_classifier import load_classifier
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_index_space
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.hdae.grid_utils import save_labeled_grid

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


class AttributeCFGWrapper(torch.nn.Module):
    """Experimental attribute-CFG wrapper for sampler-time dual forward passes."""

    def __init__(self, base_model, guidance_scale: float):
        super().__init__()
        self.base_model = base_model
        self.guidance_scale = float(guidance_scale)

    def forward(self, x, t, cond, **kwargs):
        cond_out = self.base_model.forward(x=x, t=t, cond=cond, **kwargs)
        if self.guidance_scale == 1.0:
            return cond_out
        y_null = torch.full_like(cond["y_idx"], 2)
        null_cond = {"zs": cond["zs"], "y_idx": y_null}
        uncond_out = self.base_model.forward(x=x, t=t, cond=null_cond, **kwargs)
        guided = uncond_out.pred + self.guidance_scale * (cond_out.pred - uncond_out.pred)
        return cond_out.__class__(pred=guided, cond=cond)


def encode_stochastic_with_model(module, model, x, cond, T=None):
    """DDIM reverse using the same model instance that produced ``cond``."""
    sampler = module.eval_sampler if T is None else module.conf._make_diffusion_conf(T).make_sampler()
    out = sampler.ddim_reverse_sample_loop(model, x, model_kwargs={"cond": cond})
    return out["sample"]


def render_with_attribute_cfg(module, model, noise, cond, T=None, guidance_scale: float = 1.0):
    """Render with the same model instance that produced ``noise`` and ``cond``."""
    sampler = module.eval_sampler if T is None else module.conf._make_diffusion_conf(T).make_sampler()
    render_model = model if guidance_scale == 1.0 else AttributeCFGWrapper(model, guidance_scale).to(noise.device).eval()
    with torch.inference_mode():
        pred_img = sampler.sample(model=render_model, noise=noise, model_kwargs={"cond": cond})
    return (pred_img + 1) / 2


def parse_csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def safe_model_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def conditioning_attr_indices(model, dataset_attr_names):
    e = model.hdae_conf.encoder
    attrs = list(e.conditioning_attrs) if e.conditioning_attrs else list(dataset_attr_names[:e.n_attributes])
    missing = [a for a in attrs if a not in dataset_attr_names]
    if missing:
        raise ValueError(f"conditioning_attrs not found in dataset attributes: {missing}")
    return attrs, [dataset_attr_names.index(a) for a in attrs]


def assert_partition(all_attrs, modeled):
    if len(all_attrs) != 40:
        raise ValueError(f"expected exactly 40 CelebA attributes, got {len(all_attrs)}")
    modeled = list(modeled)
    missing = [a for a in modeled if a not in all_attrs]
    if missing:
        raise ValueError(f"modeled attributes missing from CelebA attributes: {missing}")
    if len(set(modeled)) != len(modeled):
        raise ValueError(f"modeled attributes contain duplicates: {modeled}")
    unmodeled = [a for a in all_attrs if a not in set(modeled)]
    if set(modeled) & set(unmodeled) or len(modeled) + len(unmodeled) != 40:
        raise AssertionError("modeled/unmodeled partition must be disjoint and cover all 40 attributes")
    return modeled, unmodeled


def batched(seq, n):
    for start in range(0, len(seq), n):
        yield seq[start:start + n]


def evaluate_cohort_classifier_accuracy(cohorts, dataset, classifier, attr_names, thresholds, batch_size, device, out_path):
    """Evaluate frozen attribute-CNN accuracy on every unique real image in the fixed cohorts."""
    all_indices = sorted({int(idx)
                          for attr_data in cohorts.get("attributes", {}).values()
                          for side in ("pos_idx", "neg_idx")
                          for idx in attr_data.get(side, [])})
    logging.info("evaluating cohort classifier accuracy on %d unique real images", len(all_indices))
    correct = {name: 0 for name in attr_names}
    total = len(all_indices)
    for ids in batched(all_indices, batch_size):
        imgs = torch.stack([dataset[i]["img"] for i in ids]).to(device)
        gt = torch.stack([torch.as_tensor(dataset[i]["attr"]) for i in ids]).cpu().numpy()
        probs = classifier_probs(classifier, imgs)
        preds = (probs >= thresholds).astype(np.int8)
        gt01 = (gt > 0).astype(np.int8)
        for j, name in enumerate(attr_names):
            correct[name] += int((preds[:, j] == gt01[:, j]).sum())
    rows = [{"attribute": name,
             "accuracy": float(correct[name] / total) if total else float("nan"),
             "samples_evaluated": total,
             "correct_predictions": correct[name]}
            for name in attr_names]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attribute", "accuracy", "samples_evaluated", "correct_predictions"])
        writer.writeheader();
        writer.writerows(rows)
    logging.info("wrote cohort classifier accuracy for all attributes to %s", out_path)


def classifier_probs(classifier, x):
    with torch.inference_mode():
        return torch.sigmoid(classifier(x)).detach().cpu().numpy()


def rendered_to_classifier_input(x01):
    return x01


def compute_baselines_and_weights(attrs01, attr_names, modeled, cache_path):
    """Natural co-occurrence baseline and inverse-prevalence intervention weights."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        logging.info("loading cached correlation baseline from %s", cache_path)
        return json.loads(cache_path.read_text())
    logging.info("building correlation baseline from full label matrix: %s", cache_path)
    name_to_idx = {n: i for i, n in enumerate(attr_names)}
    out = {"attribute_names": attr_names, "baseline": {}, "intervention_weights": {}}
    for a in modeled:
        ai = name_to_idx[a]
        a_pos = attrs01[:, ai] == 1
        pos_count = int(a_pos.sum());
        neg_count = int((~a_pos).sum())
        out["intervention_weights"][a] = {
            "positive": float(pos_count / neg_count) if neg_count else 0.0,
            # neg -> pos weighted by target/source prevalence
            "negative": float(neg_count / pos_count) if pos_count else 0.0,  # pos -> neg
            "positive_count": pos_count,
            "negative_count": neg_count,
        }
        out["baseline"][a] = {}
        for direction, source_mask, target_mask in [("positive", ~a_pos, a_pos), ("negative", a_pos, ~a_pos)]:
            vals = {}
            for u in attr_names:
                if u == a:
                    continue
                ui = name_to_idx[u]
                p_source = float(attrs01[source_mask, ui].mean()) if source_mask.any() else 0.0
                p_target = float(attrs01[target_mask, ui].mean()) if target_mask.any() else 0.0
                vals[u] = abs(p_target - p_source)
            out["baseline"][a][direction] = vals
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2))
    return out


def pareto_frontier(points):
    pts = sorted(points, key=lambda x: (x[0], x[1]))
    frontier = []
    best_fc = -1.0
    for cc, fc, label in reversed(pts):
        if fc > best_fc:
            frontier.append((cc, fc, label));
            best_fc = fc
    return list(reversed(frontier))


def frontier_area(frontier):
    if len(frontier) < 2:
        return 0.0
    xs = np.asarray([p[0] for p in frontier], dtype=float)
    ys = np.asarray([p[1] for p in frontier], dtype=float)
    return float(np.trapz(ys, xs))


def save_frontier_plot(rows, path, title):
    import matplotlib.pyplot as plt
    points = [(float(r["CC"]), float(r["FC_success"]), f"{r['attribute']} {r['direction']}") for r in rows]
    frontier = pareto_frontier(points)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([p[0] for p in points], [p[1] for p in points])
    for cc, fc, label in points:
        ax.annotate(label, (cc, fc), fontsize=8, xytext=(3, 3), textcoords="offset points")
    if frontier:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], marker="o")
    ax.set(xlabel="Counterfactual Consistency (CC)", ylabel="Factual Consistency on successes (FC)", xlim=(0, 1.02),
           ylim=(0, 1.02), title=title)
    ax.grid(True, alpha=0.3);
    fig.tight_layout();
    fig.savefig(path, dpi=160);
    plt.close(fig)


def evaluate_intervention(module, classifier, ds, indices, cond_indices, cond_attrs, attr_names, attr, direction, batch_size, T, device, grid_images=0, guidance_scale=1.0):
    target_idx = attr_names.index(attr); target_col = cond_attrs.index(attr)
    loader = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    bases, edits = [], []
    grid_parts = [[], [], []]
    grid_count = 0
    for batch in loader:
        x = batch["img"].to(device)
        model = torch.compile(module.ema_model)
        y_raw = batch["attr"][:, cond_indices].to(device)
        y_idx = to_index_space(y_raw, model.hdae_conf.encoder.attr_input_range).to(device)
        with torch.inference_mode():
            zs = [z.clone() for z in model.encode(x)]
            source_cond = model.make_cond(zs, y_idx)
            x_t = encode_stochastic_with_model(module, model, x, source_cond, T=T)
            recon0 = render_with_attribute_cfg(module, model, x_t, source_cond, T=T, guidance_scale=guidance_scale)
            y_cf = y_idx.clone(); y_cf[:, target_col] = 1 if direction == "positive" else 0
            cf = render_with_attribute_cfg(module, model, x_t, model.make_cond(zs, y_cf), T=T, guidance_scale=guidance_scale)
        bases.append(classifier_probs(classifier, rendered_to_classifier_input(recon0)))
        edits.append(classifier_probs(classifier, rendered_to_classifier_input(cf)))
    if grid_images and grid_count < int(grid_images):
        take = min(int(grid_images) - grid_count, len(x))
        grid_parts[0].append(x[:take].add(1).div(2).detach().cpu())
        grid_parts[1].append(recon0[:take].clamp(0, 1).detach().cpu())
        grid_parts[2].append(cf[:take].clamp(0, 1).detach().cpu())
        grid_count += take
    grid_triplet = tuple(torch.cat(parts, dim=0) for parts in grid_parts) if grid_count else None
    return np.concatenate(bases, 0), np.concatenate(edits, 0), grid_triplet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True);
    p.add_argument("--ckpt", required=True);
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--cohorts", required=True);
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="hdae");
    p.add_argument("--batch-size", type=int, default=64);
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--baseline-cache", default=None)
    p.add_argument("--grid-images", type=int, default=8, help="Number of examples per grid row for each intervention.")
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="Experimental attribute-CFG scale for rendering; defaults to conditioning.cfg_guidance_scale from config.")
    args = p.parse_args()
    out = Path(args.output_dir);
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_hdae_config(args.config);
    data = cfg.raw["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    guidance_scale = float(args.guidance_scale if args.guidance_scale is not None else cfg.hdae_conf.conditioning.cfg_guidance_scale)
    if guidance_scale < 1.0:
        raise ValueError("--guidance-scale / conditioning.cfg_guidance_scale must be >= 1.0")
    logging.info("loading HDAE checkpoint=%s on device=%s guidance_scale=%.3f", args.ckpt, device, guidance_scale)
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    classifier, clf_state = load_classifier(args.attr_classifier, device=device)
    attr_names = [str(x) for x in clf_state["attribute_names"]]

    thr_map = clf_state.get("thresholds", {}) or {}
    thresholds = np.array([thr_map.get(name, 0.5) for name in attr_names], dtype=np.float32)
    logging.info("using per-attribute thresholds: %d calibrated, %d defaulted to 0.5",
                 sum(n in thr_map for n in attr_names), sum(n not in thr_map for n in attr_names))

    ds = CelebAHQPacked(data["lmdb_path"], data["attr_npz"], flip=False)
    cond_attrs, cond_indices = conditioning_attr_indices(module.ema_model, ds.attribute_names)
    modeled, unmodeled = assert_partition(attr_names, cond_attrs)
    logging.info("attribute partition fixed: modeled=%s unmodeled_count=%d", modeled, len(unmodeled))
    cohorts_doc = json.loads(Path(args.cohorts).read_text())
    evaluate_cohort_classifier_accuracy(cohorts_doc, ds, classifier, attr_names, thresholds, args.batch_size, device,
                                        out / "cohort_classifier_accuracy.csv")
    cohorts = cohorts_doc["attributes"]
    attrs01 = (np.load(data["attr_npz"], allow_pickle=True)["attrs"] > 0).astype(np.int8)
    cache = compute_baselines_and_weights(attrs01, attr_names, modeled,
                                          args.baseline_cache or out / "correlation_baseline.json")
    rows = []
    grid_rows, grid_labels = [], []
    for attr in modeled:
        for direction in ["positive", "negative"]:
            indices = cohorts[attr]["neg_idx" if direction == "positive" else "pos_idx"]
            logging.info("evaluating intervention attr=%s direction=%s source_n=%d", attr, direction, len(indices))
            base, edit, grid_triplet = evaluate_intervention(module, classifier, ds, indices, cond_indices, cond_attrs, attr_names, attr, direction, args.batch_size, args.T or cfg.raw["train"]["T_eval"], device, grid_images=args.grid_images, guidance_scale=guidance_scale)
            if grid_triplet is not None:
                orig_row, recon_row, cf_row = grid_triplet
                grid_rows.extend([orig_row, recon_row, cf_row])
                grid_labels.extend([f"{attr} {direction} original", f"{attr} {direction} recon0", f"{attr} {direction} cf"])
            b = base >= thresholds
            e = edit >= thresholds
            ti = attr_names.index(attr)
            valid = (~b[:, ti]) if direction == "positive" else b[:, ti]
            success = valid & (e[:, ti] if direction == "positive" else ~e[:, ti])
            fail = valid & ~success
            cc = float(success.sum() / valid.sum()) if valid.any() else 0.0
            un_idx = [attr_names.index(u) for u in unmodeled]
            expected = np.asarray([cache["baseline"][attr][direction][u] for u in unmodeled])

            # def fc_for(mask):
            #     if not mask.any(): return 1.0, np.zeros(len(un_idx))
            #     observed = (b[mask][:, un_idx] != e[mask][:, un_idx]).mean(axis=0)
            #     excess = np.maximum(0.0, observed - expected)
            #     return float(1.0 - excess.mean()), excess

            def fc_for(mask):
                if not mask.any():
                    return 1.0, np.zeros(len(un_idx))
                observed = (b[mask][:, un_idx] != e[mask][:, un_idx]).mean(axis=0)
                return float(1.0 - observed.mean()), observed

            fc_s, excess_s = fc_for(success);
            fc_f, _ = fc_for(fail)
            pcf = 0.0 if cc + fc_s == 0 else float(2 * cc * fc_s / (cc + fc_s))
            rows.append({"model": args.model_name, "guidance_scale": guidance_scale, "attribute": attr, "direction": direction, "CC": cc, "FC_success": fc_s, "FC_fail": fc_f, "PCF": pcf, "n": int(valid.sum()), "cohort_n": len(indices), "weight": cache["intervention_weights"][attr][direction], "excess_sum_success": float(excess_s.sum()), "success_n": int(success.sum()), "unmodeled_count": len(unmodeled)})
            logging.info("result attr=%s direction=%s CC=%.4f FC_success=%.4f FC_fail=%.4f PCF=%.4f n=%d", attr, direction, cc, fc_s, fc_f, pcf, int(valid.sum()))

    if grid_rows:
        save_labeled_grid(grid_rows, grid_labels, out / "pcf_experiments_grid.png", label_width=260)
        logging.info("wrote PCF experiment grid with %d rows and %d images per row to %s", len(grid_rows),
                     grid_rows[0].shape[0], out / "pcf_experiments_grid.png")
    with open(out / "pcf_per_intervention.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()));
        w.writeheader();
        w.writerows(rows)
    weight_sum = sum(r["weight"] for r in rows) or 1.0
    macro = float(np.mean([r["PCF"] for r in rows]))
    weighted = float(sum(r["PCF"] * r["weight"] for r in rows) / weight_sum)
    global_cc = sum(r["CC"] * r["n"] for r in rows) / (sum(r["n"] for r in rows) or 1)
    total_un = sum(r["success_n"] * r["unmodeled_count"] for r in rows) or 1
    global_fc = 1.0 - (sum(r["excess_sum_success"] * r["success_n"] for r in rows) / total_un)
    micro = 0.0 if global_cc + global_fc == 0 else float(2 * global_cc * global_fc / (global_cc + global_fc))
    points = [(r["CC"], r["FC_success"], f"{r['attribute']} {r['direction']}") for r in rows]
    area = frontier_area(pareto_frontier(points))
    agg = {"model": args.model_name, "guidance_scale": guidance_scale, "macro_PCF": macro, "micro_PCF": micro, "weighted_PCF": weighted, "frontier_area": area, "micro_macro_gap": micro - macro}
    with open(out / "pcf_aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()));
        w.writeheader();
        w.writerow(agg)
    save_frontier_plot(rows, out / f"frontier_{safe_model_name(args.model_name)}.png",
                       f"CC-FC frontier: {args.model_name}")
    logging.info("wrote PCF outputs to %s; micro-macro gap=%.4f", out, agg["micro_macro_gap"])


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
