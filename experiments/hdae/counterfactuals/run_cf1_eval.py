#!/usr/bin/env python
"""Model-agnostic Counterfactual-F1 (CF1) evaluation, driven by a CFModelAdapter + causal SCM.

CC (Counterfactual Consistency) = did the intervened attribute flip to the
target side, AND did its causal descendants (per causal_graph.yaml, TODO
item 2) move to the value the fitted SCM (causal/scm.py) predicts for them
under the intervention -- pooled into one success rate over
{intervened attribute} union {descendants}.

FC (Factual Consistency) is reported over two disjoint attribute pools,
split by whether the causal graph makes a claim about them:
- FC_observed: the graph's other conditioning attributes that are NOT
  descendants of the intervened one -- expected to stay exactly fixed,
  scored strictly (the graph "observes"/declares their causal position).
- FC_unobserved: the 36 non-conditioning CelebA attributes, outside the
  declared graph entirely -- no causal claim, same flip-rate computation as
  the pre-item-2 FC.

CF1_observed / CF1_unobserved = harmonic mean (F1-style) of CC and each FC
pool, reported macro/micro/weighted. This replaces the item-1 raw/corr
(correlation-baseline-corrected) split -- item 2 supersedes correlational
FC-adjustment with the causal one; see experiments/hdae/AGENDA.md Sec.9/11.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Adapter registration side effects.
from experiments.hdae.counterfactuals import hdae_adapter  # noqa: F401
from experiments.hdae.counterfactuals import diffae_adapter  # noqa: F401
from experiments.hdae.counterfactuals.attr_classifier import load_classifier
from experiments.hdae.counterfactuals.cf_contract import load_adapter
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.grid_utils import save_labeled_grid

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def safe_model_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


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
        writer.writeheader()
        writer.writerows(rows)
    logging.info("wrote cohort classifier accuracy for all attributes to %s", out_path)


def classifier_probs(classifier, x):
    with torch.inference_mode():
        return torch.sigmoid(classifier(x)).detach().cpu().numpy()


def pareto_frontier(points):
    pts = sorted(points, key=lambda x: (x[0], x[1]))
    frontier = []
    best_fc = -1.0
    for cc, fc, label in reversed(pts):
        if fc > best_fc:
            frontier.append((cc, fc, label))
            best_fc = fc
    return list(reversed(frontier))


def frontier_area(frontier):
    if len(frontier) < 2:
        return 0.0
    xs = np.asarray([p[0] for p in frontier], dtype=float)
    ys = np.asarray([p[1] for p in frontier], dtype=float)
    return float(np.trapz(ys, xs))


def cf1(cc, fc):
    return 0.0 if cc + fc == 0 else float(2 * cc * fc / (cc + fc))


def save_frontier_plot(rows, path, title):
    import matplotlib.pyplot as plt
    points = [(float(r["CC"]), float(r["FC_success_unobserved"]), f"{r['attribute']} {r['direction']}") for r in rows]
    frontier = pareto_frontier(points)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([p[0] for p in points], [p[1] for p in points])
    for cc, fc, label in points:
        ax.annotate(label, (cc, fc), fontsize=8, xytext=(3, 3), textcoords="offset points")
    if frontier:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], marker="o")
    ax.set(xlabel="Counterfactual Consistency (CC)", ylabel="Factual Consistency on successes (FC, unobserved pool)",
           xlim=(0, 1.02), ylim=(0, 1.02), title=title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_intervention(adapter, classifier, scm, graph, ds, indices, attr_names, attr, direction, batch_size,
                          device, grid_images=0):
    loader = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    bases, edits = [], []
    cf_attrs_all = {node: [] for node in graph.attributes}
    grid_parts = [[], [], []]
    grid_count = 0
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}
    scm_cols = [attr_names.index(a) for a in graph.attributes]
    target_val = 1.0 if direction == "positive" else 0.0

    for batch in loader:
        x = batch["img"].to(device)
        attrs_raw = batch["attr"].to(device)
        attrs01_scm = (attrs_raw[:, scm_cols] > 0).float()
        target_tensor = torch.full((x.shape[0], 1), target_val, device=device)
        cf_attrs = scm.counterfactual_binary(attrs01_scm, scm_attr_index, interventions={attr: target_tensor})
        state = adapter.encode(x, attrs_raw, attr_names)
        recon0 = adapter.render(state)
        cf_state = adapter.intervene(state, attr, direction, cf_attrs)
        cf = adapter.render(cf_state)
        bases.append(classifier_probs(classifier, recon0))
        edits.append(classifier_probs(classifier, cf))
        for node in graph.attributes:
            cf_attrs_all[node].append(cf_attrs[node].detach().cpu().numpy().reshape(-1))
        if grid_images and grid_count < int(grid_images):
            take = min(int(grid_images) - grid_count, len(x))
            grid_parts[0].append(x[:take].add(1).div(2).clamp(0, 1).detach().cpu())
            grid_parts[1].append(recon0[:take].clamp(0, 1).detach().cpu())
            grid_parts[2].append(cf[:take].clamp(0, 1).detach().cpu())
            grid_count += take
    grid_triplet = tuple(torch.cat(parts, dim=0) for parts in grid_parts) if grid_count else None
    cf_attrs_concat = {node: np.concatenate(vals, 0) for node, vals in cf_attrs_all.items()}
    return np.concatenate(bases, 0), np.concatenate(edits, 0), cf_attrs_concat, grid_triplet


def fc_for_pool(b, e, mask, cols):
    """Mean-flip-rate-derived FC over the attribute columns in ``cols``, among images in ``mask``."""
    if not cols:
        return 1.0, np.zeros(0)
    if not mask.any():
        return 1.0, np.zeros(len(cols))
    flips = (b[mask][:, cols] != e[mask][:, cols]).mean(axis=0)
    return float(1.0 - flips.mean()), flips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", default="hdae", help="Registered CFModelAdapter key (see cf_contract.py).")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--attr-classifier", required=True)
    p.add_argument("--cohorts", required=True)
    p.add_argument("--lmdb-path", required=True, help="Packed image LMDB shared across all models being compared.")
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph.yaml",
                   help="Shared causal-DAG config over the conditioning attributes (TODO item 2).")
    p.add_argument("--scm-checkpoint", default=None,
                   help="Fitted SCM (experiments/hdae/causal/train_scm.py); defaults to the causal-graph "
                        "config's scm_checkpoint field.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="hdae")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--grid-images", type=int, default=8, help="Number of examples per grid row for each intervention.")
    p.add_argument("--edit-strength", type=float, default=None,
                   help="Model-specific edit-strength hyperparameter (e.g. HDAE attribute-CFG guidance scale); "
                        "defaults to the adapter's own config-driven default.")
    p.add_argument("--max-images", type=int, default=None,
                   help="Cap the number of source images per (attribute, direction) cohort; for smoke-testing.")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.causal_graph) as f:
        causal_raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(causal_raw)
    scm_ckpt = args.scm_checkpoint or causal_raw["scm_checkpoint"]
    logging.info("loading SCM checkpoint=%s attributes=%s edges=%s", scm_ckpt, graph.attributes, graph.edges)
    scm = SCM.load(scm_ckpt, device=device)
    yaml_eps = float(causal_raw["logit_smoothing_eps"])
    if (set(scm.graph.attributes), sorted(scm.graph.edges)) != (set(graph.attributes), sorted(graph.edges)) \
            or abs(scm.eps - yaml_eps) > 1e-12:
        raise ValueError(
            f"SCM checkpoint {scm_ckpt} was fit on attributes={scm.graph.attributes} edges={scm.graph.edges} "
            f"eps={scm.eps}, but {args.causal_graph} currently declares attributes={graph.attributes} "
            f"edges={graph.edges} eps={yaml_eps} — re-run experiments/hdae/causal/train_scm.py "
            f"(edges/eps changed since the checkpoint was fit)")

    logging.info("loading model_type=%s config=%s ckpt=%s on device=%s", args.model_type, args.config, args.ckpt, device)
    adapter = load_adapter(args.model_type, args.config, args.ckpt, device, edit_strength=args.edit_strength, T=args.T)
    logging.info("adapter modeled_attrs=%s", adapter.modeled_attrs)
    if set(adapter.modeled_attrs) != set(graph.attributes):
        raise ValueError(f"causal graph attributes {graph.attributes} must exactly match adapter.modeled_attrs "
                         f"{adapter.modeled_attrs}")

    classifier, clf_state = load_classifier(args.attr_classifier, device=device)
    attr_names = [str(x) for x in clf_state["attribute_names"]]

    thr_map = clf_state.get("thresholds", {}) or {}
    thresholds = np.array([thr_map.get(name, 0.5) for name in attr_names], dtype=np.float32)
    logging.info("using per-attribute thresholds: %d calibrated, %d defaulted to 0.5",
                 sum(n in thr_map for n in attr_names), sum(n not in thr_map for n in attr_names))

    modeled, unmodeled = assert_partition(attr_names, adapter.modeled_attrs)
    unobserved_idx = [attr_names.index(u) for u in unmodeled]
    logging.info("attribute partition fixed: modeled=%s unobserved(non-graph)_count=%d", modeled, len(unobserved_idx))

    cohorts_doc = json.loads(Path(args.cohorts).read_text())
    attr_npz = cohorts_doc["attr_npz"]
    ds = CelebAHQPacked(args.lmdb_path, attr_npz, flip=False)
    evaluate_cohort_classifier_accuracy(cohorts_doc, ds, classifier, attr_names, thresholds, args.batch_size, device,
                                        out / "cohort_classifier_accuracy.csv")

    cohorts = cohorts_doc["attributes"]
    rows = []
    grid_rows, grid_labels = [], []
    for attr in modeled:
        descendants = sorted(graph.descendants(attr))
        observed_attrs = [a for a in modeled if a != attr and a not in descendants]
        observed_idx = [attr_names.index(a) for a in observed_attrs]
        for direction in ["positive", "negative"]:
            indices = cohorts[attr]["neg_idx" if direction == "positive" else "pos_idx"]
            if args.max_images:
                indices = indices[:args.max_images]
            logging.info("evaluating intervention attr=%s direction=%s source_n=%d descendants=%s observed_attrs=%s",
                         attr, direction, len(indices), descendants, observed_attrs)
            base, edit, cf_attrs_concat, grid_triplet = evaluate_intervention(
                adapter, classifier, scm, graph, ds, indices, attr_names, attr, direction, args.batch_size, device,
                grid_images=args.grid_images)
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

            cc_num = int(success.sum())
            for d in descendants:
                dcol = attr_names.index(d)
                scm_pred_bool = cf_attrs_concat[d] > 0.5
                cc_num += int((valid & (e[:, dcol] == scm_pred_bool)).sum())
            cc_den = int(valid.sum()) * (1 + len(descendants))
            cc = float(cc_num / cc_den) if cc_den else 0.0

            fc_s_observed, obs_s_observed = fc_for_pool(b, e, success, observed_idx)
            fc_s_unobserved, obs_s_unobserved = fc_for_pool(b, e, success, unobserved_idx)
            fc_f_observed, _ = fc_for_pool(b, e, fail, observed_idx)
            fc_f_unobserved, _ = fc_for_pool(b, e, fail, unobserved_idx)

            cf1_observed = cf1(cc, fc_s_observed)
            cf1_unobserved = cf1(cc, fc_s_unobserved)
            rows.append({
                "model": args.model_name, "model_type": args.model_type, "edit_strength": adapter.edit_strength,
                "attribute": attr, "direction": direction,
                "n_descendants": len(descendants), "descendants": ";".join(descendants),
                "CC": cc, "cc_numerator": cc_num, "cc_denominator": cc_den,
                "FC_success_observed": fc_s_observed, "FC_success_unobserved": fc_s_unobserved,
                "FC_fail_observed": fc_f_observed, "FC_fail_unobserved": fc_f_unobserved,
                "CF1_observed": cf1_observed, "CF1_unobserved": cf1_unobserved,
                "n": int(valid.sum()), "cohort_n": len(indices),
                "n_observed_attrs": len(observed_idx), "n_unobserved_attrs": len(unobserved_idx),
                "observed_flip_sum_success": float(obs_s_observed.sum()),
                "unobserved_flip_sum_success": float(obs_s_unobserved.sum()),
                "success_n": int(success.sum()),
            })
            logging.info("attr=%s dir=%s CC=%.4f FC_obs=%.4f FC_unobs=%.4f CF1_obs=%.4f CF1_unobs=%.4f n=%d",
                         attr, direction, cc, fc_s_observed, fc_s_unobserved, cf1_observed, cf1_unobserved,
                         int(valid.sum()))

    if grid_rows:
        save_labeled_grid(grid_rows, grid_labels, out / "cf1_experiments_grid.png", label_width=260)
        logging.info("wrote CF1 experiment grid with %d rows and %d images per row to %s", len(grid_rows),
                     grid_rows[0].shape[0], out / "cf1_experiments_grid.png")
    with open(out / "cf1_per_intervention.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # macro (unweighted mean of per-intervention CF1), for each pool
    macro_observed = float(np.mean([r["CF1_observed"] for r in rows]))
    macro_unobserved = float(np.mean([r["CF1_unobserved"] for r in rows]))

    # weighted by SUPPORT (n), not inverse-prevalence
    w_sum = sum(r["n"] for r in rows) or 1
    weighted_observed = float(sum(r["CF1_observed"] * r["n"] for r in rows) / w_sum)
    weighted_unobserved = float(sum(r["CF1_unobserved"] * r["n"] for r in rows) / w_sum)

    # micro: pool numerators/denominators, then harmonic-mean
    global_cc = sum(r["cc_numerator"] for r in rows) / (sum(r["cc_denominator"] for r in rows) or 1)
    total_observed = sum(r["success_n"] * r["n_observed_attrs"] for r in rows) or 1
    total_unobserved = sum(r["success_n"] * r["n_unobserved_attrs"] for r in rows) or 1
    global_fc_observed = 1.0 - (sum(r["observed_flip_sum_success"] * r["success_n"] for r in rows) / total_observed)
    global_fc_unobserved = 1.0 - (sum(r["unobserved_flip_sum_success"] * r["success_n"] for r in rows) / total_unobserved)
    micro_observed = cf1(global_cc, global_fc_observed)
    micro_unobserved = cf1(global_cc, global_fc_unobserved)

    # frontiers for each pool
    def _area(fc_key):
        pts = [(r["CC"], r[fc_key], f"{r['attribute']} {r['direction']}") for r in rows]
        return frontier_area(pareto_frontier(pts))

    area_observed = _area("FC_success_observed")
    area_unobserved = _area("FC_success_unobserved")

    agg = {"model": args.model_name, "model_type": args.model_type, "edit_strength": adapter.edit_strength,
           "macro_CF1_observed": macro_observed, "macro_CF1_unobserved": macro_unobserved,
           "micro_CF1_observed": micro_observed, "micro_CF1_unobserved": micro_unobserved,
           "weighted_CF1_observed": weighted_observed, "weighted_CF1_unobserved": weighted_unobserved,
           "frontier_area_observed": area_observed, "frontier_area_unobserved": area_unobserved,
           "micro_macro_gap_observed": micro_observed - macro_observed,
           "micro_macro_gap_unobserved": micro_unobserved - macro_unobserved}

    with open(out / "cf1_aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)
    save_frontier_plot(rows, out / f"frontier_{safe_model_name(args.model_name)}.png",
                       f"CC-FC frontier: {args.model_name}")
    logging.info("wrote CF1 outputs to %s; micro-macro gap observed=%.4f unobserved=%.4f", out,
                 agg["micro_macro_gap_observed"], agg["micro_macro_gap_unobserved"])


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
