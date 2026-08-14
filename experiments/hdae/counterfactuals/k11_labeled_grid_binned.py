#!/usr/bin/env python
"""Per-image labeled counterfactual grid using the precomputed bin-based flip-intervention
cohorts (precompute_intervention_cohorts.py), not a fixed p15/p85 target.

Each image cell shows two lines: "m:" (measured -- the independently-trained CNN predictor's
reading of the actual rendered/real pixels) and "gt:" (ground truth -- for original/reconstruction
rows, the real logged attribute value from the dataset; for intervention rows, the value actually
fed to the model as conditioning, i.e. the intervened attribute's target, its causal descendant's
SCM-propagated value, and every other attribute's pinned observed value). These are two genuinely
different numbers -- "m" can diverge from "gt" either because a predictor has its own measurement
error, or because the model failed to render what it was asked to condition on; comparing them
side by side is the point.

For thickness/intensity/hue, each image's intervention target is looked up from
intervention_cohorts.json -- individually computed per image (opposite population half, >=2 bins
away, target bin's midpoint), one flip row per continuous attribute. Digit is unchanged: forced
shift to (digit+5)%10.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from experiments.hdae.counterfactuals import hdae_adapter  # noqa: F401 -- registers "hdae"
from experiments.hdae.counterfactuals.cf_contract import load_adapter
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.attr_predictor import load_attr_predictor
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

ATTRS = ["digit", "thickness", "intensity", "hue"]
FMT = {"digit": lambda v: f"{v:.0f}", "thickness": lambda v: f"{v:.2f}",
      "intensity": lambda v: f"{v:.0f}", "hue": lambda v: f"{v:.2f}"}
CELL_W = 150
LABEL_H = 46
ROW_TAG_W = 200


def predict_all(predictors, img_m11, device, order):
    x = img_m11.to(device)
    return {name: predictors[name].predict_raw(x).numpy() for name in order}


def one_line(prefix, values, i, highlight=None):
    parts = [prefix]
    for name in ATTRS:
        v = FMT[name](float(values[name][i]))
        parts.append(f"*{name[0]}={v}*" if name == highlight else f"{name[0]}={v}")
    return " ".join(parts)


def two_line_label(measured, ground_truth, i, highlight=None):
    return one_line("m:", measured, i, highlight), one_line("gt:", ground_truth, i, highlight)


def draw_cell(canvas, draw, font, img_pil, label_lines, x, y):
    img_resized = img_pil.resize((CELL_W, CELL_W), Image.NEAREST)
    canvas.paste(img_resized, (x, y))
    line1, line2 = label_lines
    draw.text((x + 2, y + CELL_W + 1), line1, fill=(20, 20, 20), font=font)
    draw.text((x + 2, y + CELL_W + 1 + 13), line2, fill=(90, 90, 90), font=font)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/hdae/configs/morpho_hier_k11_v2.yaml")
    p.add_argument("--ckpt", default="experiments/hdae/outputs/morpho_hier_k11_v2/checkpoints/last.ckpt")
    p.add_argument("--model-label", default="k=11 (v2, finalized dataset)")
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph_morpho.yaml")
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist_70k.h5")
    p.add_argument("--predictors-dir", default="experiments/hdae/outputs/attr_predictors_70k")
    p.add_argument("--cohorts", default="experiments/hdae/outputs/intervention_cohorts.json")
    p.add_argument("--n-images", type=int, default=5)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--edit-strength", type=float, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.causal_graph) as f:
        causal_raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(causal_raw)
    scm = SCM.load(causal_raw["scm_checkpoint"], device=device)
    adapter = load_adapter("hdae", args.config, args.ckpt, device, edit_strength=args.edit_strength, T=args.T)

    ds = MorphoMNISTPacked(args.packed)
    summary = json.loads((Path(args.predictors_dir) / "training_summary.json").read_text())
    predictors = {name: load_attr_predictor(summary[name]["checkpoint"], attr_col=ds.attribute_names.index(name)).to(device)
                 for name in ATTRS}

    cohorts = json.loads(Path(args.cohorts).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"]
    sel = fixed_indices[:args.n_images]

    # hue is now categorical (2026-08-11) -- shift-based like digit, not a cohort bin-flip target;
    # only thickness/intensity remain in the continuous bin-flip cohort file.
    target_by_attr = {attr: {row["index"]: row["target_value"] for row in cohorts[attr]["per_image"]}
                      for attr in ["thickness", "intensity"]}
    categorical_attrs = {a for a in graph.attributes if scm.specs[a].kind == "categorical"}

    imgs = torch.stack([ds[int(i)]["img"] for i in sel]).to(device)
    attrs_raw = torch.stack([torch.as_tensor(ds[int(i)]["attr"]) for i in sel]).to(device)
    scm_cols = [ds.attribute_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}

    gt_real = {name: attrs_raw[:, scm_cols[scm_attr_index[name]]].detach().cpu().numpy() for name in ATTRS}

    state = adapter.encode(imgs, attrs_raw, ds.attribute_names)
    recon0 = adapter.render(state)
    pred_real = predict_all(predictors, imgs, device, ATTRS)
    pred0 = predict_all(predictors, recon0 * 2 - 1, device, ATTRS)

    row_blocks = []
    row_blocks.append(("original", [(imgs[i], two_line_label(pred_real, gt_real, i)) for i in range(args.n_images)]))
    row_blocks.append(("reconstruction", [(recon0[i], two_line_label(pred0, gt_real, i)) for i in range(args.n_images)]))

    for attr in ATTRS:
        descendants = sorted(graph.descendants(attr))
        observed = [a for a in graph.attributes if a != attr and a not in descendants]
        if attr in categorical_attrs:
            n_classes = scm.specs[attr].num_classes
            cur_raw = attrs_raw[:, scm_cols[scm_attr_index[attr]]]
            # raw -> class index -> shift -> back to a valid raw-units target (digit: raw value
            # already IS the class index; hue: raw value is a bin-center float and needs lo/hi
            # binning, or the shift target ends up out of [0,1] range -- see causal/scm.py).
            cur_class = scm.categorical_class_index(attr, cur_raw)
            target_class = (cur_class + n_classes // 2) % n_classes
            target_tensor = scm.class_index_to_raw(attr, target_class).view(-1, 1)
            tag = f"{attr} -> shift"
        else:
            target_vals = [target_by_attr[attr][int(i)] for i in sel]
            target_tensor = torch.tensor(target_vals, device=device).float().view(-1, 1)
            tag = f"{attr} -> flip (binned)"
        cf_attrs = scm.counterfactual(attrs_raw[:, scm_cols].float(), scm_attr_index, interventions={attr: target_tensor})
        for a in observed:
            cf_attrs[a] = attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()
        gt_cf = {name: cf_attrs[name].detach().cpu().numpy().reshape(-1) for name in ATTRS}

        cf_state = adapter.intervene(state, attr, "flip", cf_attrs)
        cf = adapter.render(cf_state)
        pred_cf = predict_all(predictors, cf * 2 - 1, device, ATTRS)
        row_blocks.append((tag, [(cf[i], two_line_label(pred_cf, gt_cf, i, highlight=attr)) for i in range(args.n_images)]))

    n_rows = len(row_blocks)
    width = ROW_TAG_W + args.n_images * CELL_W
    height = n_rows * (CELL_W + LABEL_H) + 60
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    try:
        title_font = ImageFont.load_default(size=16)
        tag_font = ImageFont.load_default(size=13)
    except TypeError:
        title_font = tag_font = font
    draw.text((8, 4), f"MorphoMNIST HDAE {args.model_label} -- guidance scale {args.edit_strength} "
                      f"-- binned flip interventions -- d=digit t=thickness i=intensity h=hue "
                      f"(m=measured by predictor, gt=given to model / ground truth, * = intervened)",
              fill=(0, 0, 0), font=title_font)

    y0 = 32
    for row_idx, (tag, cells) in enumerate(row_blocks):
        y = y0 + row_idx * (CELL_W + LABEL_H)
        draw.text((8, y + CELL_W // 2), tag, fill=(80, 40, 160) if "->" in tag else (0, 0, 0), font=tag_font)
        for col_idx, (img_t, label_lines) in enumerate(cells):
            x = ROW_TAG_W + col_idx * CELL_W
            draw_cell(canvas, draw, font, Image.fromarray(
                img_t.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()), label_lines, x, y)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
