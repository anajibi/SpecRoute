#!/usr/bin/env python
"""3-way comparison grid: same source images, same interventions, one row per model per
intervention (grouped by intervention so a reader can scan across k=1/5/11 for the same edit).
Reuses k11_labeled_grid_binned.py's per-image target/measurement logic, just looped over 3
(config, ckpt, label) tuples instead of one.
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
ROW_TAG_W = 220

MODELS = [
    ("k=1", "experiments/hdae/configs/morpho_hier_k1_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k1_v3/checkpoints/last.ckpt"),
    ("k=5", "experiments/hdae/configs/morpho_hier_k5_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k5_v3/checkpoints/last.ckpt"),
    ("k=11", "experiments/hdae/configs/morpho_hier_k11_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last.ckpt"),
]


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


def run_one_model(label, config, ckpt, imgs, attrs_raw, ds, graph, scm, predictors, categorical_attrs,
                  target_by_attr, scm_cols, scm_attr_index, gt_real, device, edit_strength, T, n_images):
    adapter = load_adapter("hdae", config, ckpt, device, edit_strength=edit_strength, T=T, compile_model=False)
    state = adapter.encode(imgs, attrs_raw, ds.attribute_names)
    recon0 = adapter.render(state)
    pred0 = predict_all(predictors, recon0 * 2 - 1, device, ATTRS)

    out_rows = {"reconstruction": [(recon0[i], two_line_label(pred0, gt_real, i)) for i in range(n_images)]}
    for attr in ATTRS:
        descendants = sorted(graph.descendants(attr))
        observed = [a for a in graph.attributes if a != attr and a not in descendants]
        if attr in categorical_attrs:
            n_classes = scm.specs[attr].num_classes
            cur_raw = attrs_raw[:, scm_cols[scm_attr_index[attr]]]
            cur_class = scm.categorical_class_index(attr, cur_raw)
            target_class = (cur_class + n_classes // 2) % n_classes
            target_tensor = scm.class_index_to_raw(attr, target_class).view(-1, 1)
        else:
            target_vals = [target_by_attr[attr][i] for i in range(n_images)]
            target_tensor = torch.tensor(target_vals, device=device).float().view(-1, 1)
        cf_attrs = scm.counterfactual(attrs_raw[:, scm_cols].float(), scm_attr_index, interventions={attr: target_tensor})
        for a in observed:
            cf_attrs[a] = attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()
        gt_cf = {name: cf_attrs[name].detach().cpu().numpy().reshape(-1) for name in ATTRS}

        cf_state = adapter.intervene(state, attr, "flip", cf_attrs)
        cf = adapter.render(cf_state)
        pred_cf = predict_all(predictors, cf * 2 - 1, device, ATTRS)
        out_rows[attr] = [(cf[i], two_line_label(pred_cf, gt_cf, i, highlight=attr)) for i in range(n_images)]
    del adapter
    torch.cuda.empty_cache()
    return out_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph_morpho.yaml")
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist_70k.h5")
    p.add_argument("--predictors-dir", default="experiments/hdae/outputs/attr_predictors_70k")
    p.add_argument("--cohorts", default="experiments/hdae/outputs/intervention_cohorts.json")
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--edit-strength", type=float, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.causal_graph) as f:
        causal_raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(causal_raw)
    scm = SCM.load(causal_raw["scm_checkpoint"], device=device)

    ds = MorphoMNISTPacked(args.packed)
    summary = json.loads((Path(args.predictors_dir) / "training_summary.json").read_text())
    predictors = {name: load_attr_predictor(summary[name]["checkpoint"], attr_col=ds.attribute_names.index(name)).to(device)
                 for name in ATTRS}

    cohorts = json.loads(Path(args.cohorts).read_text())
    fixed_indices = cohorts["_meta"]["fixed_indices"]
    sel = fixed_indices[: args.n_images]
    target_by_attr = {attr: {row["index"]: row["target_value"] for row in cohorts[attr]["per_image"]}
                      for attr in ["thickness", "intensity"]}
    categorical_attrs = {a for a in graph.attributes if scm.specs[a].kind == "categorical"}

    imgs = torch.stack([ds[int(i)]["img"] for i in sel]).to(device)
    attrs_raw = torch.stack([torch.as_tensor(ds[int(i)]["attr"]) for i in sel]).to(device)
    scm_cols = [ds.attribute_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}
    gt_real = {name: attrs_raw[:, scm_cols[scm_attr_index[name]]].detach().cpu().numpy() for name in ATTRS}

    # re-derive target_by_attr keyed by sel-position, not dataset index, and pass into run_one_model
    target_by_attr_pos = {attr: {i: target_by_attr[attr][int(sel[i])] for i in range(len(sel))}
                          for attr in ["thickness", "intensity"]}

    per_model_rows = {}
    for label, config, ckpt in MODELS:
        print(f"running {label} ...")
        per_model_rows[label] = run_one_model(label, config, ckpt, imgs, attrs_raw, ds, graph, scm, predictors,
                                              categorical_attrs, target_by_attr_pos, scm_cols, scm_attr_index,
                                              gt_real, device, args.edit_strength, args.T, len(sel))

    row_blocks = [("original", [(imgs[i], two_line_label(gt_real, gt_real, i)) for i in range(len(sel))])]
    for block_name in ["reconstruction"] + ATTRS:
        for label, _, _ in MODELS:
            tag = f"{block_name} [{label}]" if block_name == "reconstruction" else f"{block_name} -> shift/flip [{label}]"
            row_blocks.append((tag, per_model_rows[label][block_name]))

    n_rows = len(row_blocks)
    width = ROW_TAG_W + len(sel) * CELL_W
    height = n_rows * (CELL_W + LABEL_H) + 60
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    try:
        title_font = ImageFont.load_default(size=16)
        tag_font = ImageFont.load_default(size=12)
    except TypeError:
        title_font = tag_font = font
    draw.text((8, 4), f"MorphoMNIST HDAE k=1 vs k=5 vs k=11 (v3, concat_film + attr_dropout=0.08) "
                      f"-- guidance scale {args.edit_strength} -- d=digit t=thickness i=intensity h=hue",
              fill=(0, 0, 0), font=title_font)

    y0 = 32
    for row_idx, (tag, cells) in enumerate(row_blocks):
        y = y0 + row_idx * (CELL_W + LABEL_H)
        color = (0, 0, 0) if row_idx == 0 else (80, 40, 160)
        draw.text((8, y + CELL_W // 2), tag, fill=color, font=tag_font)
        for col_idx, (img_t, label_lines) in enumerate(cells):
            x = ROW_TAG_W + col_idx * CELL_W
            draw_cell(canvas, draw, font, Image.fromarray(
                img_t.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()), label_lines, x, y)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
