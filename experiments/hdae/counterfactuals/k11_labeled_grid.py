#!/usr/bin/env python
"""Per-image labeled counterfactual grid for a single model (default k=11).

Unlike `grid_utils.save_labeled_grid` (one label per row only), every image here gets its own
label showing all 4 conditioning attributes' predicted values (digit/thickness/intensity/hue,
via the independently-trained CNN predictors), not just the one being intervened on -- so a
reader can see at a glance whether an intervention leaked into the other three attributes
without cross-referencing a separate CSV.
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

CONTINUOUS_TARGETS = {"thickness": {"low": 2.513, "high": 3.210},
                      "intensity": {"low": 153.155, "high": 204.982},
                      "hue": {"low": 0.151, "high": 0.850}}
FMT = {"digit": lambda v: f"{v:.0f}", "thickness": lambda v: f"{v:.2f}",
      "intensity": lambda v: f"{v:.0f}", "hue": lambda v: f"{v:.2f}"}
CELL_W, CELL_H = 140, 140
LABEL_H = 34
ROW_TAG_W = 200


def predict_all(predictors, img_m11, device, order):
    x = img_m11.to(device)
    return {name: predictors[name].predict_raw(x).numpy() for name in order}


def attr_label(preds, i, highlight=None):
    parts = []
    for name in ["digit", "thickness", "intensity", "hue"]:
        v = FMT[name](float(preds[name][i]))
        parts.append(f"*{name[0]}={v}*" if name == highlight else f"{name[0]}={v}")
    return "  ".join(parts)


def draw_cell(canvas, draw, font, img_pil, label, x, y):
    img_resized = img_pil.resize((CELL_W, CELL_W), Image.NEAREST)
    canvas.paste(img_resized, (x, y))
    draw.text((x + 2, y + CELL_W + 2), label, fill=(20, 20, 20), font=font)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/hdae/configs/morpho_hier_k11.yaml")
    p.add_argument("--ckpt", default="experiments/hdae/outputs/morpho_hier_k11/checkpoints/last.ckpt")
    p.add_argument("--model-label", default="k=11")
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph_morpho.yaml")
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist.h5")
    p.add_argument("--predictors-dir", default="experiments/hdae/outputs/attr_predictors")
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--edit-strength", type=float, default=8.0)
    p.add_argument("--output", default="experiments/hdae/outputs/morpho_hier_k11/k11_labeled_grid.png")
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
                 for name in ["digit", "thickness", "intensity", "hue"]}

    test_idx = np.nonzero(ds.partitions == 1)[0]
    rng = np.random.RandomState(args.seed)
    sel = rng.choice(test_idx, size=args.n_images, replace=False)
    imgs = torch.stack([ds[int(i)]["img"] for i in sel]).to(device)
    attrs_raw = torch.stack([torch.as_tensor(ds[int(i)]["attr"]) for i in sel]).to(device)
    scm_cols = [ds.attribute_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}

    state = adapter.encode(imgs, attrs_raw, ds.attribute_names)
    recon0 = adapter.render(state)
    pred0 = predict_all(predictors, recon0 * 2 - 1, device, ["digit", "thickness", "intensity", "hue"])

    interventions = [("digit", "shift")]
    for attr in ["thickness", "intensity", "hue"]:
        interventions += [(attr, "high"), (attr, "low")]

    row_blocks = []  # (row_tag, [(img_tensor, label), ...]) per sub-row
    row_blocks.append(("original", [(imgs[i], attr_label(
        predict_all(predictors, imgs, device, ["digit", "thickness", "intensity", "hue"]), i)) for i in range(args.n_images)]))
    row_blocks.append(("reconstruction", [(recon0[i], attr_label(pred0, i)) for i in range(args.n_images)]))

    for attr, direction in interventions:
        descendants = sorted(graph.descendants(attr))
        observed = [a for a in graph.attributes if a != attr and a not in descendants]
        if attr == "digit":
            cur = attrs_raw[:, scm_cols[scm_attr_index["digit"]]]
            target_tensor = ((cur.long() + 5) % 10).float().view(-1, 1)
        else:
            target_tensor = torch.full((args.n_images, 1), CONTINUOUS_TARGETS[attr][direction], device=device)
        cf_attrs = scm.counterfactual(attrs_raw[:, scm_cols].float(), scm_attr_index, interventions={attr: target_tensor})
        for a in observed:
            cf_attrs[a] = attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()
        cf_state = adapter.intervene(state, attr, direction, cf_attrs)
        cf = adapter.render(cf_state)
        pred_cf = predict_all(predictors, cf * 2 - 1, device, ["digit", "thickness", "intensity", "hue"])
        tag = f"{attr} -> {direction}"
        row_blocks.append((tag, [(cf[i], attr_label(pred_cf, i, highlight=attr)) for i in range(args.n_images)]))

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
                      f"-- labels: d=digit t=thickness i=intensity h=hue (* = intervened attribute)",
              fill=(0, 0, 0), font=title_font)

    y0 = 32
    for row_idx, (tag, cells) in enumerate(row_blocks):
        y = y0 + row_idx * (CELL_W + LABEL_H)
        draw.text((8, y + CELL_W // 2), tag, fill=(80, 40, 160) if "->" in tag else (0, 0, 0), font=tag_font)
        for col_idx, (img_t, label) in enumerate(cells):
            x = ROW_TAG_W + col_idx * CELL_W
            draw_cell(canvas, draw, font, Image.fromarray(
                img_t.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()), label, x, y)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
