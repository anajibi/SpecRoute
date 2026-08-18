#!/usr/bin/env python
"""Big side-by-side grid: digit counterfactual only, K1 vs K11, many images."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from experiments.hdae.counterfactuals import hdae_adapter  # noqa: F401
from experiments.hdae.counterfactuals.cf_contract import load_adapter
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.attr_predictor import load_attr_predictor
from experiments.hdae.data.morphomnist import MorphoMNISTPacked

ATTRS = ["digit", "thickness", "intensity", "hue"]
FMT = {"digit": lambda v: f"{v:.0f}", "thickness": lambda v: f"{v:.2f}",
      "intensity": lambda v: f"{v:.0f}", "hue": lambda v: f"{v:.2f}"}
CELL_W = 130
LABEL_H = 32
ROW_TAG_W = 160

MODELS = [
    ("k=1", "experiments/hdae/configs/morpho_hier_k1_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k1_v3/checkpoints/last.ckpt"),
    ("k=11", "experiments/hdae/configs/morpho_hier_k11_v3.yaml",
     "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last.ckpt"),
]


def predict_all(predictors, img_m11, device, order):
    x = img_m11.to(device)
    return {name: predictors[name].predict_raw(x).numpy() for name in order}


def label(prefix, values, i):
    d = FMT["digit"](float(values["digit"][i]))
    return f"{prefix}d={d}"


def draw_cell(canvas, draw, font, img_pil, m_label, gt_label, x, y):
    img_resized = img_pil.resize((CELL_W, CELL_W), Image.NEAREST)
    canvas.paste(img_resized, (x, y))
    draw.text((x + 2, y + CELL_W + 1), m_label, fill=(20, 20, 20), font=font)
    draw.text((x + 2, y + CELL_W + 1 + 13), gt_label, fill=(90, 90, 90), font=font)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph_morpho.yaml")
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist_70k.h5")
    p.add_argument("--predictors-dir", default="experiments/hdae/outputs/attr_predictors_70k")
    p.add_argument("--cohorts", default="experiments/hdae/outputs/intervention_cohorts.json")
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--edit-strength", type=float, default=8.0)
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
    sel = cohorts["_meta"]["fixed_indices"][: args.n_images]

    imgs = torch.stack([ds[int(i)]["img"] for i in sel]).to(device)
    attrs_raw = torch.stack([torch.as_tensor(ds[int(i)]["attr"]) for i in sel]).to(device)
    scm_cols = [ds.attribute_names.index(a) for a in graph.attributes]
    scm_attr_index = {name: i for i, name in enumerate(graph.attributes)}
    n_classes = scm.specs["digit"].num_classes
    descendants = sorted(graph.descendants("digit"))
    observed = [a for a in graph.attributes if a != "digit" and a not in descendants]

    gt_digit_real = attrs_raw[:, scm_cols[scm_attr_index["digit"]]].detach().cpu().numpy()

    row_blocks = []
    orig_labels = [(f"orig d={int(gt_digit_real[i])}", "") for i in range(len(sel))]
    row_blocks.append(("original", [(imgs[i], orig_labels[i]) for i in range(len(sel))]))

    for model_label, config, ckpt in MODELS:
        print(f"running {model_label} ...")
        adapter = load_adapter("hdae", config, ckpt, device, edit_strength=args.edit_strength, T=args.T, compile_model=False)
        state = adapter.encode(imgs, attrs_raw, ds.attribute_names)

        cur_class = scm.categorical_class_index("digit", attrs_raw[:, scm_cols[scm_attr_index["digit"]]])
        target_class = (cur_class + n_classes // 2) % n_classes
        target_tensor = scm.class_index_to_raw("digit", target_class).view(-1, 1)
        cf_attrs = scm.counterfactual(attrs_raw[:, scm_cols].float(), scm_attr_index, interventions={"digit": target_tensor})
        for a in observed:
            cf_attrs[a] = attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()
        gt_target = target_class.detach().cpu().numpy()

        cf_state = adapter.intervene(state, "digit", "shift", cf_attrs)
        cf = adapter.render(cf_state)
        pred_cf = predict_all(predictors, cf * 2 - 1, device, ATTRS)

        cells = []
        for i in range(len(sel)):
            m_lbl = f"m: d={pred_cf['digit'][i]:.0f}"
            gt_lbl = f"gt: d={int(gt_target[i])}"
            cells.append((cf[i], (m_lbl, gt_lbl)))
        row_blocks.append((f"digit->shift [{model_label}]", cells))
        del adapter
        torch.cuda.empty_cache()

    n_rows = len(row_blocks)
    width = ROW_TAG_W + len(sel) * CELL_W
    height = n_rows * (CELL_W + LABEL_H) + 50
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    try:
        title_font = ImageFont.load_default(size=16)
        tag_font = ImageFont.load_default(size=13)
    except TypeError:
        title_font = tag_font = font
    draw.text((8, 4), f"MorphoMNIST HDAE digit counterfactual: k=1 vs k=11 -- guidance scale {args.edit_strength}",
              fill=(0, 0, 0), font=title_font)

    y0 = 28
    for row_idx, (tag, cells) in enumerate(row_blocks):
        y = y0 + row_idx * (CELL_W + LABEL_H)
        color = (0, 0, 0) if row_idx == 0 else (80, 40, 160)
        draw.text((8, y + CELL_W // 2), tag, fill=color, font=tag_font)
        for col_idx, (img_t, (m_lbl, gt_lbl)) in enumerate(cells):
            x = ROW_TAG_W + col_idx * CELL_W
            draw_cell(canvas, draw, font, Image.fromarray(
                img_t.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()), m_lbl, gt_lbl, x, y)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
