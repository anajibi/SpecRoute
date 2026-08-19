#!/usr/bin/env python
"""T8: predictor hue-sensitivity. FC_observed for a hue intervention is measured by CNN
predictors that must read digit/thickness/intensity off a RECOLORED image. If the non-hue heads
are hue-sensitive, part of F2 (and every CC/FC number that touches these predictors) is
measurement, not model behavior.

Method: bucket real held-out images by hue class, compute digit top-1 accuracy and thickness/
intensity MAE per bucket, report max/min ratio and whether the predictors were trained with any
color augmentation.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.hdae.causal.scm import SCM
from experiments.hdae.data.attr_predictor import load_attr_predictor
from experiments.hdae.data.morphomnist import MorphoMNISTPacked
import yaml

PACKED = "experiments/hdae/data/packed/morphomnist_70k.h5"
PREDICTORS_DIR = "experiments/hdae/outputs/attr_predictors_70k"
CAUSAL_GRAPH = "experiments/hdae/configs/causal_graph_morpho.yaml"
OUT_PATH = "experiments/hdae/outputs/diagnostics_t8_hue_sensitivity.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MorphoMNISTPacked(PACKED)
    names = ds.attribute_names
    summary = json.loads((Path(PREDICTORS_DIR) / "training_summary.json").read_text())
    predictors = {a: load_attr_predictor(summary[a]["checkpoint"], attr_col=names.index(a)).to(device)
                 for a in ["digit", "thickness", "intensity"]}
    with open(CAUSAL_GRAPH) as f:
        causal_raw = yaml.safe_load(f)
    scm = SCM.load(causal_raw["scm_checkpoint"], device=device)

    val_mask = ds.partitions == 1
    val_idx = np.where(val_mask)[0]
    hue_raw = ds.attrs[val_idx, names.index("hue")]
    hue_class = scm.categorical_class_index("hue", torch.from_numpy(hue_raw).float()).cpu().numpy()

    def predict_batch(attr, indices, batch=256):
        preds = []
        for start in range(0, len(indices), batch):
            chunk = indices[start:start + batch]
            imgs = torch.stack([ds[int(i)]["img"] for i in chunk]).to(device)
            with torch.no_grad():
                p = predictors[attr].predict_raw(imgs).cpu().numpy()
            preds.append(p)
        return np.concatenate(preds)

    digit_true = ds.attrs[val_idx, names.index("digit")].astype(int)
    thickness_true = ds.attrs[val_idx, names.index("thickness")]
    intensity_true = ds.attrs[val_idx, names.index("intensity")]

    digit_pred = predict_batch("digit", val_idx).round().astype(int).clip(0, 9)
    thickness_pred = predict_batch("thickness", val_idx)
    intensity_pred = predict_batch("intensity", val_idx)

    per_bucket = {}
    for h in range(10):
        m = hue_class == h
        n = int(m.sum())
        digit_acc = float((digit_pred[m] == digit_true[m]).mean()) if n else None
        thickness_mae = float(np.abs(thickness_pred[m] - thickness_true[m]).mean()) if n else None
        intensity_mae = float(np.abs(intensity_pred[m] - intensity_true[m]).mean()) if n else None
        per_bucket[str(h)] = {"n": n, "digit_acc": digit_acc, "thickness_mae": thickness_mae, "intensity_mae": intensity_mae}
        print(f"hue_class={h} n={n} digit_acc={digit_acc:.4f} thickness_mae={thickness_mae:.4f} intensity_mae={intensity_mae:.4f}")

    digit_accs = [v["digit_acc"] for v in per_bucket.values() if v["n"] > 0]
    thickness_maes = [v["thickness_mae"] for v in per_bucket.values() if v["n"] > 0]
    intensity_maes = [v["intensity_mae"] for v in per_bucket.values() if v["n"] > 0]

    summary_out = {
        "per_bucket": per_bucket,
        "digit_acc_range": [min(digit_accs), max(digit_accs)],
        "digit_acc_drop_points": float((max(digit_accs) - min(digit_accs)) * 100),
        "thickness_mae_ratio_max_min": float(max(thickness_maes) / min(thickness_maes)),
        "intensity_mae_ratio_max_min": float(max(intensity_maes) / min(intensity_maes)),
        "color_augmentation_used": "not checked in this pass -- see training script for attr predictors if needed",
    }
    print()
    print(json.dumps(summary_out, indent=2, default=str))
    Path(OUT_PATH).write_text(json.dumps(summary_out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
