#!/usr/bin/env python
"""Compare the trained CNN predictors (`train_morpho_attr_predictors.py`) against the closed-form
`measure_morphomnist.py` script, on `partition==1` -- the packed test split, never seen by any
CNN during training or checkpoint selection.

Two families of metric, matching the two families of predictor:

- Continuous attributes (thickness, intensity, hue, scale, translate_x/y, bg_freq/phase/amplitude,
  texture_amplitude): mean absolute error, in original units. Most have a real closed-form
  comparison point in measure_morphomnist.py; scale doesn't (no closed-form access to a
  multiplicative factor), so its baseline is a train-only linear calibration of the deterministic
  size proxy instead -- the baseline's best possible shot, not an equivalent estimator.
- Categorical attributes (rotation, slant, digit): accuracy and macro-F1 (sklearn) over the
  predictor's own bins, no regression fallback. None of the three has a closed-form baseline --
  rotation/slant because a single image can't disentangle them from each other or from the
  digit's own shape, digit because recognizing it needs a classifier by definition (same reason
  measure_morphomnist.py has always excluded it).

Writes a JSON results file for the publish step; does not itself render anything.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score

from experiments.hdae.data.attr_predictor import AttrSpec, load_attr_predictor
from experiments.hdae.data.measure_morphomnist import measure_all, measure_scale_proxy_from_image
from experiments.hdae.data.morphomnist import MorphoMNISTPacked
from experiments.hdae.data.train_morpho_attr_predictors import CATEGORICAL_ATTRS, TARGET_ATTRS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# attribute -> deterministic measurement key in measure_all()'s output (None = no deterministic estimate)
DETERMINISTIC_KEY = {
    "thickness": "thickness", "intensity": "intensity", "hue": "hue", "bg_freq": "bg_freq",
    "translate_x": "translate_x", "translate_y": "translate_y",
    "bg_amplitude": "bg_amplitude", "texture_amplitude": "texture_amplitude",
    "bg_phase": "bg_phase",
    "rotation": None, "slant": None, "digit": None,  # categorical, no closed-form estimator at all
    "scale": None,  # calibrated separately below (fitted linear proxy, not in measure_all)
}
CIRCULAR = {"hue": 1.0, "bg_phase": 2 * np.pi}
# Categorical attrs whose classes have a real ordering (bin 5 is "close to" bin 6) -- exact-bin
# accuracy alone understates these badly (a 1-bin miss scores identically to a 19-bin miss), so
# also report MAE in original units via the bin-center-decoded prediction. Digit is NOT ordinal
# in the relevant sense (misclassifying 6 as 8 isn't "closer" than misclassifying it as 0) and is
# excluded.
ORDINAL_CATEGORICAL = {"rotation", "slant", "translate_x", "translate_y"}


def to_rgb_u8(img_tensor: torch.Tensor) -> np.ndarray:
    arr = ((img_tensor.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return arr


def circular_error(pred, true, period):
    diff = np.abs(pred - true) % period
    return np.minimum(diff, period - diff)


def fit_scale_calibration(ds: MorphoMNISTPacked, train_indices: np.ndarray, n_fit: int = 3000, seed: int = 0):
    """Least-squares linear fit: scale ~= a * size_proxy + b, on TRAIN images only."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(train_indices, size=min(n_fit, len(train_indices)), replace=False)
    col_scale = ds.attribute_names.index("scale")
    proxies, trues = [], []
    for i in idx:
        item = ds[int(i)]
        rgb = to_rgb_u8(item["img"])
        proxies.append(measure_scale_proxy_from_image(rgb))
        trues.append(float(item["attr"][col_scale]))
    proxies, trues = np.array(proxies), np.array(trues)
    a, b = np.polyfit(proxies, trues, deg=1)
    resid = trues - (a * proxies + b)
    logging.info("scale calibration (train, n=%d): scale ~= %.5f*proxy + %.4f, residual std=%.4f",
                len(idx), a, b, resid.std())
    return float(a), float(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packed", default="experiments/hdae/data/packed/morphomnist.h5")
    p.add_argument("--predictors-dir", default="experiments/hdae/outputs/attr_predictors")
    p.add_argument("--n-test", type=int, default=2000, help="subset of the test split to evaluate "
                   "(measure_morphomnist.py's curve_fit-based background estimate is slow per-image)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="experiments/hdae/outputs/attr_predictors/comparison_results.json")
    args = p.parse_args()

    ds = MorphoMNISTPacked(args.packed)
    train_indices = np.nonzero(ds.partitions == 0)[0]
    test_indices_all = np.nonzero(ds.partitions == 1)[0]
    rng = np.random.RandomState(args.seed)
    test_indices = rng.choice(test_indices_all, size=min(args.n_test, len(test_indices_all)), replace=False)
    logging.info("evaluating on %d/%d test images (partition==1, untouched by any CNN training)",
                len(test_indices), len(test_indices_all))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = json.loads((Path(args.predictors_dir) / "training_summary.json").read_text())
    models = {name: load_attr_predictor(info["checkpoint"], attr_col=info["attr_col"]).to(device)
             for name, info in summary.items()}
    specs = {name: AttrSpec(name=name, kind=info["kind"], lo=info.get("lo"), hi=info.get("hi"),
                            period=info.get("period"), num_bins=info.get("num_bins"))
            for name, info in summary.items()}

    scale_a, scale_b = fit_scale_calibration(ds, train_indices, seed=args.seed)

    cnn_pred = {name: [] for name in TARGET_ATTRS}
    det_pred = {name: [] for name in TARGET_ATTRS}
    gt = {name: [] for name in TARGET_ATTRS}

    imgs_batch = []
    BATCH = 128

    def flush_cnn():
        if not imgs_batch:
            return
        batch = torch.stack(imgs_batch).to(device)
        for name in TARGET_ATTRS:
            preds = models[name].predict_raw(batch).numpy()
            cnn_pred[name].extend(preds.tolist())
        imgs_batch.clear()

    for count, i in enumerate(test_indices):
        item = ds[int(i)]
        for name in TARGET_ATTRS:
            gt[name].append(float(item["attr"][ds.attribute_names.index(name)]))
        imgs_batch.append(item["img"])
        if len(imgs_batch) >= BATCH:
            flush_cnn()
        rgb = to_rgb_u8(item["img"])
        meas = measure_all(rgb)
        for name in TARGET_ATTRS:
            key = DETERMINISTIC_KEY[name]
            det_pred[name].append(meas[key] if key is not None else float("nan"))
        det_pred["scale"][-1] = scale_a * meas["scale_proxy"] + scale_b
        if (count + 1) % 500 == 0:
            logging.info("  measured %d/%d", count + 1, len(test_indices))
    flush_cnn()

    results = {}
    for name in TARGET_ATTRS:
        g = np.array(gt[name])
        c = np.array(cnn_pred[name])
        d = np.array(det_pred[name])

        if name in CATEGORICAL_ATTRS:
            spec = specs[name]
            true_class = spec.normalize(torch.from_numpy(g).float()).numpy()
            pred_class = spec.normalize(torch.from_numpy(c).float()).numpy()  # c is already bin-center-decoded
            acc = float((pred_class == true_class).mean())
            macro_f1 = float(f1_score(true_class, pred_class, average="macro", zero_division=0))
            entry = {"tier": "classification", "accuracy": acc, "macro_f1": macro_f1,
                    "num_classes": spec.num_bins, "n": len(g), "ordinal": name in ORDINAL_CATEGORICAL,
                    "deterministic_accuracy": None, "deterministic_macro_f1": None,
                    "cnn_mae": None, "deterministic_mae": None, "within_1_bin": None, "within_2_bin": None}
            if name in ORDINAL_CATEGORICAL:
                bin_off = np.abs(pred_class - true_class)
                entry["cnn_mae"] = float(np.abs(c - g).mean())
                entry["within_1_bin"] = float((bin_off <= 1).mean())
                entry["within_2_bin"] = float((bin_off <= 2).mean())
            key = DETERMINISTIC_KEY[name]
            if key is not None:
                # A real closed-form estimator exists for this attribute (e.g. translate_x/y's
                # centroid) -- bin its continuous output into the same classes so the comparison
                # survives the switch to classification instead of silently disappearing.
                det_class = spec.normalize(torch.from_numpy(d).float()).numpy()
                entry["deterministic_accuracy"] = float((det_class == true_class).mean())
                entry["deterministic_macro_f1"] = float(f1_score(true_class, det_class, average="macro",
                                                                  zero_division=0))
                if name in ORDINAL_CATEGORICAL:
                    entry["deterministic_mae"] = float(np.abs(d - g).mean())
            results[name] = entry
            logging.info("%-20s accuracy=%.4f macro_f1=%.4f (%d classes) cnn_mae=%s deterministic_accuracy=%s",
                        name, acc, macro_f1, spec.num_bins,
                        f"{entry['cnn_mae']:.4f}" if entry["cnn_mae"] is not None else "n/a",
                        f"{entry['deterministic_accuracy']:.4f}" if key is not None else "n/a")
            continue

        if name in CIRCULAR:
            cnn_err = circular_error(c, g, CIRCULAR[name])
            det_err = circular_error(d, g, CIRCULAR[name]) if not np.isnan(d).all() else None
        else:
            cnn_err = np.abs(c - g)
            det_err = np.abs(d - g) if not np.isnan(d).all() else None
        results[name] = {
            "tier": "calibrated_baseline" if name == "scale" else "direct",
            "cnn_mae": float(cnn_err.mean()), "cnn_mae_std": float(cnn_err.std()),
            "deterministic_mae": float(det_err.mean()) if det_err is not None else None,
            "deterministic_mae_std": float(det_err.std()) if det_err is not None else None,
            "n": len(g),
        }
        logging.info("%-20s cnn_mae=%.4f  deterministic_mae=%s", name, results[name]["cnn_mae"],
                    f"{results[name]['deterministic_mae']:.4f}" if det_err is not None else "n/a")

    output = {"n_test": len(test_indices), "scale_calibration": {"a": scale_a, "b": scale_b},
             "per_attribute": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    logging.info("wrote comparison results to %s", args.output)


if __name__ == "__main__":
    main()
