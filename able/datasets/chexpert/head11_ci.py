"""Bootstrap CIs on radiologist_val for the 11-head measurement checkpoint.

The 11-head run reports point AUCs; the v2 gate needs the CI lower bound too.
All eleven are bootstrapped, not just the new head, because the other ten are
how you tell whether adding the eleventh loss term disturbed the trunk -- if
they move a lot, the new head's number is not comparable to best.ckpt's.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--data", required=True, help="the 11-label scratch data dir")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--frozen-report", required=True, help="v2 predictor_report.json, for comparison")
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.src)
import train_predictors as T                                   # noqa: E402
from train_head11 import BINARY11                              # noqa: E402

T.BINARY = BINARY11
from train_predictors import Chex, Net, predict                # noqa: E402

dev = "cuda"
ck = torch.load(args.ckpt, map_location="cpu")
assert ck["binary"] == BINARY11, f"checkpoint heads {ck['binary']} != {BINARY11}"
net = Net().to(dev)
net.load_state_dict(ck["state_dict"])
net.eval()

rmeta = pd.read_csv(f"{args.data}/radiologist_val_meta.csv")
p, a, y, ab = predict(net, DataLoader(
    Chex(f"{args.data}/radiologist_val_224.npy", rmeta, np.arange(len(rmeta)), False),
    batch_size=128, num_workers=2), dev)


def boot(yy, pp, n=2000, seed=7):
    yy, pp = np.asarray(yy), np.asarray(pp)
    if len(np.unique(yy)) < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[i])) > 1:
            v.append(roc_auc_score(yy[i], pp[i]))
    return (round(float(roc_auc_score(yy, pp)), 4),
            round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4))


frozen = json.load(open(args.frozen_report))["v2_gate"]
AUC_MIN, CI_LO_MIN = 0.85, 0.80
res = {}
print(f"{'label':30s} {'AUC':>7s} {'95% CI':>18s} {'pos':>4s}  {'best.ckpt':>9s}  {'delta':>7s} gate")
for j, k in enumerate(BINARY11):
    auc, lo, hi = boot(y[:, j], p[:, j])
    fz = frozen.get(k, {}).get("radiologist_val_auc")
    d = None if fz is None else round(auc - fz, 4)
    gp = bool(auc is not None and auc >= AUC_MIN and lo is not None and lo >= CI_LO_MIN)
    res[k] = {"radiologist_val_auc": auc, "ci95": [lo, hi],
              "n_pos": int(y[:, j].sum()), "best_ckpt_auc": fz, "delta_vs_best_ckpt": d,
              "v2_gate_pass": gp, "new_head": k == "enlarged_cardiomediastinum"}
    print(f"{k:30s} {auc:7.4f} [{lo:.4f},{hi:.4f}] {int(y[:,j].sum()):4d}  "
          f"{'n/a' if fz is None else f'{fz:9.4f}'}  {'n/a' if d is None else f'{d:+7.4f}'} "
          f"{'PASS' if gp else 'fail'}{'   <-- NEW HEAD' if k=='enlarged_cardiomediastinum' else ''}")

deltas = [v["delta_vs_best_ckpt"] for v in res.values() if v["delta_vs_best_ckpt"] is not None]
rec = {
    "checkpoint": args.ckpt, "selected_epoch": ck.get("epoch"),
    "select_mean_auc": ck.get("select_mean_auc"),
    "heads": BINARY11, "n_radiologist_val": int(len(rmeta)),
    "gate": "radiologist_val AUC >= 0.85 and 95% bootstrap CI lower bound >= 0.80",
    "labels": res,
    "trunk_disturbance": {
        "note": "the ten shared heads vs best.ckpt. A large shift would mean the eleventh "
                "loss term changed the trunk enough that the new head's number is not "
                "comparable to the frozen predictor's.",
        "mean_abs_delta": round(float(np.mean(np.abs(deltas))), 4),
        "max_abs_delta": round(float(np.max(np.abs(deltas))), 4),
        "per_label": {k: v["delta_vs_best_ckpt"] for k, v in res.items()
                      if v["delta_vs_best_ckpt"] is not None},
    },
}
os.makedirs(args.out, exist_ok=True)
json.dump(rec, open(f"{args.out}/head11_radval_ci.json", "w"), indent=2)
print(f"\ntrunk disturbance: mean |delta| {rec['trunk_disturbance']['mean_abs_delta']}  "
      f"max {rec['trunk_disturbance']['max_abs_delta']}")
print(f"wrote {args.out}/head11_radval_ci.json")
