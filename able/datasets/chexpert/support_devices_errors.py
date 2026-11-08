"""Error breakdown for support_devices on radiologist_val, from the FROZEN predictor.

The question is whether this project's classifier misses devices that are
there (false negatives) or invents devices that are not (false positives). The
two failure modes mean different things: FN-heavy says the head is
under-sensitive, which the report-label training would explain -- the report
only mentions a device when it is clinically relevant, so "device present,
never mentioned" is a huge silent-negative class. FP-heavy would instead say
the head is firing on something else in the image.

Thresholds: Youden's J on the SELECT set (inside the train split, report
labels), which is the protocol's rule, plus 0.5 for reference. best.ckpt is
read, never written.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="dir holding train_predictors.py")
ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=20261107)
ap.add_argument("--select-frac", type=float, default=0.10)
args = ap.parse_args()

sys.path.insert(0, args.src)
from train_predictors import BINARY, Chex, Net, predict      # noqa: E402

dev = "cuda"
J = BINARY.index("support_devices")
meta = pd.read_csv(f"{args.data}/chexpert_meta.csv")
split = np.load(f"{args.data}/split.npy")
train_rows = np.flatnonzero(split == 0)
tp = np.sort(meta.loc[train_rows, "patient_id"].unique())
perm = np.random.default_rng(args.seed + 1).permutation(tp)
sel_pat = set(perm[:int(round(len(tp) * args.select_frac))].tolist())
sel_rows = train_rows[meta["patient_id"].isin(sel_pat).to_numpy()[train_rows]]

ck = torch.load(args.ckpt, map_location="cpu")
assert ck["binary"] == BINARY, "this is not the 10-head frozen checkpoint"
net = Net().to(dev)
net.load_state_dict(ck["state_dict"])
net.eval()

p_sel, _, y_sel, _ = predict(net, DataLoader(
    Chex(f"{args.data}/images_224.npy", meta, sel_rows, False),
    batch_size=256, num_workers=8, pin_memory=True), dev)
rmeta = pd.read_csv(f"{args.data}/radiologist_val_meta.csv")
p_rad, _, _, _ = predict(net, DataLoader(
    Chex(f"{args.data}/radiologist_val_224.npy", rmeta, np.arange(len(rmeta)), False),
    batch_size=128, num_workers=2), dev)

# radiologist truth, straight from valid.csv
valid = pd.read_csv(f"{args.data}/source_csv/valid.csv").set_index("Path")
y = pd.to_numeric(valid.loc[rmeta["path"].values, "Support Devices"],
                  errors="coerce").to_numpy().astype(np.int64)
p = p_rad[:, J]

fpr, tpr, t = roc_curve(y_sel[:, J], p_sel[:, J])
youden = float(t[int(np.argmax(tpr - fpr))])

res = {"label": "support_devices", "set": "radiologist_val (200 frontal)",
       "n": int(len(y)), "n_pos_radiologist": int(y.sum()), "n_neg_radiologist": int((y == 0).sum()),
       "auc_vs_radiologist": round(float(roc_auc_score(y, p)), 4),
       "report_label_in_frozen_csv_agreement": None, "thresholds": {}}

# how often the frozen report-derived column disagrees with the radiologist here
rep_col = rmeta["support_devices"].to_numpy()
res["report_label_in_frozen_csv_agreement"] = {
    "n_report_pos": int(rep_col.sum()),
    "report_vs_radiologist_disagree": int((rep_col != y).sum()),
    "report_says_absent_radiologist_says_present": int(((rep_col == 0) & (y == 1)).sum()),
    "report_says_present_radiologist_says_absent": int(((rep_col == 1) & (y == 0)).sum()),
}

for name, th in (("youden_on_select", youden), ("half", 0.5)):
    pred = (p >= th).astype(np.int64)
    tp_ = int(((pred == 1) & (y == 1)).sum())
    fp_ = int(((pred == 1) & (y == 0)).sum())
    fn_ = int(((pred == 0) & (y == 1)).sum())
    tn_ = int(((pred == 0) & (y == 0)).sum())
    err = fp_ + fn_
    res["thresholds"][name] = {
        "threshold": round(th, 4), "tp": tp_, "fp": fp_, "fn": fn_, "tn": tn_,
        "errors": err,
        "false_negatives_device_present_predicted_absent": fn_,
        "fn_share_of_errors": None if err == 0 else round(fn_ / err, 4),
        "tpr": round(tp_ / max(1, tp_ + fn_), 4), "fpr": round(fp_ / max(1, fp_ + tn_), 4),
        "accuracy": round((tp_ + tn_) / len(y), 4),
    }

os.makedirs(args.out, exist_ok=True)
json.dump(res, open(f"{args.out}/support_devices_errors.json", "w"), indent=2)
print(json.dumps(res, indent=2))
print(f"\nwrote {args.out}/support_devices_errors.json")
