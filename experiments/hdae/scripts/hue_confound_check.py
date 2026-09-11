"""Is the intensity/thickness predictor reading hue?

Under do(hue) the FC term for thickness and intensity is pinned at exactly 0.0000 for every
non-degenerate config -- the measured error EXCEEDS the best-constant-predictor baseline. The
only config that escapes is `fusesum`, which fails to change hue at all. That pattern says the
readout is hue-sensitive rather than the generator being destructive, but the sweep cannot
distinguish the two: it only ever sees generated images.

This runs the predictors on REAL test images, where the ground truth is known and hue is
independent of thickness and intensity by construction. If the predictor is colour-blind its
error is flat across hue bins and its prediction is uncorrelated with hue. If it is not, the
do(hue) FC collapse is a measurement artifact and those columns do not mean what they say.
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from experiments.hdae.scripts.cfg_sweep_morpho import Reader  # noqa: E402

H5 = os.path.join(REPO, "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5")
N = 4096
dev = "cuda"

with h5py.File(H5, "r") as h:
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["attribute_names"][:]]
    A = h["attrs"][:].astype(np.float64)
    ntot = len(A)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(ntot, size=min(N, ntot), replace=False))
    X = torch.from_numpy(h["images"][idx].astype(np.float32) / 255.).permute(0, 3, 1, 2) * 2 - 1
COL = {k: names.index(k) for k in names}
hue_raw = A[idx, COL["hue"]]
hue_bin = np.round(hue_raw * 10 - 0.5).astype(int)

print(f"n={len(idx)} real test images, hue bins present: {sorted(set(hue_bin.tolist()))}\n")
for attr in ["intensity", "thickness"]:
    truth = A[idx, COL[attr]]
    r = Reader(attr, dev)
    pred = r(X).numpy().astype(np.float64)
    err = np.abs(pred - truth)
    # Correlation of the PREDICTION with hue, next to the truth's own correlation with hue.
    # Ground truth has no hue->attr edge, so the truth row is the null this is measured against.
    c_pred = np.corrcoef(pred, hue_raw)[0, 1]
    c_true = np.corrcoef(truth, hue_raw)[0, 1]
    print(f"[{attr}]  overall MAE {err.mean():.4f}   spread(truth) {truth.std():.4f}")
    print(f"  corr(hue, TRUE {attr})      {c_true:+.4f}   <- the null: independent by construction")
    print(f"  corr(hue, PREDICTED {attr}) {c_pred:+.4f}   <- non-zero means the readout sees colour")
    print(f"  {'bin':>4s} {'n':>5s} {'MAE':>9s} {'mean pred':>11s} {'mean true':>11s} {'bias':>9s}")
    for b in sorted(set(hue_bin.tolist())):
        m = hue_bin == b
        print(f"  {b:4d} {m.sum():5d} {err[m].mean():9.4f} {pred[m].mean():11.4f} "
              f"{truth[m].mean():11.4f} {(pred[m]-truth[m]).mean():+9.4f}")
    lo, hi = err[hue_bin == min(hue_bin)].mean(), err[hue_bin == max(hue_bin)].mean()
    spread = max(err[hue_bin == b].mean() for b in set(hue_bin.tolist())) - \
             min(err[hue_bin == b].mean() for b in set(hue_bin.tolist()))
    print(f"  MAE spread across hue bins: {spread:.4f}  ({spread/err.mean()*100:.1f}% of overall MAE)\n")
