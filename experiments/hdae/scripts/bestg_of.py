"""Print `digit=.. thickness=.. intensity=.. hue=..` : the per-intervention argmax of CC
for one sweep, in the form cfg_sweep_morpho.py's --attr-g expects."""
import json, os, sys
import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SW = os.path.join(REPO, "experiments/hdae/outputs/morpho_sweep")
MOD = ["digit", "thickness", "intensity", "hue"]
CAT = {"digit", "hue"}
with h5py.File(os.path.join(REPO, "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5"), "r") as h:
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["attribute_names"][:]]
    A = h["attrs"][:].astype(np.float64)
COL = {k: names.index(k) for k in names}
js = json.load(open(os.path.join(SW, f"sweep_{sys.argv[1]}.json")))
r, S = js["results"], js["strengths"]
out = []
for m in MOD:
    f0 = r["recon"][m]["value"]
    src = A[js["cohort_idx"], COL[m]]; tg = np.asarray(js["targets"][m], float)
    cv = (lambda v: np.round(v)) if m == "digit" else (lambda v: np.round(v * 10 - 0.5))
    mv = float((cv(src) != cv(tg)).mean()) if m in CAT else float(np.abs(src - tg).mean())
    cc = [np.clip(1 - max(0, r[f"do({m})|g{g:g}"][m]["value"] - f0) / max(mv - f0, 1e-9), 0, 1) for g in S]
    out.append(f"{m}={S[int(np.argmax(cc))]:g}")
print(" ".join(out))
