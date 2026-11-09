"""Score multi-intervention pixels: did each do(X) land, and what did it disturb?

Reports per intervention and guidance:
  target hit / real flips   did the INTERVENED attribute reach its post-intervention value,
                            and did it get there by moving off the reconstruction's reading
                            (a real flip) rather than by the reader already misreading the
                            reconstruction as the target (a freebie)?
  direction                 for binaries, 0->1 and 1->0 separately: on an imbalanced label a
                            diffusion prior makes one direction nearly free.
  descendants               for do(age_bin) only, whether effusion and edema followed the SCM.
  preserved                 every attribute the intervention should NOT have moved.

sex is the control. The frozen predictor reads it at AUC .9945 off real images, so if the
conditioning pathway works at all, do(sex) is where it shows.
"""
import argparse, json, os, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from able.datasets.chexpert.cf_score_pixels import score_ours     # noqa: E402
from able.datasets.chexpert.cf_images import PRED_BINARY, COND    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--px", required=True)
ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
ap.add_argument("--ckpt", default="/home/exouser/able_data/chexpert/predictors/best.ckpt")
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
man = json.load(open(f"{a.px}/manifest.json"))
pr = json.load(open(f"{a.data}/predictors/predictor_report.json"))
t = pr.get("thresholds_youden_on_select_v2", pr["thresholds_youden_on_select"])
th = {k: (float(t[k]) if not isinstance(t[k], dict) else float(t[k]["threshold"]))
      for k in ("sex", "effusion", "edema")}
fac = {k: np.array(man["factual"][k]) for k in COND}
n = len(fac["sex"])

def read(key):
    pb, pa = score_ours(np.load(f"{a.px}/{key}.npy"), a.ckpt, dev)
    out = {k: (pb[:, PRED_BINARY.index(k)] >= th[k]).astype(int) for k in ("sex", "effusion", "edema")}
    out["age_bin"] = pa.argmax(1)
    return out

rec = read("recon")
print(f"\nstep {man['global_step']}   n={n}")
print("reconstruction read back == factual:  " +
      "   ".join(f"{k} {int((rec[k]==fac[k]).sum())}/{n}" for k in COND) + "\n")
hdr = (f"{'do(X)':10s} {'g':>3s} | {'hit':>6s} {'real':>5s} {'free':>5s} {'wrong':>6s} | "
       f"{'0->1':>7s} {'1->0':>7s} | preserved")
print(hdr); print("-" * (len(hdr) + 14))
rep = {"global_step": man["global_step"], "n": n, "recon_vs_factual":
       {k: int((rec[k] == fac[k]).sum()) for k in COND}, "interventions": {}}
for v in man["vars"]:
    tgt = {k: np.array(man["targets"][v][k]) for k in COND}
    for g in man["guidances"]:
        key = f"{v}_g{g:g}"
        got = read(key)
        ch = got[v] != rec[v]
        hit = int((got[v] == tgt[v]).sum())
        real = int((ch & (got[v] == tgt[v])).sum())
        free = int(((~ch) & (got[v] == tgt[v])).sum())
        wrong = int((ch & (got[v] != tgt[v])).sum())
        if v in ("sex", "effusion", "edema"):
            up = fac[v] == 0; dn = fac[v] == 1
            d01 = f"{int((up & ch & (got[v]==1)).sum())}/{int(up.sum())}"
            d10 = f"{int((dn & ch & (got[v]==0)).sum())}/{int(dn.sum())}"
        else:
            d01 = d10 = "n/a"
        keep = [k for k in COND if k != v and (tgt[k] == fac[k]).all()]
        pres = "  ".join(f"{k} {int((got[k]==fac[k]).sum())}/{n}" for k in keep)
        desc = {k: int((got[k] == tgt[k]).sum()) for k in COND
                if k != v and (tgt[k] != fac[k]).any()}
        rep["interventions"][key] = {"hit": hit, "real": real, "free": free, "wrong": wrong,
                                     "up_0to1": d01, "down_1to0": d10,
                                     "preserved": {k: int((got[k]==fac[k]).sum()) for k in keep},
                                     "descendants_match_target": desc}
        print(f"{('do('+v+')'):10s} {g:3g} | {hit:3d}/{n:<2d} {real:5d} {free:5d} {wrong:6d} | "
              f"{d01:>7s} {d10:>7s} | {pres}" + (f"   desc {desc}" if desc else ""))
if a.out:
    json.dump(rep, open(a.out, "w"), indent=2); print(f"\nwrote {a.out}")
