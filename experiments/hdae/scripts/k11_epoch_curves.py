"""CC / FC / CF1 versus training epoch for k=11, plus the stitched loss curve.

Answers one question: past epoch 50, does more training still buy anything, or is k=11
saturated? The k=1 answer was emphatic (it degrades), so the same measurement on k=11 is what
decides whether the extension is worth finishing.

Metric definitions are the ones settled on in the design review:
  CC = 1 - (E|pred-target| - floor) / (E|source-target| - floor)      per target + descendants
  FC = 1 - max(0, E|pred-source| - floor) / (best-constant-error - floor)
  CF1 = harmonic mean, on values clipped to [0,1]
Raw accuracy and MAE are carried alongside every derived number.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRATCH = "/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/hdae/causal"))

from experiments.hdae.data.causal3dident import CLASS_NAMES, Causal3DIdentPacked  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from train_scm_causal3dident import SCM, CausalGraph  # noqa: E402

MOD = ["class", "pos_spl", "pos_obj", "rot_obj"]
UNOBS = ["hue_obj", "hue_spl", "hue_bg"]
ALL7 = MOD + UNOBS
COLS = {"class": [0], "pos_spl": [1], "pos_obj": [2, 3, 4], "rot_obj": [5, 6, 7],
        "hue_obj": [8], "hue_spl": [9], "hue_bg": [10]}
EDGES = [("class", "rot_obj"), ("class", "pos_obj"), ("pos_spl", "pos_obj")]
EPOCHS = [12, 25, 37, 50, 53, 59]
G = {"class": 8, "pos_spl": 1.5, "pos_obj": 2.5, "rot_obj": 3}
SWEEP = os.path.join(REPO, "experiments/hdae/outputs/cfg_sweep")


def desc(a):
    out, fr = set(), [a]
    while fr:
        n = fr.pop()
        for p, c in EDGES:
            if p == n and c not in out:
                out.add(c); fr.append(c)
    return out


def dist(attr, u, v):
    return (u.reshape(-1) != v.reshape(-1)).astype(float) if attr == "class" else np.abs(u - v).mean(1)


def main():
    ds = Causal3DIdentPacked(os.path.join(
        REPO, "experiments/hdae/data/causal3dident/causal3dident_testset_128.h5"))
    A_all = np.asarray(ds.attr, dtype=np.float64)
    base = {}
    for a in ALL7:
        v = A_all[:, COLS[a]]
        if a == "class":
            p = np.bincount(v.reshape(-1).astype(int), minlength=7) / len(v)
            base[a] = float(1.0 - p.max())
        else:
            base[a] = float(np.abs(v - v.mean(0)).mean())

    b = torch.load(os.path.join(REPO, "experiments/hdae/outputs/scm/causal3dident_scm_spline.pt"),
                   map_location="cpu")
    c = b["config"]
    scm = SCM(CausalGraph(c["attributes"], c["edges"]), c["nodes"], mechanism=b["mechanism"], bins=b["bins"])
    scm.load_state_dict(b["state_dict"]); scm.eval()
    specs = load_hdae_config(os.path.join(REPO, "experiments/hdae/configs/c3di_hier_k11_final.yaml"),
                             require_data=False).hdae_conf.encoder.cond_specs

    rows = []
    for ep in EPOCHS:
        jp = f"{SWEEP}/sweep_k11ep{ep}.json"; npz = f"{SWEEP}/persample_k11ep{ep}.npz"
        if not (os.path.exists(jp) and os.path.exists(npz)):
            print(f"epoch {ep}: missing, skipped"); continue
        js = json.load(open(jp)); ps = np.load(npz)
        idx = [r["i"] for r in js["cohort"]]
        A = np.asarray(ds.attr[idx], dtype=np.float64)
        y = to_cond_values(torch.from_numpy(A[:, :8]), specs).numpy()
        tgt = {"class": np.array([[CLASS_NAMES.index(r["class_tgt"])] for r in js["cohort"]], float)}
        for a in ["pos_spl", "pos_obj", "rot_obj"]:
            tgt[a] = np.array([r[a]["target"] for r in js["cohort"]], float)
        obs = {k: torch.from_numpy(y[:, COLS[k]]).float().contiguous() for k in MOD}
        floor = {a: (1 - js["results"]["ema|recon"][a]["value"]) if a == "class"
                 else js["results"]["ema|recon"][a]["value"] for a in ALL7}

        per = {}
        for a in MOD:
            with torch.no_grad():
                cfa = {k: v.numpy().astype(np.float64) for k, v in
                       scm.propagate(scm.abduct(obs), obs, {a: torch.from_numpy(tgt[a]).float()}).items()}
            d = desc(a); cc, fc, raw = {}, {}, {}
            for k in ALL7:
                rv = float(ps[f"ema|do({a})|g{G[a]:g}|{k}"].astype(np.float64).mean())
                raw[k] = rv
                err = (1 - rv) if k == "class" else rv
                fl = floor[k]
                if k == a or k in d:
                    ref = tgt[k] if k == a else cfa[k]
                    mv = float(dist(k, A[:, COLS[k]], ref).mean())
                    cc[k] = 1 - max(0.0, err - fl) / (mv - fl)
                else:
                    fc[k] = 1 - max(0.0, err - fl) / (base[k] - fl)
            cl = lambda d_: [min(1.0, max(0.0, x)) for x in d_.values()]
            CC = float(np.mean(cl(cc)))
            FO = float(np.mean([min(1, max(0, v)) for k, v in fc.items() if k in MOD]))
            FU = float(np.mean([min(1, max(0, v)) for k, v in fc.items() if k in UNOBS]))
            hm = lambda x, z: 0.0 if x + z == 0 else 2 * x * z / (x + z)
            per[a] = dict(CC=CC, FC_obs=FO, FC_unobs=FU, CF1_obs=hm(CC, FO), CF1_unobs=hm(CC, FU),
                          raw_target=raw[a], cc_target=cc[a])
        rows.append(dict(epoch=ep,
                         CC=float(np.mean([per[a]["CC"] for a in MOD])),
                         FC_obs=float(np.mean([per[a]["FC_obs"] for a in MOD])),
                         FC_unobs=float(np.mean([per[a]["FC_unobs"] for a in MOD])),
                         CF1_obs=float(np.mean([per[a]["CF1_obs"] for a in MOD])),
                         CF1_unobs=float(np.mean([per[a]["CF1_unobs"] for a in MOD])),
                         per=per))

    print(f"{'epoch':>6s} {'CC':>8s} {'FC_obs':>8s} {'FC_un':>8s} {'CF1_obs':>8s} {'CF1_un':>8s}   "
          + "  ".join(f"{a[:7]:>8s}" for a in MOD))
    for r in rows:
        print(f"{r['epoch']:6d} {r['CC']:8.4f} {r['FC_obs']:8.4f} {r['FC_unobs']:8.4f} "
              f"{r['CF1_obs']:8.4f} {r['CF1_unobs']:8.4f}   "
              + "  ".join(f"{r['per'][a]['raw_target']:8.4f}" for a in MOD))
    json.dump({"guidance": G, "rows": rows},
              open(f"{REPO}/experiments/hdae/outputs/k11_epoch_curves.json", "w"), indent=2)
    print(f"\nwrote {REPO}/experiments/hdae/outputs/k11_epoch_curves.json")


if __name__ == "__main__":
    main()
