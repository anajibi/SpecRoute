"""Render do(X) for SEVERAL interventions from one checkpoint, sharing the encode.

The point of this script is the control that was missing. Every counterfactual number so far
has been do(edema) -- synthesising or erasing a subtle diffuse pathology, the hardest of the
four modelled attributes. If the conditioning pathway were simply undertrained, do(sex) would
fail too: sex is read by the frozen predictor at AUC .9945 and has gross, global image
correlates. So running sex, effusion and age_bin beside edema separates two very different
conclusions:

    sex flips, edema does not   ->  the pathway works; subtle pathology is the hard part
    nothing flips               ->  the pathway is not learning, and the setup is wrong

DDIM-encoding a source costs as much as rendering from it, and x_T does not depend on which
attribute is intervened on. So each source is encoded ONCE and every (variable, guidance) pair
renders from that same x_T and the same latents -- making the comparison across interventions
exact rather than merely paired.

do(age_bin) is the one that legitimately propagates: age is a parent of both findings, so the
SCM moves effusion and edema downstream through the fitted mechanisms. The others are sinks.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as Fn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
import able.datasets.chexpert.scm_exact as S                      # noqa: E402
from able.datasets.chexpert.cf_images import COND, render, y_of   # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config      # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="experiments/hdae/configs/chexpert256_k1.yaml")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
ap.add_argument("--out", required=True)
ap.add_argument("--vars", default="sex,effusion,age_bin,edema")
ap.add_argument("--guidances", default="3,5")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--T", type=int, default=100)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
VARS = a.vars.split(","); GS = [float(x) for x in a.guidances.split(",")]
dev = "cuda"; os.makedirs(a.out, exist_ok=True)

cfg = load_hdae_config(a.config, require_data=False)
lit = HDAELitModule(cfg.train_conf)
blob = torch.load(a.ckpt, map_location="cpu")
sd = blob.get("state_dict", blob)
ema = {k[len("ema_model."):]: v for k, v in sd.items() if k.startswith("ema_model.")}
lit.ema_model.load_state_dict(ema, strict=True)
lit.ema_model = lit.ema_model.to(dev).eval(); lit.eval()
step = int(blob.get("global_step", -1)); del blob, sd, ema
specs = lit.model.hdae_conf.encoder.cond_specs
sampler = cfg.train_conf._make_diffusion_conf(a.T).make_sampler()
print(f"EMA step {step}; vars {VARS}; g {GS}", flush=True)

import pandas as pd
meta = pd.read_csv(f"{a.data}/chexpert_meta.csv")
split = np.load(f"{a.data}/split.npy")
unc = pd.read_csv(f"{a.data}/label_uncertainty.csv")
pool = np.flatnonzero((split == 1) & (unc["uncertain_edema"].to_numpy() == 0))
rows = np.sort(np.random.default_rng(a.seed).choice(pool, a.n, replace=False))   # SAME 16
arr = np.load(f"{a.data}/images_224.npy", mmap_mode="r")
vals = {k: meta.loc[rows, k].to_numpy(np.int64) for k in COND}
params = S.load(f"{a.data}/scm_params.json")

rng = np.random.default_rng(a.seed)
cfs = {}
for v in VARS:
    K = S.CARD[v]
    new = (1 - vals[v]) if K == 2 else (vals[v] + 1 + rng.integers(0, K - 1, len(rows))) % K
    c, _ = S.counterfactual(vals, params, v, new, seed=a.seed)
    cfs[v] = {k: np.asarray(c[k]) for k in COND}
    moved = [k for k in COND if (cfs[v][k] != vals[k]).any()]
    print(f"  do({v}): moves {moved}", flush=True)

keys = ["source", "recon"] + [f"{v}_g{g:g}" for v in VARS for g in GS]
store = {k: np.zeros((len(rows), 256, 256), np.uint8) for k in keys}
to8 = lambda t: (t[0].mean(0).cpu().numpy() * 255).astype(np.uint8)
for i in range(len(rows)):
    x = torch.from_numpy(np.asarray(arr[rows[i]], dtype=np.float32) / 127.5 - 1.0)[None, None]
    x = Fn.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False).clamp_(-1, 1)
    x = x.expand(-1, 3, -1, -1).contiguous().to(dev)
    with torch.no_grad():
        zs = [z.clone() for z in lit.ema_model.encode(x)]
        cond = lit.ema_model.make_cond(zs, y_of({k: vals[k][i:i+1] for k in COND}, specs, dev))
        x_T = sampler.ddim_reverse_sample_loop(lit.ema_model, x,
                                               model_kwargs={"cond": cond})["sample"]
        store["source"][i] = to8(((x + 1) / 2).clamp(0, 1))
        store["recon"][i] = to8(render(lit, sampler, x_T, cond, 1.0))
        for v in VARS:
            cc = lit.ema_model.make_cond(zs, y_of({k: cfs[v][k][i:i+1] for k in COND}, specs, dev))
            for g in GS:
                store[f"{v}_g{g:g}"][i] = to8(render(lit, sampler, x_T, cc, g))
    print(f"  {i+1}/{len(rows)}", flush=True)

for k in keys:
    np.save(f"{a.out}/{k}.npy", store[k])
json.dump({"global_step": step, "rows": rows.tolist(), "vars": VARS, "guidances": GS,
           "T": a.T, "keys": keys,
           "factual": {k: vals[k].tolist() for k in COND},
           "targets": {v: {k: cfs[v][k].tolist() for k in COND} for v in VARS}},
          open(f"{a.out}/manifest.json", "w"), indent=2)
print(f"wrote {a.out}  step={step}")
