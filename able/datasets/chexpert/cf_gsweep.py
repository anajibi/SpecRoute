"""Guidance sweep for do(edema) on one CheXpert k=1 checkpoint, with per-sample detail.

Two questions this answers that the single-guidance run could not:

 1. WHICH FLIPS LANDED. A hit rate of 4/16 hides its own direction. Edema is ~26% prevalent,
    so flipping every source turns 11 negatives into "add edema" requests and 5 positives
    into "remove edema" requests. Those are not the same task -- adding a diffuse alveolar
    opacity is a different edit from clearing one -- and a model that only ever succeeds in
    one direction is telling you something specific. Every sample is recorded.

 2. WHETHER GUIDANCE IS THE BINDING CONSTRAINT. g=3.0 was inherited from CelebA-HQ, where it
    was tuned on faces. If the conditioning signal exists but is weak, raising g should trade
    preservation for effect and CC should climb before it collapses. If CC stays flat and only
    the artifacts grow, the signal is not there yet and no amount of guidance invents it.

THE EXPENSIVE PART IS SHARED. DDIM-encoding a source to x_T costs as much as rendering from
it, and x_T does not depend on the guidance scale. So each source is encoded ONCE and every
guidance value renders from that same x_T and the same latents -- which also makes the
comparison exact rather than merely paired, since nothing but g differs between the rows.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as Fn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
import able.datasets.chexpert.scm_exact as S                      # noqa: E402
from able.adapters.base import cc_from_error, fc_from_error       # noqa: E402
from able.datasets.chexpert.cf_images import (COND, Pred, PRED_BINARY,  # noqa: E402
                                              read_back, render, y_of)
from experiments.hdae.hdae.config_io import load_hdae_config      # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/hdae/configs/chexpert256_k1.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--predictor", default="/home/exouser/able_data/chexpert/predictors/best.ckpt")
    ap.add_argument("--out", default="outputs/chexpert_cf/gsweep.json")
    ap.add_argument("--guidances", default="1,3,5,8,12")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    gs = [float(x) for x in a.guidances.split(",")]
    dev = "cuda"

    cfg = load_hdae_config(a.config, require_data=False)
    lit = HDAELitModule(cfg.train_conf)
    blob = torch.load(a.ckpt, map_location="cpu")
    sd = blob.get("state_dict", blob)
    ema = {k[len("ema_model."):]: v for k, v in sd.items() if k.startswith("ema_model.")}
    lit.ema_model.load_state_dict(ema, strict=True)
    lit.ema_model = lit.ema_model.to(dev).eval()          # EMA only; the trainer owns the rest
    lit.eval()
    step = int(blob.get("global_step", -1))
    specs = lit.model.hdae_conf.encoder.cond_specs
    sampler = cfg.train_conf._make_diffusion_conf(a.T).make_sampler()

    pb = torch.load(a.predictor, map_location="cpu")
    pred = Pred(); pred.load_state_dict(pb["state_dict"], strict=True)
    pred = pred.to(dev).eval()

    import pandas as pd
    meta = pd.read_csv(f"{a.data}/chexpert_meta.csv")
    split = np.load(f"{a.data}/split.npy")
    unc = pd.read_csv(f"{a.data}/label_uncertainty.csv")
    pool = np.flatnonzero((split == 1) & (unc["uncertain_edema"].to_numpy() == 0))
    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(pool, a.n, replace=False))       # SAME seed, same 16 sources
    arr = np.load(f"{a.data}/images_224.npy", mmap_mode="r")
    vals = {k: meta.loc[rows, k].to_numpy(np.int64) for k in COND}
    params = S.load(f"{a.data}/scm_params.json")
    tgt_edema = 1 - vals["edema"]
    cf, _ = S.counterfactual(vals, params, "edema", tgt_edema, seed=a.seed)
    cf = {k: np.asarray(cf[k]) for k in COND}

    pr = json.load(open(f"{a.data}/predictors/predictor_report.json"))
    tsrc = pr.get("thresholds_youden_on_select_v2", pr["thresholds_youden_on_select"])
    thr = {k: (float(tsrc[k]) if not isinstance(tsrc[k], dict) else float(tsrc[k]["threshold"]))
           for k in ("sex", "effusion", "edema")}
    print(f"step {step}  guidances {gs}  n={len(rows)}  "
          f"factual edema {np.bincount(vals['edema'],minlength=2).tolist()}", flush=True)

    per_g = {g: {"edema": [], "effusion": [], "sex": [], "age_bin": []} for g in gs}
    recon_edema = []
    for i in range(len(rows)):
        x = torch.from_numpy(np.asarray(arr[rows[i]], dtype=np.float32) / 127.5 - 1.0)[None, None]
        x = Fn.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False).clamp_(-1, 1)
        x = x.expand(-1, 3, -1, -1).contiguous().to(dev)
        yf = y_of({k: vals[k][i:i+1] for k in COND}, specs, dev)
        with torch.no_grad():
            zs = [z.clone() for z in lit.ema_model.encode(x)]
            cond = lit.ema_model.make_cond(zs, yf)
            x_T = sampler.ddim_reverse_sample_loop(lit.ema_model, x,
                                                   model_kwargs={"cond": cond})["sample"]
            rec = render(lit, sampler, x_T, cond, 1.0).cpu()
            pbn, _ = read_back(rec, pred, dev)
            recon_edema.append(int(pbn[0, PRED_BINARY.index("edema")] >= thr["edema"]))
            ycf = y_of({k: cf[k][i:i+1] for k in COND}, specs, dev)
            ccond = lit.ema_model.make_cond(zs, ycf)
            for g in gs:                      # same x_T, same latents: only g differs
                im = render(lit, sampler, x_T, ccond, g).cpu()
                pbn, pan = read_back(im, pred, dev)
                per_g[g]["edema"].append(int(pbn[0, PRED_BINARY.index("edema")] >= thr["edema"]))
                per_g[g]["effusion"].append(int(pbn[0, PRED_BINARY.index("effusion")] >= thr["effusion"]))
                per_g[g]["sex"].append(int(pbn[0, PRED_BINARY.index("sex")] >= thr["sex"]))
                per_g[g]["age_bin"].append(int(pan[0].argmax()))
        print(f"  source {i+1}/{len(rows)}", flush=True)

    add = tgt_edema == 1            # factual 0 -> asked to ADD edema
    rem = tgt_edema == 0            # factual 1 -> asked to REMOVE edema
    _, cnt = np.unique(tgt_edema, return_counts=True)
    triv = 1.0 - cnt.max() / len(rows)
    rec_hit = float((np.array(recon_edema) == tgt_edema).mean())

    out = {"checkpoint": os.path.basename(a.ckpt), "global_step": step, "T": a.T,
           "n": int(len(rows)), "rows": rows.tolist(),
           "factual_edema": vals["edema"].tolist(), "target_edema": tgt_edema.tolist(),
           "recon_edema_readback": recon_edema,
           "inaction_baseline": rec_hit,
           "trivial_error": triv, "constant_predictor_hit": 1.0 - triv,
           "n_add": int(add.sum()), "n_remove": int(rem.sum()), "by_guidance": {}}
    print(f"\n{'g':>5s} {'hit':>7s} {'CC':>8s} | {'add 0->1':>9s} {'rem 1->0':>9s} | "
          f"{'FC eff':>7s} {'FC sex':>7s} {'FC age':>7s}")
    print(f"{'inact':>5s} {rec_hit:7.3f} {cc_from_error(1-rec_hit,triv):8.3f} | "
          f"{'-':>9s} {'-':>9s} |")
    for g in gs:
        e = np.array(per_g[g]["edema"])
        hit = float((e == tgt_edema).mean())
        cc = cc_from_error(1.0 - hit, triv)
        a_rate = float((e[add] == 1).mean()) if add.any() else None
        r_rate = float((e[rem] == 0).mean()) if rem.any() else None
        fcs = {}
        for k in ("effusion", "sex", "age_bin"):
            got = np.array(per_g[g][k]); err = float((got != vals[k]).mean())
            _, c2 = np.unique(vals[k], return_counts=True); t2 = 1.0 - c2.max() / len(rows)
            fcs[k] = fc_from_error(err, t2) if t2 > 0 else None
        out["by_guidance"][str(g)] = {
            "hit": hit, "cc": cc, "add_success": a_rate, "remove_success": r_rate,
            "fc": fcs, "edema_readback": e.tolist()}
        f = lambda v: "   n/a" if v is None else f"{v:7.3f}"
        print(f"{g:5g} {hit:7.3f} {cc:8.3f} | "
              f"{(a_rate if a_rate is not None else float('nan')):9.3f} "
              f"{(r_rate if r_rate is not None else float('nan')):9.3f} | "
              f"{f(fcs['effusion'])} {f(fcs['sex'])} {f(fcs['age_bin'])}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nadd requests {int(add.sum())}  remove requests {int(rem.sum())}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
