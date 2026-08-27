"""CC / FC_observed / FC_unobserved / CF1 for Causal3DIdent.

For each modelled attribute we take its fixed 256-image cohort, DDIM-encode each source
image, ask the SCM what the world looks like under `do(attr = target)`, render with
attribute-CFG, and then read every attribute back off the generated image with the trained
predictors. Metric definitions follow the repo's conventions (AGENDA sec.10), instantiated
on this graph:

    class -> rot_obj,  class -> pos_obj,  pos_spl -> pos_obj

CC (counterfactual consistency) -- did the edit land? Pooled success over
  {the intervened attribute reaches its target} union {each causal DESCENDANT matches the
  value the SCM propagated for it}. Descendants are scored against the SCM's prediction,
  not against the source, because the graph says they are supposed to move.

FC_observed -- did everything else hold still? The modelled attributes that are NOT the
  target and NOT its descendants must still read as their SOURCE values. Scored strictly.

FC_unobserved -- the three hues, outside the graph entirely. Same test, no causal claim.

CF1 -- harmonic mean of CC with each FC pool, reported separately for observed and
  unobserved since the pools differ in size and in what they assert.

TOLERANCE is the load-bearing choice. A continuous attribute "matches" when the predicted
value is within `mult * (that predictor's own test-set MAE)`. Tying the window to the
instrument's measured error is what stops the previous failure mode, where MorphoMNIST's
tolerance (0.055) was half its predictor's error (0.115) and capped CC below 1.0 no matter
how good the model was. Results are reported at several `mult` so the sensitivity is
visible rather than hidden in one number. Categorical attributes use exact match and need
no tolerance at all.

    python experiments/hdae/scripts/eval_cf1_c3di.py --guidance 3.0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/hdae/causal"))

from experiments.hdae.data.causal3dident import ATTRIBUTE_NAMES, Causal3DIdentPacked  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule  # noqa: E402
from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper  # noqa: E402
from torchvision.models import convnext_tiny  # noqa: E402
from train_scm_causal3dident import SCM, CausalGraph  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PRED_DIR = os.path.join(REPO, "experiments/hdae/outputs/attr_predictors_c3di")
COHORT_DIR = os.path.join(REPO, "experiments/hdae/outputs/cohorts_c3di")

MODELLED = ["class", "pos_spl", "pos_obj", "rot_obj"]
UNOBSERVED = ["hue_obj", "hue_spl", "hue_bg"]
COLS = {"class": [0], "pos_spl": [1], "pos_obj": [2, 3, 4], "rot_obj": [5, 6, 7]}
EDGES = [("class", "rot_obj"), ("class", "pos_obj"), ("pos_spl", "pos_obj")]


def descendants(attr):
    out, frontier = set(), [attr]
    while frontier:
        n = frontier.pop()
        for p, c in EDGES:
            if p == n and c not in out:
                out.add(c); frontier.append(c)
    return out


def load_predictor(attr, device):
    blob = torch.load(os.path.join(PRED_DIR, f"{attr}.pt"), map_location="cpu")
    sd = {k[len("net."):] if k.startswith("net.") else k: v for k, v in blob["state_dict"].items()}
    m = convnext_tiny()
    in_f = m.classifier[2].in_features
    m.classifier[2] = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(in_f, blob["out_dim"]))
    m.load_state_dict(sd)
    return m.to(device).eval(), blob


@torch.no_grad()
def predict(model, img_pm1, device, bs=64):
    out = []
    for i in range(0, img_pm1.shape[0], bs):
        x = (img_pm1[i:i + bs].to(device) + 1) / 2
        x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/hdae/configs/c3di_hier_k1_final.yaml")
    ap.add_argument("--ckpt", default="experiments/hdae/outputs/c3di_k1_final/checkpoints/last.ckpt")
    ap.add_argument("--label", default="k1")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=0, help="cap cohort size (0 = use all)")
    ap.add_argument("--mults", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--scm", default="experiments/hdae/outputs/scm/causal3dident_scm_spline.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0); np.random.seed(0)

    # models
    cfg = load_hdae_config(os.path.join(REPO, args.config), require_data=False)
    module = HDAELitModule.load_from_checkpoint(os.path.join(REPO, args.ckpt),
                                                conf=cfg.train_conf, map_location="cpu").to(dev).eval()
    sampler = module.conf._make_diffusion_conf(args.T).make_sampler()
    specs = cfg.hdae_conf.encoder.cond_specs
    blob = torch.load(os.path.join(REPO, args.scm), map_location=dev)
    c = blob["config"]
    scm = SCM(CausalGraph(c["attributes"], c["edges"]), c["nodes"],
              mechanism=blob.get("mechanism", "gaussian"), bins=blob.get("bins", 16)).to(dev)
    scm.load_state_dict(blob["state_dict"]); scm.eval()
    preds = {a: load_predictor(a, dev) for a in MODELLED + UNOBSERVED}
    tol_base = {a: preds[a][1]["test_metrics"]["mae"] for a in MODELLED + UNOBSERVED if a != "class"}
    print(f"model={args.label}  guidance={args.guidance}  T={args.T}")
    print("per-attribute tolerance base (predictor test MAE): " +
          "  ".join(f"{a}={v:.5f}" for a, v in tol_base.items()), flush=True)

    ds = Causal3DIdentPacked(os.path.join(REPO, "experiments/hdae/data/causal3dident/causal3dident_testset_128.h5"))
    results = {}

    for attr in MODELLED:
        z = np.load(os.path.join(COHORT_DIR, f"{attr}.npz"), allow_pickle=True)
        idx = z["indices"]; target = torch.from_numpy(z["target"]).float()
        if args.n:
            idx, target = idx[:args.n], target[:args.n]
        rows = [ds[int(i)] for i in idx]
        X = torch.stack([r["img"] for r in rows])
        A = torch.stack([r["attr"] for r in rows])
        desc = descendants(attr)
        print(f"\n=== do({attr}) on {len(idx)} images   descendants={sorted(desc) or 'none'} ===", flush=True)

        CF = torch.empty_like(X)
        prop = {a: torch.zeros(len(idx), len(COLS[a])) for a in MODELLED}
        for i in range(0, len(idx), args.batch):
            x = X[i:i + args.batch].to(dev)
            y = to_cond_values(A[i:i + args.batch, :8], specs).to(dev)
            tg = target[i:i + args.batch].to(dev)
            obs = {a: y[:, COLS[a]].contiguous() for a in MODELLED}
            with torch.no_grad():
                cfa = scm.propagate(scm.abduct(obs), obs, {attr: tg})
                y_cf = y.clone()
                for a in MODELLED:
                    y_cf[:, COLS[a]] = cfa[a].to(y.dtype)
                    prop[a][i:i + args.batch] = cfa[a].cpu()
                zs = [zz.clone() for zz in module.ema_model.encode(x)]
                cond = module.ema_model.make_cond(zs, y)
                x_T = sampler.ddim_reverse_sample_loop(module.ema_model, x,
                                                       model_kwargs={"cond": cond})["sample"]
                cond_cf = module.ema_model.make_cond(zs, y_cf)
                m = module.ema_model if args.guidance == 1.0 else \
                    AttributeCFGWrapper(module.ema_model, args.guidance).to(dev).eval()
                with torch.inference_mode():
                    CF[i:i + args.batch] = sampler.sample(model=m, noise=x_T,
                                                          model_kwargs={"cond": cond_cf}).cpu()
            print(f"    {min(i+args.batch, len(idx))}/{len(idx)}", flush=True)

        # read every attribute off the counterfactual image
        read = {a: predict(preds[a][0], CF, dev) for a in MODELLED + UNOBSERVED}

        per_mult = {}
        for mult in args.mults:
            def ok(a, pred, ref):
                if a == "class":
                    return (pred.argmax(1) == ref[:, 0].long()).float()
                t = mult * tol_base[a]
                return ((pred - ref).abs() <= t).float().mean(1)

            # CC: target attribute + descendants (descendants judged against SCM propagation)
            cc_parts = [ok(attr, read[attr], target)]
            for d in sorted(desc):
                cc_parts.append(ok(d, read[d], prop[d]))
            cc = torch.stack(cc_parts).mean().item()

            obs_pool = [a for a in MODELLED if a != attr and a not in desc]
            fc_o = torch.stack([ok(a, read[a], A[:, COLS[a]].float()) for a in obs_pool]).mean().item() \
                if obs_pool else float("nan")
            un_cols = {a: [ATTRIBUTE_NAMES.index(a)] for a in UNOBSERVED}
            fc_u = torch.stack([ok(a, read[a], A[:, un_cols[a]].float()) for a in UNOBSERVED]).mean().item()
            hm = lambda a, b: 0.0 if (a + b) == 0 or np.isnan(b) else 2 * a * b / (a + b)
            per_mult[f"mult{mult:g}"] = {
                "CC": round(cc, 4), "FC_observed": round(fc_o, 4), "FC_unobserved": round(fc_u, 4),
                "CF1_observed": round(hm(cc, fc_o), 4), "CF1_unobserved": round(hm(cc, fc_u), 4),
                "cc_components": {a: round(v.mean().item(), 4) for a, v in
                                  zip([attr] + sorted(desc), cc_parts)},
                "n_observed_attrs": len(obs_pool), "n_unobserved_attrs": len(UNOBSERVED),
            }
            print(f"  mult {mult:g}:  CC {cc:.4f}   FC_obs {fc_o:.4f}   FC_unobs {fc_u:.4f}"
                  f"   CF1_obs {hm(cc, fc_o):.4f}", flush=True)
        results[attr] = {"n": int(len(idx)), "descendants": sorted(desc), **per_mult}

    out = args.out or os.path.join(REPO, f"experiments/hdae/outputs/cf1_{args.label}_g{args.guidance:g}.json")
    with open(out, "w") as f:
        json.dump({"model": args.label, "ckpt": args.ckpt, "guidance": args.guidance,
                   "T": args.T, "tolerance_base_mae": tol_base, "results": results}, f, indent=2)
    print(f"\nwrote {out}")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
