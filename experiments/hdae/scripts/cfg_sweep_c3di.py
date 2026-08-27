"""Guidance-strength sweep: images and per-attribute measurements at every cell.

Renders the same fixed cohort under every intervention at every guidance strength, for
both the EMA weights and the raw training weights, and reads all seven attributes back off
each result. The DDIM inversion is computed ONCE per weight set and reused for every
(intervention x strength) cell -- it does not depend on the conditioning vector, and it is
~35% of the cost, so sharing it is free speed.

On the two weight sets: everything in this project has used `module.ema_model` -- the
encoder, the inversion, the reconstruction, and both passes inside AttributeCFGWrapper
(which holds ONE base model and calls it twice, so conditional and unconditional can never
diverge). `module.model` -- the raw weights the optimiser actually updates -- has never
been evaluated. Running both is what tells you whether EMA is doing real work here.

Guidance 1.0 is a genuine baseline, not a degenerate case: the wrapper short-circuits and
returns the conditional pass alone, so it costs half as much and shows what conditioning
achieves with no extrapolation at all.

T=50 IS THE OPERATING POINT, not T=100. Measured against T=100 on the same seed-0 cohort
(k=11, g=3): class 100.00% vs 100.00%, pos_spl 0.0276 vs 0.0275, pos_obj 0.0326 vs 0.0321,
rot_obj 0.0752 vs 0.0790 (inside that cell's bootstrap CI). Exactly 2x faster, no measurable
change. T=25 was tested and REJECTED -- pos_obj degrades 49% and pos_spl 41%, a systematic
bias large enough to move conclusions. T=10 is destroyed.

Two speed levers that do NOT work here, both measured so nobody re-tries them: --fp16 gives
1.149x on a bare UNet forward and 0% in the sampling loop (metrics identical to 4 decimals),
and --compile produces NaNs AND is slower (162s vs 156s per cell). The loop is memory-bound
-- GroupNorm, SiLU, attention softmax, kernel launches -- not matmul-bound, so precision and
batching tricks are dead ends. The levers that work are T and doing fewer cells.

    python experiments/hdae/scripts/cfg_sweep_c3di.py --strengths 1 2 3 5 8 --T 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/hdae/causal"))

from PIL import Image, ImageDraw  # noqa: E402
from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper  # noqa: E402
from experiments.hdae.data.causal3dident import CLASS_NAMES, Causal3DIdentPacked  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule  # noqa: E402
from torchvision.models import convnext_tiny  # noqa: E402
from train_scm_causal3dident import SCM, CausalGraph  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PRED_DIR = os.path.join(REPO, "experiments/hdae/outputs/attr_predictors_c3di")
MODELLED = ["class", "pos_spl", "pos_obj", "rot_obj"]
ALL7 = MODELLED + ["hue_obj", "hue_spl", "hue_bg"]
COLS = {"class": [0], "pos_spl": [1], "pos_obj": [2, 3, 4], "rot_obj": [5, 6, 7],
        "hue_obj": [8], "hue_spl": [9], "hue_bg": [10]}
EDGES = [("class", "rot_obj"), ("class", "pos_obj"), ("pos_spl", "pos_obj")]


def descendants(a):
    out, fr = set(), [a]
    while fr:
        n = fr.pop()
        for p, c in EDGES:
            if p == n and c not in out:
                out.add(c); fr.append(c)
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
def predict(model, img, device, bs=32):
    out = []
    for i in range(0, img.shape[0], bs):
        x = (img[i:i + bs].to(device) + 1) / 2
        x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/hdae/configs/c3di_hier_k1_final.yaml")
    ap.add_argument("--ckpt", default="experiments/hdae/outputs/c3di_k1_final/checkpoints/last.ckpt")
    ap.add_argument("--label", default="k1")
    ap.add_argument("--strengths", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--weights", nargs="+", default=["ema", "raw"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--min-delta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(REPO, "experiments/hdae/outputs/cfg_sweep"))
    ap.add_argument("--fp16", action="store_true",
                    help="run the diffusion loop under autocast fp16; verify metrics before trusting")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the sampling net; ~2-4 min warmup, amortised over a long sweep")
    ap.add_argument("--sample-bs", type=int, default=64,
                    help="images per diffusion call; a 256-image cohort will not fit in one")
    ap.add_argument("--grid-rows", type=int, default=0,
                    help="rows to render in the contact sheet; 0 skips grids entirely")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.RandomState(args.seed)

    ds = Causal3DIdentPacked(os.path.join(REPO, "experiments/hdae/data/causal3dident/causal3dident_testset_128.h5"))
    idx = sorted(rng.choice(len(ds), args.n, replace=False).tolist())
    rows = [ds[i] for i in idx]
    X = torch.stack([r["img"] for r in rows]); A = torch.stack([r["attr"] for r in rows])

    targets = {}
    for a in MODELLED:
        src = A[:, COLS[a]].float()
        if a == "class":
            targets[a] = ((src[:, 0].long() + 3) % 7).float()[:, None]
        else:
            t = torch.empty_like(src)
            for j in range(src.shape[1]):
                v = torch.from_numpy(rng.uniform(-0.9, 0.9, len(idx)).astype(np.float32))
                for _ in range(300):
                    bad = (v - src[:, j]).abs() < args.min_delta
                    if not bad.any():
                        break
                    v[bad] = torch.from_numpy(rng.uniform(-0.9, 0.9, int(bad.sum())).astype(np.float32))
                t[:, j] = v
            targets[a] = t

    cfg = load_hdae_config(os.path.join(REPO, args.config), require_data=False)
    module = HDAELitModule.load_from_checkpoint(os.path.join(REPO, args.ckpt),
                                                conf=cfg.train_conf, map_location="cpu").to(dev).eval()
    sampler = module.conf._make_diffusion_conf(args.T).make_sampler()
    specs = cfg.hdae_conf.encoder.cond_specs
    b = torch.load(os.path.join(REPO, "experiments/hdae/outputs/scm/causal3dident_scm_spline.pt"), map_location=dev)
    c = b["config"]
    scm = SCM(CausalGraph(c["attributes"], c["edges"]), c["nodes"],
              mechanism=b["mechanism"], bins=b["bins"]).to(dev)
    scm.load_state_dict(b["state_dict"]); scm.eval()
    preds = {a: load_predictor(a, dev) for a in ALL7}

    x = X.to(dev); y = to_cond_values(A[:, :8], specs).to(dev)
    results, panels, per_sample = {}, {}, {}

    def score(key, k, read, ref):
        """Mean for the table, per-sample vector for the bootstrap."""
        if k == "class":
            e = (read.argmax(1) == ref[:, 0].long()).float()
            per_sample[f"{key}|{k}"] = e.numpy()
            return {"metric": "accuracy", "value": round(float(e.mean()), 4)}
        e = (read - ref).abs().mean(dim=1)
        per_sample[f"{key}|{k}"] = e.numpy()
        return {"metric": "mae", "value": round(float(e.mean()), 5)}

    BS = args.sample_bs
    chunks = [(i, min(i + BS, len(idx))) for i in range(0, len(idx), BS)]

    def render(net, model, y_vec, x_T, zs):
        """Sample the whole cohort in GPU-sized chunks. 256 at once does not fit in 40 GB."""
        out = []
        for lo, hi in chunks:
            with torch.no_grad():
                c = net.make_cond([z[lo:hi] for z in zs], y_vec[lo:hi])
            with torch.no_grad(), torch.inference_mode(), \
                    torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                out.append(sampler.sample(model=model, noise=x_T[lo:hi],
                                          model_kwargs={"cond": c}).float().cpu())
        return torch.cat(out)

    for wname in args.weights:
        net = module.ema_model if wname == "ema" else module.model
        net.eval()
        if args.compile:
            net = torch.compile(net)
        with torch.no_grad():
            zs = [z.clone() for z in net.encode(x)]
            inv = []
            for lo, hi in chunks:
                c = net.make_cond([z[lo:hi] for z in zs], y[lo:hi])
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                    inv.append(sampler.ddim_reverse_sample_loop(
                        net, x[lo:hi], model_kwargs={"cond": c})["sample"].float())
            x_T = torch.cat(inv)
        recon = render(net, net, y, x_T, zs)
        if args.grid_rows:
            panels[(wname, "recon", None)] = recon[:args.grid_rows]
        read_r = {a: predict(preds[a][0], recon, dev) for a in ALL7}
        results[f"{wname}|recon"] = {
            a: dict(score(f"{wname}|recon", a, read_r[a], A[:, COLS[a]].float()), role="unchanged")
            for a in ALL7}
        print(f"[{wname}] reconstruction done", flush=True)

        for a in MODELLED:
            desc = descendants(a)
            obs = {k: y[:, COLS[k]].contiguous() for k in MODELLED}
            with torch.no_grad():
                cfa = scm.propagate(scm.abduct(obs), obs, {a: targets[a].to(dev)})
                y_cf = y.clone()
                for k in MODELLED:
                    y_cf[:, COLS[k]] = cfa[k].to(y.dtype)
            for g in args.strengths:
                m = net if g == 1.0 else AttributeCFGWrapper(net, g).to(dev).eval()
                t0 = time.time()
                img = render(net, m, y_cf, x_T, zs)
                if args.grid_rows:
                    panels[(wname, a, g)] = img[:args.grid_rows]
                read = {k: predict(preds[k][0], img, dev) for k in ALL7}
                key = f"{wname}|do({a})|g{g:g}"
                row = {}
                for k in ALL7:
                    if k == a:
                        ref, role = targets[a], "target"
                    elif k in desc:
                        ref, role = cfa[k].cpu(), "descendant"
                    else:
                        ref, role = A[:, COLS[k]].float(), "unchanged"
                    row[k] = dict(score(key, k, read[k], ref), role=role)
                results[key] = row
                tgt = row[a]["value"]
                print(f"  [{wname}] do({a:8s}) g={g:<4g}  target-cell {tgt:.4f}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)

    # ---- grids: one PNG per weight set, columns = source, recon, then each strength ----
    grids = {}
    nrows = min(args.grid_rows, len(idx))
    for wname in args.weights if args.grid_rows else []:
        cols = [("source", X[:nrows]), ("recon", panels[(wname, "recon", None)])]
        for a in MODELLED:
            for g in args.strengths:
                cols.append((f"do({a})\ng={g:g}", panels[(wname, a, g)]))
        cell, pad, hdr = 128, 4, 30
        W = len(cols) * (cell + pad) + pad
        H = hdr + nrows * (cell + pad) + pad
        sheet = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(sheet)
        for ci, (nm, imgs) in enumerate(cols):
            for li, line in enumerate(nm.split("\n")):
                d.text((pad + ci * (cell + pad) + 2, 3 + li * 12), line, fill=(15, 15, 25))
            arr = ((imgs.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(0, 2, 3, 1).numpy()
            for ri in range(nrows):
                sheet.paste(Image.fromarray(arr[ri]), (pad + ci * (cell + pad), hdr + ri * (cell + pad)))
        p = os.path.join(args.outdir, f"grid_{args.label}_{wname}.png")
        sheet.save(p); grids[wname] = os.path.relpath(p, REPO)
        print(f"wrote {p}  {sheet.size}", flush=True)

    cohort = [{"i": int(idx[i]), "class_src": CLASS_NAMES[int(A[i, 0])],
               "class_tgt": CLASS_NAMES[int(targets['class'][i, 0])],
               **{a: {"source": [round(v, 3) for v in A[i, COLS[a]].tolist()],
                      "target": [round(v, 3) for v in targets[a][i].tolist()]}
                  for a in ["pos_spl", "pos_obj", "rot_obj"]}} for i in range(len(idx))]
    out = {"model": args.label, "ckpt": args.ckpt, "T": args.T, "n": len(idx),
           "strengths": args.strengths, "weights": args.weights, "grids": grids,
           "predictor_test_mae": {a: preds[a][1]["test_metrics"].get("mae") for a in ALL7 if a != "class"},
           "results": results, "cohort": cohort}
    np.savez_compressed(os.path.join(args.outdir, f"persample_{args.label}.npz"), **per_sample)
    jp = os.path.join(args.outdir, f"sweep_{args.label}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {jp}")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
