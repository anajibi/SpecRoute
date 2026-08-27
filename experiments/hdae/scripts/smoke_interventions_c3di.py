"""Per-attribute intervention smoke test -- no pooling, one number per attribute.

One cohort of N images. Every intervention is applied to that SAME cohort, so the rows of
the output table are directly comparable and the image grid can show all interventions on
the same scenes side by side.

For each intervention we render the counterfactual, then read ALL SEVEN attributes back
off the generated image with the trained predictors and report the error of each against
the value it is *supposed* to have:

    the intervened attribute   -> its TARGET            (did the edit land?)
    a causal descendant        -> the SCM's PROPAGATED value  (the graph says it must move)
    anything else              -> the SOURCE value      (it must hold still)

Nothing is averaged across attributes. `class` is scored as accuracy, everything else as
mean absolute error in raw [-1,1] units, so a cell is always comparable down its own column.

Outputs: a JSON table, a cohort table (source / target / delta per image), and an image
grid of source + reconstruction + one column per intervention.

    python experiments/hdae/scripts/smoke_interventions_c3di.py --n 32
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/hdae/causal"))

from PIL import Image, ImageDraw  # noqa: E402
from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper  # noqa: E402
from experiments.hdae.data.causal3dident import (ATTRIBUTE_NAMES, CLASS_NAMES,  # noqa: E402
                                                 Causal3DIdentPacked)
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
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--min-delta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(REPO, "experiments/hdae/outputs/smoke_interventions"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.RandomState(args.seed)

    ds = Causal3DIdentPacked(os.path.join(REPO, "experiments/hdae/data/causal3dident/causal3dident_testset_128.h5"))
    idx = sorted(rng.choice(len(ds), args.n, replace=False).tolist())
    rows = [ds[i] for i in idx]
    X = torch.stack([r["img"] for r in rows])
    A = torch.stack([r["attr"] for r in rows])

    # ---- targets: same cohort, one target per modelled attribute --------------
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

    cohort = []
    for i in range(len(idx)):
        e = {"i": int(idx[i]), "class_src": CLASS_NAMES[int(A[i, 0])],
             "class_tgt": CLASS_NAMES[int(targets["class"][i, 0])]}
        for a in ["pos_spl", "pos_obj", "rot_obj"]:
            s = A[i, COLS[a]].tolist(); t = targets[a][i].tolist()
            e[a] = {"source": [round(v, 3) for v in s], "target": [round(v, 3) for v in t],
                    "abs_delta": [round(abs(x - y), 3) for x, y in zip(s, t)]}
        cohort.append(e)
    dmin = min(min(e[a]["abs_delta"]) for e in cohort for a in ["pos_spl", "pos_obj", "rot_obj"])
    print(f"cohort: {len(idx)} images | smallest |target-source| across all scalar dims = {dmin:.3f} "
          f"(floor {args.min_delta})", flush=True)

    # ---- models --------------------------------------------------------------
    cfg = load_hdae_config(os.path.join(REPO, args.config), require_data=False)
    module = HDAELitModule.load_from_checkpoint(os.path.join(REPO, args.ckpt),
                                                conf=cfg.train_conf, map_location="cpu").to(dev).eval()
    sampler = module.conf._make_diffusion_conf(args.T).make_sampler()
    specs = cfg.hdae_conf.encoder.cond_specs
    blob = torch.load(os.path.join(REPO, "experiments/hdae/outputs/scm/causal3dident_scm_spline.pt"),
                      map_location=dev)
    c = blob["config"]
    scm = SCM(CausalGraph(c["attributes"], c["edges"]), c["nodes"],
              mechanism=blob["mechanism"], bins=blob["bins"]).to(dev)
    scm.load_state_dict(blob["state_dict"]); scm.eval()
    preds = {a: load_predictor(a, dev) for a in ALL7}

    x = X.to(dev)
    y = to_cond_values(A[:, :8], specs).to(dev)
    with torch.no_grad():
        zs = [z.clone() for z in module.ema_model.encode(x)]
        cond = module.ema_model.make_cond(zs, y)
        x_T = sampler.ddim_reverse_sample_loop(module.ema_model, x, model_kwargs={"cond": cond})["sample"]
        with torch.inference_mode():
            recon = sampler.sample(model=module.ema_model, noise=x_T, model_kwargs={"cond": cond}).cpu()
    print("reconstruction done", flush=True)

    def read_all(img):
        return {a: predict(preds[a][0], img, dev) for a in ALL7}

    panels = [("source", X), ("recon", recon)]
    table, prop_store = {}, {}
    read_recon = read_all(recon)

    for a in MODELLED:
        desc = descendants(a)
        obs = {k: y[:, COLS[k]].contiguous() for k in MODELLED}
        with torch.no_grad():
            cfa = scm.propagate(scm.abduct(obs), obs, {a: targets[a].to(dev)})
            y_cf = y.clone()
            for k in MODELLED:
                y_cf[:, COLS[k]] = cfa[k].to(y.dtype)
            cond_cf = module.ema_model.make_cond(zs, y_cf)
            m = module.ema_model if args.guidance == 1.0 else \
                AttributeCFGWrapper(module.ema_model, args.guidance).to(dev).eval()
            with torch.inference_mode():
                cf = sampler.sample(model=m, noise=x_T, model_kwargs={"cond": cond_cf}).cpu()
        panels.append((f"do({a})", cf))
        prop_store[a] = {k: cfa[k].cpu() for k in MODELLED}
        read = read_all(cf)

        row = {}
        for k in ALL7:
            if k == a:
                ref, role = targets[a], "target"
            elif k in desc:
                ref, role = prop_store[a][k], "descendant (SCM)"
            else:
                ref, role = A[:, COLS[k]].float(), "unchanged"
            if k == "class":
                acc = float((read[k].argmax(1) == ref[:, 0].long()).float().mean())
                row[k] = {"metric": "accuracy", "value": round(acc, 4), "role": role}
            else:
                mae = float((read[k] - ref).abs().mean())
                row[k] = {"metric": "mae", "value": round(mae, 5), "role": role,
                          "per_dim": [round(v, 5) for v in (read[k] - ref).abs().mean(0).tolist()]}
        table[f"do({a})"] = row
        print(f"do({a:8s}) " + "  ".join(
            f"{k}={row[k]['value']:.4f}{'*' if row[k]['role']!='unchanged' else ''}" for k in ALL7), flush=True)

    # baseline row: the reconstruction itself, everything should equal source
    base = {}
    for k in ALL7:
        ref = A[:, COLS[k]].float()
        if k == "class":
            base[k] = {"metric": "accuracy",
                       "value": round(float((read_recon[k].argmax(1) == ref[:, 0].long()).float().mean()), 4),
                       "role": "unchanged"}
        else:
            base[k] = {"metric": "mae", "value": round(float((read_recon[k] - ref).abs().mean()), 5),
                       "role": "unchanged"}
    table["reconstruction (no intervention)"] = base

    # ---- image grid ----------------------------------------------------------
    cell, pad, hdr = 128, 5, 20
    W = len(panels) * (cell + pad) + pad
    H = hdr + len(idx) * (cell + pad) + pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for ci, (name, imgs) in enumerate(panels):
        d.text((pad + ci * (cell + pad) + 2, 5), name, fill=(15, 15, 25))
        arr = ((imgs.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(0, 2, 3, 1).numpy()
        for ri in range(len(idx)):
            sheet.paste(Image.fromarray(arr[ri]), (pad + ci * (cell + pad), hdr + ri * (cell + pad)))
    grid_path = os.path.join(args.outdir, f"grid_{args.label}.png")
    sheet.save(grid_path)

    out = {"model": args.label, "ckpt": args.ckpt, "guidance": args.guidance, "n": len(idx),
           "predictor_test_mae": {a: preds[a][1]["test_metrics"].get("mae") for a in ALL7 if a != "class"},
           "min_abs_delta": round(float(dmin), 4), "table": table, "cohort": cohort,
           "grid": os.path.relpath(grid_path, REPO)}
    jp = os.path.join(args.outdir, f"smoke_{args.label}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {jp}\nwrote {grid_path}  {sheet.size}")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
