"""Reconstruction + counterfactual image grids for a trained Causal3DIdent HDAE arm.

For each source image: DDIM-encode to x_T, re-render (reconstruction), then render a
counterfactual for each intervention. Counterfactual attribute vectors come from the
trained spline SCM via the real abduct -> intervene -> predict recipe, so descendants
propagate (e.g. do(class) moves pos_obj/rot_obj) instead of only the intervened column
changing.

Self-contained apart from the model/config loaders; does NOT use counterfactuals/
hdae_adapter.py, which is still scalar-per-attribute and would mis-index the 8 raw
columns that this dataset's 4 vector-valued attributes span.

    python experiments/hdae/scripts/cf_grid_causal3dident.py \
        --config experiments/hdae/configs/c3di_hier_k1_both.yaml \
        --ckpt   .../epoch=6-step=50000.ckpt --out grid.png
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/hdae/causal"))

from experiments.hdae.data.causal3dident import Causal3DIdentPacked, CLASS_NAMES  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule  # noqa: E402
from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper  # noqa: E402
from train_scm_causal3dident import SCM, CausalGraph  # noqa: E402

# model conditioning column layout: class(1) pos_spl(1) pos_obj(3) rot_obj(3)
COLS = {"class": [0], "pos_spl": [1], "pos_obj": [2, 3, 4], "rot_obj": [5, 6, 7]}
ORDER = ["class", "pos_spl", "pos_obj", "rot_obj"]
# dataset column layout (see data/causal3dident.ATTRIBUTE_NAMES) -> same first 8 columns
DS_COLS = [0, 1, 2, 3, 4, 5, 6, 7]


def y_to_scm(y):
    return {k: y[:, COLS[k]].contiguous() for k in ORDER}


def scm_to_y(d, like):
    y = like.clone()
    for k in ORDER:
        y[:, COLS[k]] = d[k].to(y.device, y.dtype)
    return y


def to_img(t):
    """model output in [0,1] (B,3,H,W) -> list of PIL"""
    a = (t.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(x) for x in a]


@torch.no_grad()
def render(module, sampler, x_T, cond, guidance):
    model = module.ema_model
    m = model if guidance == 1.0 else AttributeCFGWrapper(model, guidance).to(x_T.device).eval()
    with torch.inference_mode():
        out = sampler.sample(model=m, noise=x_T, model_kwargs={"cond": cond})
    return (out + 1) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scm", default="experiments/hdae/outputs/scm/causal3dident_scm_spline.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    cfg = load_hdae_config(args.config, require_data=False)
    module = HDAELitModule.load_from_checkpoint(args.ckpt, conf=cfg.train_conf,
                                                map_location="cpu").to(dev).eval()
    sampler = module.conf._make_diffusion_conf(args.T).make_sampler()
    specs = cfg.hdae_conf.encoder.cond_specs

    blob = torch.load(os.path.join(REPO, args.scm), map_location=dev)
    scm_cfg = blob["config"]
    scm = SCM(CausalGraph(scm_cfg["attributes"], scm_cfg["edges"]), scm_cfg["nodes"],
              mechanism=blob.get("mechanism", "gaussian"), bins=blob.get("bins", 16)).to(dev)
    scm.load_state_dict(blob["state_dict"]); scm.eval()

    ds = Causal3DIdentPacked(cfg.raw["data"]["test_h5_path"], preload_images=False)
    rng = np.random.RandomState(args.seed)
    idx = sorted(rng.choice(len(ds), args.n, replace=False).tolist())
    batch = [ds[i] for i in idx]
    x = torch.stack([b["img"] for b in batch]).to(dev)
    y_raw = torch.stack([b["attr"] for b in batch])[:, DS_COLS].to(dev)
    y = to_cond_values(y_raw, specs).to(dev)
    src_cls = [int(v) for v in y[:, 0].tolist()]

    with torch.no_grad():
        zs = [z.clone() for z in module.ema_model.encode(x)]
        cond = module.ema_model.make_cond(zs, y)
        x_T = sampler.ddim_reverse_sample_loop(module.ema_model, x, model_kwargs={"cond": cond})["sample"]

    panels = [("source", to_img((x + 1) / 2)),
              ("recon", to_img(render(module, sampler, x_T, cond, 1.0)))]

    obs = y_to_scm(y)
    u = scm.abduct(obs)
    interventions = [
        ("do(class=dragon)", {"class": torch.full((1, 1), 2.0, device=dev)}),
        ("do(pos_spl=+0.9)", {"pos_spl": torch.full((1, 1), 0.9, device=dev)}),
        ("do(pos_spl=-0.9)", {"pos_spl": torch.full((1, 1), -0.9, device=dev)}),
        ("do(rot_obj=+0.9)", {"rot_obj": torch.full((1, 3), 0.9, device=dev)}),
    ]
    for name, iv in interventions:
        with torch.no_grad():
            cf = scm.propagate(u, obs, iv)
            y_cf = scm_to_y(cf, y)
            cond_cf = module.ema_model.make_cond(zs, y_cf)
            panels.append((name, to_img(render(module, sampler, x_T, cond_cf, args.guidance))))

    cell, pad, hdr = 128, 6, 22
    W = len(panels) * (cell + pad) + pad
    H = hdr + args.n * (cell + pad) + pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for c, (name, imgs) in enumerate(panels):
        d.text((pad + c * (cell + pad) + 2, 6), name, fill=(0, 0, 0))
        for r, im in enumerate(imgs):
            sheet.paste(im, (pad + c * (cell + pad), hdr + r * (cell + pad)))
    sheet.save(args.out)
    print(f"wrote {args.out}  ({len(panels)} panels x {args.n} rows)  "
          f"source classes = {[CLASS_NAMES[c] for c in src_cls]}", flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
