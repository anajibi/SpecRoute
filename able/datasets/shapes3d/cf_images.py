"""Reconstructions and counterfactuals for a trained 3DShapes Graph A model.

WHY THIS DATASET MAKES THE CLAIM CHECKABLE. The 3DShapes grid is a FULL FACTORIAL: every one of
the 480,000 factor combinations exists as a render. So for any intervention the TRUE
counterfactual image is not a matter of opinion -- `scm_exact.counterfactual_index` names the
exact grid cell, and we can put the model's counterfactual beside the real render of it. On
CelebA or Causal3DIdent no such ground truth exists and the figure can only be eyeballed.

Four rows per intervention:

    source        the real held-out render, the factual input
    recon         DDIM-encode -> decode, no guidance. Isolates autoencoding quality from
                  conditioning: anything broken here is not an intervention failure.
    cf            the counterfactual. do(X) propagates downstream through the SCM -- abduct,
                  act, predict -- so the model is conditioned on every post-intervention value,
                  not just the intervened one. This is the only causally valid way to build it.
    truth         the real render of the counterfactual cell. `cf` should match THIS.

Graph A is shape -> scale -> floor_hue <- object_hue, so the four interventions differ in how far
they propagate: do(shape) moves scale AND floor_hue (two steps), do(object_hue) and do(scale)
move floor_hue only, and do(floor_hue) is a sink that must leave everything else alone. The last
one is a control in the other direction -- a model that "propagates" anything there is wrong.

Sources are drawn from the HELD-OUT cells, which the model never trained on.

Every panel is also SCORED, not just drawn: the six predictors (1.000000 held-out accuracy each)
read the generated images back and are compared against the SCM's exact targets. That is the
quantitative form of the same claim, and it is also the first time those instruments are asked
to score diffusion samples rather than real renders.

    python able/datasets/shapes3d/cf_images.py --ckpt <path> --config <yaml> --n 6
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from able.datasets.shapes3d.scm_exact import (CARDINALITY, LAYOUT_COLUMNS,  # noqa: E402
                                              MODELED, abduce, counterfactual_index,
                                              decode_index, grid_index, intervene)
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule  # noqa: E402

COND = ["shape", "object_hue", "scale", "floor_hue"]   # model conditioning column order
ROWS = ["source", "recon", "cf", "truth"]


def load_model(config, ckpt, device):
    cfg = load_hdae_config(config, require_data=False)
    lit = HDAELitModule(cfg.train_conf)
    blob = torch.load(ckpt, map_location="cpu")
    sd = blob.get("state_dict", blob)
    ema = {k[len("ema_model."):]: v for k, v in sd.items() if k.startswith("ema_model.")}
    if not ema:
        raise RuntimeError("checkpoint has no ema_model.* keys")
    missing, unexpected = lit.ema_model.load_state_dict(ema, strict=True), None
    lit = lit.to(device).eval()
    step = int(blob.get("global_step", -1))
    print(f"loaded EMA weights from {os.path.basename(ckpt)} (global_step={step}, "
          f"{len(ema)} keys)", flush=True)
    return lit, cfg, step


def y_of(fac, specs, device):
    """(n,4) raw-unit conditioning tensor from a dict of factor arrays."""
    y = np.stack([fac[k] for k in COND], axis=1).astype(np.float32)
    return to_cond_values(torch.from_numpy(y).to(device), specs)


@torch.no_grad()
def render(lit, sampler, x_T, cond, guidance):
    model = lit.ema_model
    if guidance == 1.0:
        out = sampler.sample(model=model, noise=x_T, model_kwargs={"cond": cond})
    else:
        from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper
        m = AttributeCFGWrapper(model, guidance).to(x_T.device).eval()
        out = sampler.sample(model=m, noise=x_T, model_kwargs={"cond": cond})
    return ((out + 1) / 2).clamp(0, 1)


@torch.no_grad()
def score(imgs, predictors, device):
    """Read all six factors off a batch of images in [0,1]. Returns (n,6) int array."""
    IM = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    IS = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (imgs.to(device) - IM) / IS
    out = np.zeros((len(imgs), 6), dtype=np.int64)
    for j, name in enumerate(LAYOUT_COLUMNS):
        with torch.autocast("cuda", dtype=torch.float16):
            p = predictors[name](x)
        out[:, j] = p.float().argmax(1).cpu().numpy()
    return out


def load_predictors(pdir, device):
    from able.datasets.shapes3d.train_predictors import Predictor
    preds = {}
    for name in LAYOUT_COLUMNS:
        blob = torch.load(os.path.join(pdir, f"{name}.pt"), map_location="cpu")
        m = Predictor(blob["n_classes"], 0.0, pretrained=False)
        m.load_state_dict(blob["state_dict"])
        preds[name] = m.to(device).eval()
    return preds


def annotate(grid_np, labels, cell, pad_top, pad_left, row_names):
    from PIL import Image, ImageDraw
    h, w = grid_np.shape[:2]
    canvas = Image.new("RGB", (w + pad_left, h + pad_top), (255, 255, 255))
    canvas.paste(Image.fromarray(grid_np), (pad_left, pad_top))
    d = ImageDraw.Draw(canvas)
    for i, txt in enumerate(labels):
        d.text((pad_left + i * cell + 3, 4), txt, fill=(0, 0, 0))
    for r, nm in enumerate(row_names):
        d.text((4, pad_top + r * cell + cell // 2 - 4), nm, fill=(0, 0, 0))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="able/datasets/shapes3d/configs/shapes3d_a_k11_d256.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", default="/home/exouser/able_data/src_3dshapes/3dshapes.h5")
    ap.add_argument("--layout-dir", default="/home/exouser/able_data/shapes3d/graph_a/v1")
    ap.add_argument("--predictors", default="experiments/hdae/outputs/attr_predictors_shapes3d")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="experiments/hdae/outputs/shapes3d_cf_figs")
    ap.add_argument("--mem-frac", type=float, default=0.0,
                    help="hard cap on this process's GPU share; protects a concurrent training run")
    args = ap.parse_args()

    dev = torch.device("cuda")
    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    lit, cfg, step = load_model(args.config, args.ckpt, dev)
    specs = lit.model.hdae_conf.encoder.cond_specs
    sampler = lit.conf._make_diffusion_conf(args.T).make_sampler()
    preds = load_predictors(args.predictors, dev)

    heldout = np.load(f"{args.layout_dir}/heldout_cells.npy").astype(np.int64)
    rng = np.random.RandomState(args.seed)
    cells = np.sort(rng.choice(heldout, args.n, replace=False))
    Hf, Hw, Ho, C, S, R = decode_index(cells)
    fac = {"shape": S, "object_hue": Ho, "scale": C, "floor_hue": Hf}
    U_C, U_F = abduce(S, Ho, C, Hf)
    print("sources (held-out cells):")
    for i in range(args.n):
        print(f"  cell {cells[i]:6d}  shape={S[i]} object_hue={Ho[i]} scale={C[i]} "
              f"floor_hue={Hf[i]} | wall_hue={Hw[i]} orient={R[i]}  U_C={U_C[i]} U_F={U_F[i]}")

    with h5py.File(args.h5, "r") as f:
        imgs = f["images"][:]

    def grab(idx):
        a = np.stack([imgs[int(g)] for g in np.atleast_1d(idx)])
        return torch.from_numpy(a).permute(0, 3, 1, 2).float().div_(127.5).sub_(1).to(dev)

    x = grab(cells)
    y = y_of(fac, specs, dev)
    with torch.no_grad():
        zs = [z.clone() for z in lit.ema_model.encode(x)]
        cond = lit.ema_model.make_cond(zs, y)
        x_T = sampler.ddim_reverse_sample_loop(
            lit.ema_model, x, model_kwargs={"cond": cond})["sample"]
        recon = render(lit, sampler, x_T, cond, 1.0)

    src01 = ((x + 1) / 2).clamp(0, 1)
    results = {"checkpoint": os.path.basename(args.ckpt), "global_step": step,
               "guidance": args.guidance, "T": args.T, "cells": cells.tolist(),
               "interventions": {}}
    # Reconstruction fidelity, and what the instruments read off a reconstruction.
    rec_pred = score(recon, preds, dev)
    tgt_fac = np.stack([S, Ho, C, Hf, Hw, R], axis=1)
    results["recon"] = {
        "l1_vs_source": float((src01 - recon).abs().mean()),
        "factor_agreement": {n: float((rec_pred[:, j] == tgt_fac[:, j]).mean())
                             for j, n in enumerate(LAYOUT_COLUMNS)}}
    print(f"\nrecon: L1 vs source {results['recon']['l1_vs_source']:.4f}  "
          f"factor agreement {results['recon']['factor_agreement']}")

    import torchvision
    for var in MODELED:
        K = CARDINALITY[var]
        # a target different from the factual value for every source, deterministic given seed
        cur = fac[var]
        val = (cur + 1 + rng.randint(0, K - 1, args.n)) % K
        S2, Ho2, C2, Hf2 = intervene(S, Ho, C, Hf, var, val)
        cf_fac = {"shape": S2, "object_hue": Ho2, "scale": C2, "floor_hue": Hf2}
        truth_idx = counterfactual_index(S, Ho, C, Hf, Hw, R, var, val)

        with torch.no_grad():
            img_prop = render(lit, sampler, x_T, lit.ema_model.make_cond(zs, y_of(cf_fac, specs, dev)),
                              args.guidance)
        truth = ((grab(truth_idx) + 1) / 2).clamp(0, 1)

        tgt = np.stack([S2, Ho2, C2, Hf2, Hw, R], axis=1)
        p_prop = score(img_prop, preds, dev)
        moved = [n for j, n in enumerate(LAYOUT_COLUMNS) if (tgt[:, j] != tgt_fac[:, j]).any()]
        rec = {"target_values": val.tolist(), "factors_that_move": moved,
               "truth_cells": np.atleast_1d(truth_idx).tolist(),
               "accuracy": {n: float((p_prop[:, j] == tgt[:, j]).mean())
                            for j, n in enumerate(LAYOUT_COLUMNS)},
               "l1_vs_truth": float((img_prop - truth).abs().mean())}
        results["interventions"][var] = rec
        print(f"\ndo({var}) -> moves {moved}")
        print(f"  accuracy     {rec['accuracy']}")
        print(f"  L1 vs truth  {rec['l1_vs_truth']:.4f}")

        stack = torch.cat([src01, recon, img_prop, truth]).cpu()
        g = torchvision.utils.make_grid(stack, nrow=args.n, padding=2)
        gnp = (g.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        short = {"shape": "shp", "object_hue": "obj", "scale": "scl", "floor_hue": "flr"}[var]
        labels = [f"{short} {cur[i]}>{val[i]}" for i in range(args.n)]
        canvas = annotate(gnp, labels, 66, 18, 62, ROWS)
        out = os.path.join(args.outdir, f"cf_{var}_step{step}.png")
        canvas.save(out)
        print(f"  wrote {out}")

    with open(os.path.join(args.outdir, f"cf_report_step{step}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.outdir}/cf_report_step{step}.json")


if __name__ == "__main__":
    main()
