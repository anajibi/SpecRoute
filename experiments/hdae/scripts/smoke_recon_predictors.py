"""Smoke test: is reconstruction good, and do the predictors actually work end to end?

This runs BEFORE any counterfactual scoring, because CC/FC numbers are meaningless if
either half of the measurement chain is broken. It checks three things in order, each of
which isolates one failure mode:

  A. PREDICTORS ON REAL IMAGES -- the control. Feeds untouched test images through the
     predictors and compares to ground truth. If this does not reproduce the numbers from
     predictor training, the inference pipeline (normalisation, channel order, resize,
     column mapping) is wrong -- not the generator. Everything downstream is void.

  B. RECONSTRUCTION QUALITY -- DDIM-encode each image to x_T, decode with no guidance,
     compare to the source: MSE / PSNR / LPIPS.

  C. PREDICTORS ON RECONSTRUCTIONS -- the same predictors reading the model's output.
     The gap between B and C is what the autoencode round-trip costs in *attribute* terms,
     which is the quantity CC and FC are built on.

Range discipline is the point of this script, so every hand-off is printed and asserted:

    dataset __getitem__ -> CHW float in [-1, 1]
    predictor input     -> (x+1)/2 gives [0,1], then ImageNet mean/std  (NOT [0,1] direct)
    model encode/sample -> [-1, 1] throughout; the render is NOT rescaled to [0,1] here,
                           which is the easy mistake: feeding a [0,1] render into a prep()
                           that expects [-1,1] silently halves the contrast.
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

from experiments.hdae.data.causal3dident import ATTRIBUTE_NAMES, CLASS_NAMES, Causal3DIdentPacked  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule  # noqa: E402
from torchvision.models import convnext_tiny  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PRED_DIR = os.path.join(REPO, "experiments/hdae/outputs/attr_predictors_c3di")
COND_COLS = list(range(8))          # class, pos_spl, pos_obj(3), rot_obj(3)


def load_predictor(attr, device):
    """The checkpoints were saved from a wrapper holding `self.net = convnext_tiny(...)`,
    so every key is prefixed `net.`. Strip it, then the weights load into a plain
    torchvision ConvNeXt with a replaced head. (Loading without stripping fails loudly
    with ~350 missing/unexpected keys -- it does not silently half-load.)"""
    blob = torch.load(os.path.join(PRED_DIR, f"{attr}.pt"), map_location="cpu")
    sd = {k[len("net."):] if k.startswith("net.") else k: v for k, v in blob["state_dict"].items()}
    m = convnext_tiny()
    in_f = m.classifier[2].in_features
    m.classifier[2] = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(in_f, blob["out_dim"]))
    missing, unexpected = m.load_state_dict(sd, strict=True), None
    return m.to(device).eval(), blob


def prep_for_predictor(img_pm1, size=None):
    """img_pm1: (B,3,H,W) in [-1,1]  ->  ImageNet-normalised. Mirrors training exactly."""
    x = (img_pm1 + 1) / 2
    if size and size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


@torch.no_grad()
def predict(model, blob, img_pm1, device, bs=64):
    out = []
    for i in range(0, img_pm1.shape[0], bs):
        x = prep_for_predictor(img_pm1[i:i + bs].to(device), blob.get("img_size"))
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x).float().cpu())
    return torch.cat(out)


def score(attr, pred, truth):
    if attr == "class":
        cls = pred.argmax(1)
        tgt = truth[:, 0].long()
        return {"accuracy": float((cls == tgt).float().mean()),
                "n_wrong": int((cls != tgt).sum())}
    mae = (pred - truth).abs().mean(0)
    ss_res = ((pred - truth) ** 2).sum(0)
    ss_tot = ((truth - truth.mean(0)) ** 2).sum(0)
    return {"mae": float(mae.mean()), "mae_per_dim": [round(float(v), 5) for v in mae],
            "r2": float((1 - ss_res / ss_tot).mean()),
            "pred_range": [round(float(pred.min()), 4), round(float(pred.max()), 4)],
            "truth_range": [round(float(truth.min()), 4), round(float(truth.max()), 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/hdae/configs/c3di_hier_k1_final.yaml")
    ap.add_argument("--ckpt", default="experiments/hdae/outputs/c3di_k1_final/checkpoints/last.ckpt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/hdae/outputs/smoke_recon_predictors.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    ds = Causal3DIdentPacked(os.path.join(REPO, "experiments/hdae/data/causal3dident/causal3dident_testset_128.h5"))
    idx = sorted(np.random.RandomState(args.seed).choice(len(ds), args.n, replace=False).tolist())
    batch = [ds[i] for i in idx]
    X = torch.stack([b["img"] for b in batch])            # (N,3,128,128) in [-1,1]
    A = torch.stack([b["attr"] for b in batch])           # (N,11) raw units

    print("=" * 78)
    print("RANGE CHECK 0 -- dataset output")
    print(f"  images  {tuple(X.shape)}  dtype={X.dtype}  range [{X.min():.4f}, {X.max():.4f}]  "
          f"mean {X.mean():+.4f}")
    assert -1.001 <= X.min() and X.max() <= 1.001, "dataset images are not in [-1,1]"
    print(f"  attrs   {tuple(A.shape)}  range [{A.min():.4f}, {A.max():.4f}]  "
          f"(class col: {A[:,0].min():.0f}..{A[:,0].max():.0f})")
    xp = prep_for_predictor(X[:4])
    print(f"  predictor input after ImageNet norm: [{xp.min():.3f}, {xp.max():.3f}] "
          f"(expected roughly [-2.12, 2.64])")

    # ---------- A. predictors on REAL images ----------
    print("\n" + "=" * 78)
    print("A. PREDICTORS ON REAL IMAGES  (control -- must match training-time test metrics)")
    preds = {}
    results = {"A_real": {}, "B_recon_quality": {}, "C_recon_attrs": {}}
    for attr in ["class", "pos_spl", "pos_obj", "rot_obj", "hue_obj", "hue_spl", "hue_bg"]:
        m, blob = load_predictor(attr, dev)
        cols = blob["cols"]
        truth = A[:, cols].float()
        p = predict(m, blob, X, dev)
        s = score(attr, p, truth)
        results["A_real"][attr] = s
        preds[attr] = (m, blob)
        ref = blob.get("test_metrics", {})
        if attr == "class":
            print(f"  {attr:8s} accuracy {s['accuracy']*100:7.3f}%  ({s['n_wrong']} wrong of {args.n})"
                  f"   [training-time test: {ref.get('accuracy', float('nan'))*100:.3f}%]")
        else:
            print(f"  {attr:8s} MAE {s['mae']:.5f}  R2 {s['r2']:.5f}  "
                  f"pred{s['pred_range']} truth{s['truth_range']}"
                  f"   [test MAE {ref.get('mae', float('nan')):.5f}]")

    # ---------- B. reconstruction ----------
    print("\n" + "=" * 78)
    print("B. RECONSTRUCTION QUALITY")
    cfg = load_hdae_config(os.path.join(REPO, args.config), require_data=False)
    module = HDAELitModule.load_from_checkpoint(os.path.join(REPO, args.ckpt),
                                                conf=cfg.train_conf, map_location="cpu").to(dev).eval()
    sampler = module.conf._make_diffusion_conf(args.T).make_sampler()
    specs = cfg.hdae_conf.encoder.cond_specs
    import lpips
    lp = lpips.LPIPS(net="alex").to(dev).eval()

    R = torch.empty_like(X)
    for i in range(0, args.n, args.batch):
        x = X[i:i + args.batch].to(dev)
        y = to_cond_values(A[i:i + args.batch, COND_COLS], specs).to(dev)
        with torch.no_grad():
            zs = [z.clone() for z in module.ema_model.encode(x)]
            cond = module.ema_model.make_cond(zs, y)
            x_T = sampler.ddim_reverse_sample_loop(module.ema_model, x, model_kwargs={"cond": cond})["sample"]
            with torch.inference_mode():
                rec = sampler.sample(model=module.ema_model, noise=x_T, model_kwargs={"cond": cond})
        R[i:i + args.batch] = rec.cpu()      # kept in [-1,1] on purpose -- see module docstring
        print(f"    reconstructed {min(i+args.batch, args.n)}/{args.n}", flush=True)

    print(f"\n  RANGE CHECK 1 -- model render: [{R.min():.4f}, {R.max():.4f}] mean {R.mean():+.4f}"
          f"   (must be ~[-1,1], NOT [0,1])")
    mse = F.mse_loss(R, X).item()
    psnr = 10 * np.log10(4.0 / max(mse, 1e-12))
    lpv = torch.cat([lp(R[i:i+32].to(dev).clamp(-1, 1), X[i:i+32].to(dev)).flatten().cpu()
                     for i in range(0, args.n, 32)]).mean().item()
    results["B_recon_quality"] = {"mse": round(mse, 6), "psnr": round(float(psnr), 3),
                                  "lpips": round(lpv, 5),
                                  "render_range": [round(float(R.min()), 4), round(float(R.max()), 4)]}
    print(f"  MSE {mse:.6f}   PSNR {psnr:.2f} dB   LPIPS {lpv:.5f}")

    # ---------- C. predictors on reconstructions ----------
    print("\n" + "=" * 78)
    print("C. PREDICTORS ON RECONSTRUCTIONS  (gap vs A = what the round-trip costs)")
    for attr, (m, blob) in preds.items():
        truth = A[:, blob["cols"]].float()
        p = predict(m, blob, R, dev)
        s = score(attr, p, truth)
        results["C_recon_attrs"][attr] = s
        a = results["A_real"][attr]
        if attr == "class":
            print(f"  {attr:8s} accuracy {s['accuracy']*100:7.3f}%   real {a['accuracy']*100:7.3f}%"
                  f"   delta {(s['accuracy']-a['accuracy'])*100:+.3f} pts")
        else:
            print(f"  {attr:8s} MAE {s['mae']:.5f}   real {a['mae']:.5f}"
                  f"   degradation x{s['mae']/max(a['mae'],1e-9):5.2f}   R2 {s['r2']:.5f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
