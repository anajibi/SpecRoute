"""Attribute predictors for Causal3DIdent: one ConvNeXt-Tiny per attribute.

These are the measurement instruments for CC / FC / CF1 later -- they score generated
images, so they must be stronger than the generator's own conditioning, not weaker.

One model per attribute (7 models, 11 outputs total):

    class     categorical, 7 classes        -> cross-entropy, accuracy
    pos_spl   scalar, 1 dim                 -> MSE, MAE / R^2
    pos_obj   scalar, 3 dims (x, y, z)      -> MSE, MAE / R^2
    rot_obj   scalar, 3 dims (alpha,beta,gamma)
    hue_obj   scalar, 1 dim
    hue_spl   scalar, 1 dim
    hue_bg    scalar, 1 dim

All continuous targets are plain scalars, NOT circular. Verified empirically: sweeping
hue_bg from -1 to +1 walks the background purple -> magenta -> red -> orange -> yellow ->
green, i.e. about half a colour wheel with clearly different colours at the two ends, so
there is no wraparound to model. The rotations likewise span [-pi/2, pi/2], a half turn.

AUGMENTATION IS DELIBERATELY MINIMAL. Every standard augmentation is label-destroying for
at least one attribute here: horizontal flip negates pos_x and rotation angles, random
crop/translate moves pos_obj, colour jitter destroys all three hue labels, and rotation
changes rot_obj. Only photometric noise and slight blur leave every label intact, so those
are the only ones used; overfitting is otherwise controlled with dropout and weight decay.

    python experiments/hdae/scripts/train_causal3dident_attr_predictors.py --attr pos_obj
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from experiments.hdae.data.causal3dident import (ATTRIBUTE_NAMES, CLASS_NAMES,  # noqa: E402
                                                 Causal3DIdentPacked)

DATA = os.path.join(REPO, "experiments/hdae/data/causal3dident")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# name -> (dataset columns, kind, output dim)
ATTRS = {
    "class":   ([ATTRIBUTE_NAMES.index("class")], "categorical", 7),
    "pos_spl": ([ATTRIBUTE_NAMES.index("pos_spl")], "scalar", 1),
    "pos_obj": ([ATTRIBUTE_NAMES.index(f"pos_obj_{j}") for j in range(3)], "scalar", 3),
    "rot_obj": ([ATTRIBUTE_NAMES.index(f"rot_obj_{j}") for j in range(3)], "scalar", 3),
    "hue_obj": ([ATTRIBUTE_NAMES.index("hue_obj")], "scalar", 1),
    "hue_spl": ([ATTRIBUTE_NAMES.index("hue_spl")], "scalar", 1),
    "hue_bg":  ([ATTRIBUTE_NAMES.index("hue_bg")], "scalar", 1),
}


class Predictor(nn.Module):
    def __init__(self, out_dim, dropout=0.2, pretrained=True):
        super().__init__()
        w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = convnext_tiny(weights=w)
        in_f = self.net.classifier[2].in_features
        self.net.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, out_dim))

    def forward(self, x):
        return self.net(x)


def label_preserving_aug(img, noise_std, blur_p, gen):
    """img: (B,3,H,W) in [-1,1]. Only photometric ops that leave every attribute label
    unchanged -- no flip/crop/rotate (geometry) and no colour jitter (hue)."""
    if noise_std > 0:
        img = img + torch.randn(img.shape, device=img.device, generator=gen) * noise_std
    if blur_p > 0:
        b = img.shape[0]
        pick = torch.rand(b, device=img.device, generator=gen) < blur_p
        if pick.any():
            k = torch.tensor([1.0, 2.0, 1.0], device=img.device)
            k = (k[:, None] * k[None, :]); k = (k / k.sum()).expand(3, 1, 3, 3)
            blurred = F.conv2d(F.pad(img[pick], (1, 1, 1, 1), mode="reflect"), k, groups=3)
            img = img.clone(); img[pick] = blurred
    return img.clamp(-1, 1)


def prep(img, size):
    """[-1,1] -> ImageNet-normalized, optionally resized."""
    x = (img + 1) / 2
    if size and size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


@torch.no_grad()
def evaluate(model, loader, cols, kind, size, device, max_batches=None):
    model.eval()
    P, T = [], []
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = prep(b["img"].to(device, non_blocking=True), size)
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x)
        P.append(p.float().cpu()); T.append(b["attr"][:, cols].float())
    P, T = torch.cat(P), torch.cat(T)
    if kind == "categorical":
        pred = P.argmax(1)
        tgt = T[:, 0].long()
        acc = (pred == tgt).float().mean().item()
        per = {CLASS_NAMES[c]: float((pred[tgt == c] == c).float().mean())
               for c in range(len(CLASS_NAMES)) if (tgt == c).any()}
        return {"accuracy": acc, "per_class_accuracy": per}
    mae = (P - T).abs().mean(0)
    ss_res = ((P - T) ** 2).sum(0)
    ss_tot = ((T - T.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    return {"mae_per_dim": [round(v, 5) for v in mae.tolist()],
            "mae": round(mae.mean().item(), 5),
            "r2_per_dim": [round(v, 5) for v in r2.tolist()],
            "r2": round(r2.mean().item(), 5),
            "target_sd_per_dim": [round(v, 5) for v in T.std(0).tolist()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True, choices=list(ATTRS))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--blur-p", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(REPO, "experiments/hdae/outputs/attr_predictors_c3di"))
    ap.add_argument("--limit-train-batches", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    cols, kind, out_dim = ATTRS[args.attr]
    os.makedirs(args.outdir, exist_ok=True)

    full = Causal3DIdentPacked(os.path.join(DATA, "causal3dident_trainset_128.h5"))
    idx = np.arange(len(full)); np.random.RandomState(args.seed).shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_ds, tr_ds = Subset(full, idx[:n_val].tolist()), Subset(full, idx[n_val:].tolist())
    test_ds = Causal3DIdentPacked(os.path.join(DATA, "causal3dident_testset_128.h5"))

    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
                                   pin_memory=True, drop_last=sh, persistent_workers=args.workers > 0)
    tr, va, te = mk(tr_ds, True), mk(val_ds, False), mk(test_ds, False)

    model = Predictor(out_dim, args.dropout, not args.no_pretrained).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.limit_train_batches or len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps,
                                                pct_start=0.25)
    scaler = torch.cuda.amp.GradScaler()
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)

    print(f"attr={args.attr} kind={kind} out_dim={out_dim} cols={cols}", flush=True)
    print(f"train={len(tr_ds)} val={len(val_ds)} test={len(test_ds)}  params={nparam/1e6:.1f}M  "
          f"img_size={args.img_size} pretrained={not args.no_pretrained}", flush=True)

    best, best_ep = None, -1
    key = "accuracy" if kind == "categorical" else "mae"
    better = (lambda a, b: a > b) if kind == "categorical" else (lambda a, b: a < b)
    ckpt = os.path.join(args.outdir, f"{args.attr}.pt")

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; n = 0
        for i, b in enumerate(tr):
            if args.limit_train_batches and i >= args.limit_train_batches:
                break
            img = b["img"].to(dev, non_blocking=True)
            img = label_preserving_aug(img, args.noise_std, args.blur_p, gen)
            x = prep(img, args.img_size)
            t = b["attr"][:, cols].to(dev, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.float16):
                p = model(x)
                loss = F.cross_entropy(p, t[:, 0].long()) if kind == "categorical" else F.mse_loss(p, t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * img.shape[0]; n += img.shape[0]
        m = evaluate(model, va, cols, kind, args.img_size, dev)
        cur = m[key]
        star = ""
        if best is None or better(cur, best):
            best, best_ep = cur, ep
            torch.save({"state_dict": model.state_dict(), "attr": args.attr, "kind": kind,
                        "cols": cols, "out_dim": out_dim, "img_size": args.img_size,
                        "val_metrics": m, "epoch": ep, "args": vars(args)}, ckpt)
            star = "  *best"
        print(f"  ep {ep}  train_loss {tot/max(n,1):.5f}  val_{key} {cur:.5f}  "
              f"({time.time()-t0:.0f}s){star}", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    test_m = evaluate(model, te, cols, kind, args.img_size, dev)
    print(f"BEST epoch {best_ep}  val_{key}={best:.5f}")
    print(f"TEST {json.dumps(test_m)}", flush=True)
    blob = torch.load(ckpt, map_location="cpu"); blob["test_metrics"] = test_m
    torch.save(blob, ckpt)
    with open(os.path.join(args.outdir, f"{args.attr}_metrics.json"), "w") as f:
        json.dump({"attr": args.attr, "kind": kind, "best_epoch": best_ep,
                   "val": blob["val_metrics"], "test": test_m}, f, indent=2)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
