"""Attribute predictors for 3DShapes Graph A: one ConvNeXt-Tiny per factor.

These are the measurement instruments for CF evaluation -- they score GENERATED images, so
they must be stronger than the generator's own conditioning, not weaker. Six models, one per
factor, every one CATEGORICAL:

    shape         4 classes    conditioned
    object_hue   10 classes    conditioned
    scale         8 classes    conditioned
    floor_hue    10 classes    conditioned
    wall_hue     10 classes    UNOBSERVED -- needed to test exogenous preservation
    orientation  15 classes    UNOBSERVED -- needed to test exogenous preservation

All six are categorical on purpose, matching the generator's embedding-table conditioning.
`scale` wraps mod 8 in the SCM (index 7 (+)1 = 0) and the hues wrap mod 10, so an L1/MAE score
would read a legitimate wrap as the largest possible miss. Accuracy is EXACT MATCH. Adjacent
confusion is reported separately as a diagnostic (cyclic neighbours for the wrapping factors),
never folded into the headline number.

SPLIT. Train on the 432,000 non-held-out cells, i.e. exactly the cells ABLE is allowed to see.
A 2% slice of those is carved off for epoch selection. The reported accuracy is on the 48,000
HELD-OUT cells, which no model in this pipeline has trained on -- that is what the hold-out is
for, and it makes "almost perfect" a claim about unseen factor combinations rather than about
memorised ones. Each cell is a unique factor tuple appearing exactly once, so there is no
duplicate-image leakage between the splits.

AUGMENTATION IS DELIBERATELY MINIMAL, and more constrained here than for Causal3DIdent. Every
standard augmentation is label-destroying for at least one of the six factors: horizontal flip
changes the azimuth, i.e. `orientation`; colour jitter destroys all three hue labels; random
crop or resize changes `scale`; rotation changes `orientation`. Only photometric noise and a
slight blur leave all six labels intact, so those are the only ones used.

    python able/datasets/shapes3d/train_predictors.py --attr orientation
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from able.datasets.shapes3d.scm_exact import (CARDINALITY, LAYOUT_COLUMNS,  # noqa: E402
                                              N_GRID, decode_index)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Which factors wrap, so adjacency is cyclic for the off-by-one diagnostic. `shape` is a bare
# nominal label -- cube/cylinder/sphere/pill have no order at all, so no adjacency is reported.
CYCLIC = {"object_hue": 10, "scale": 8, "floor_hue": 10, "wall_hue": 10}
ORDERED = {"orientation": 15}


class GridSubset(Dataset):
    """Images addressed by grid index; the six factor labels decoded from the index itself."""

    def __init__(self, images, cells):
        self.images = images
        self.cells = np.asarray(cells, dtype=np.int64)
        Hf, Hw, Ho, C, S, R = decode_index(self.cells)
        # LAYOUT_COLUMNS order: shape, object_hue, scale, floor_hue, wall_hue, orientation
        self.labels = np.stack([S, Ho, C, Hf, Hw, R], axis=1).astype(np.int64)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, i):
        g = int(self.cells[i])
        img = torch.from_numpy(self.images[g].copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
        return {"img": img, "y": torch.from_numpy(self.labels[i])}


class Predictor(nn.Module):
    def __init__(self, n_classes, dropout=0.2, pretrained=True):
        super().__init__()
        w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = convnext_tiny(weights=w)
        in_f = self.net.classifier[2].in_features
        self.net.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, n_classes))

    def forward(self, x):
        return self.net(x)


def label_preserving_aug(img, noise_std, blur_p, gen):
    """img: (B,3,H,W) in [-1,1]. Photometric only -- see the module docstring for why no flip,
    crop or colour jitter is admissible for ANY of the six factors."""
    if noise_std > 0:
        img = img + torch.randn(img.shape, device=img.device, generator=gen) * noise_std
    if blur_p > 0:
        pick = torch.rand(img.shape[0], device=img.device, generator=gen) < blur_p
        if pick.any():
            k = torch.tensor([1.0, 2.0, 1.0], device=img.device)
            k = (k[:, None] * k[None, :])
            k = (k / k.sum()).expand(3, 1, 3, 3)
            blurred = F.conv2d(F.pad(img[pick], (1, 1, 1, 1), mode="reflect"), k, groups=3)
            img = img.clone()
            img[pick] = blurred
    return img.clamp(-1, 1)


def prep(img, size):
    x = (img + 1) / 2
    if size and size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


@torch.no_grad()
def evaluate(model, loader, col, n_classes, size, device, attr, max_batches=0):
    model.eval()
    P, T = [], []
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = prep(b["img"].to(device, non_blocking=True), size)
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x)
        P.append(p.float().argmax(1).cpu())
        T.append(b["y"][:, col])
    pred, tgt = torch.cat(P), torch.cat(T)
    acc = (pred == tgt).float().mean().item()
    per = {int(c): round(float((pred[tgt == c] == c).float().mean()), 5)
           for c in range(n_classes) if (tgt == c).any()}
    out = {"n": int(len(tgt)), "accuracy": round(acc, 6),
           "n_wrong": int((pred != tgt).sum()), "per_class_accuracy": per,
           "worst_class": min(per, key=per.get) if per else None,
           "worst_class_accuracy": min(per.values()) if per else None}
    # Off-by-one diagnostic, cyclic where the factor wraps. NOT part of the headline accuracy.
    if attr in CYCLIC:
        K = CYCLIC[attr]
        d = (pred - tgt) % K
        out["adjacent_error_rate"] = round(float(((d == 1) | (d == K - 1)).float().mean()), 6)
        out["adjacency"] = "cyclic"
    elif attr in ORDERED:
        out["adjacent_error_rate"] = round(float(((pred - tgt).abs() == 1).float().mean()), 6)
        out["adjacency"] = "linear"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True, choices=list(LAYOUT_COLUMNS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--img-size", type=int, default=64, help="0 or 64 = native, no resize")
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--blur-p", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--h5", default="/home/exouser/able_data/src_3dshapes/3dshapes.h5")
    ap.add_argument("--layout-dir", default="/home/exouser/able_data/shapes3d/graph_a/v1")
    ap.add_argument("--outdir", default=os.path.join(REPO, "experiments/hdae/outputs/attr_predictors_shapes3d"))
    ap.add_argument("--limit-train", type=int, default=0, help="cap training cells (calibration)")
    ap.add_argument("--limit-train-batches", type=int, default=0)
    ap.add_argument("--limit-eval-batches", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.0,
                    help="hard cap on this process's share of GPU memory; protects a concurrent run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    # A hard cap makes THIS process fail rather than starve a training run sharing the GPU.
    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, dev.index or 0)
    col = list(LAYOUT_COLUMNS).index(args.attr)
    n_classes = CARDINALITY[args.attr]
    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"{args.attr}{args.tag}"

    heldout = np.load(f"{args.layout_dir}/heldout_cells.npy").astype(np.int64)
    held_mask = np.zeros(N_GRID, dtype=bool)
    held_mask[heldout] = True
    trainable = np.flatnonzero(~held_mask)
    assert len(trainable) + len(heldout) == N_GRID, (len(trainable), len(heldout))

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(trainable))
    n_val = max(1, int(len(trainable) * args.val_frac))
    val_cells = trainable[perm[:n_val]]
    tr_cells = trainable[perm[n_val:]]
    if args.limit_train:
        tr_cells = tr_cells[:args.limit_train]
    assert not held_mask[tr_cells].any() and not held_mask[val_cells].any()

    t0 = time.time()
    with h5py.File(args.h5, "r") as f:
        images = f["images"][:]
    print(f"loaded {len(images):,} images ({images.nbytes/1e9:.1f} GB) in {time.time()-t0:.0f}s",
          flush=True)

    tr_ds = GridSubset(images, tr_cells)
    va_ds = GridSubset(images, val_cells)
    te_ds = GridSubset(images, heldout)

    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.workers, pin_memory=True, drop_last=sh,
                                   persistent_workers=args.workers > 0)
    tr, va, te = mk(tr_ds, True), mk(va_ds, False), mk(te_ds, False)

    model = Predictor(n_classes, args.dropout, not args.no_pretrained).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.limit_train_batches or len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=max(1, args.epochs * steps),
                                                pct_start=0.25)
    scaler = torch.cuda.amp.GradScaler()
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)

    print(f"attr={args.attr} col={col} classes={n_classes}  params={nparam/1e6:.1f}M  "
          f"img_size={args.img_size or 64} pretrained={not args.no_pretrained}", flush=True)
    print(f"train={len(tr_ds):,} val={len(va_ds):,} heldout_test={len(te_ds):,}", flush=True)

    best, best_ep = None, -1
    ckpt = os.path.join(args.outdir, f"{suffix}.pt")
    for ep in range(args.epochs):
        model.train()
        t1 = time.time()
        tot = n = 0
        for i, b in enumerate(tr):
            if args.limit_train_batches and i >= args.limit_train_batches:
                break
            img = label_preserving_aug(b["img"].to(dev, non_blocking=True),
                                       args.noise_std, args.blur_p, gen)
            x = prep(img, args.img_size)
            y = b["y"][:, col].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * img.shape[0]
            n += img.shape[0]
        m = evaluate(model, va, col, n_classes, args.img_size, dev, args.attr,
                     args.limit_eval_batches)
        star = ""
        if best is None or m["accuracy"] > best:
            best, best_ep = m["accuracy"], ep
            torch.save({"state_dict": model.state_dict(), "attr": args.attr, "col": col,
                        "n_classes": n_classes, "img_size": args.img_size,
                        "val_metrics": m, "epoch": ep, "args": vars(args)}, ckpt)
            star = "  *best"
        print(f"  ep {ep}  train_loss {tot/max(n,1):.5f}  val_acc {m['accuracy']:.5f}  "
              f"worst_class {m['worst_class_accuracy']}  peak_vram "
              f"{torch.cuda.max_memory_allocated()/2**30:.2f}GiB  ({time.time()-t1:.0f}s){star}",
              flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    test_m = evaluate(model, te, col, n_classes, args.img_size, dev, args.attr,
                      args.limit_eval_batches)
    print(f"BEST epoch {best_ep}  val_acc={best:.5f}")
    print(f"HELDOUT {json.dumps(test_m)}", flush=True)
    blob = torch.load(ckpt, map_location="cpu")
    blob["heldout_metrics"] = test_m
    torch.save(blob, ckpt)
    with open(os.path.join(args.outdir, f"{suffix}_metrics.json"), "w") as f:
        json.dump({"attr": args.attr, "n_classes": n_classes, "best_epoch": best_ep,
                   "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   "val": blob["val_metrics"], "heldout": test_m,
                   "args": vars(args)}, f, indent=2)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
