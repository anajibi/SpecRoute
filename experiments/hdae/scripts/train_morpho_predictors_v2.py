"""Retrain the MorphoMNIST attribute predictors with the recipe that works.

WHY. The shipped predictors are a 4-layer SmallCNN (~1.2M params, 5 MB) trained from scratch,
and they are not good enough to measure counterfactuals with. On the held-out split:

    digit          90.52% accuracy      (Causal3DIdent's class predictor: 99.996%)
    thickness      MAE 0.1182 = 39.2% of the attribute's own spread
    intensity      MAE 4.2251 = 21.2% of spread
    bg_amplitude   MAE 4.4652 = 98.9% of spread -- no better than answering the mean

An instrument that consumes 39% of an attribute's dynamic range cannot resolve a counterfactual
edit in it. Causal3DIdent's predictors sit at 2-4.5% of spread on the same measure, on a HARDER
128x128 dataset, using ImageNet-pretrained ConvNeXt-Tiny. This script ports that recipe.

THREE CHANGES FROM THE OLD TRAINER:

  1. ConvNeXt-Tiny, ImageNet-pretrained, instead of a from-scratch SmallCNN. 28M params against
     1.2M, and the pretrained stem matters more than the size: these images are textured, coloured
     and cluttered, which is much closer to ImageNet than to plain MNIST.

  2. Inputs upsampled 64 -> 128. ConvNeXt downsamples by 32, so at 64x64 the final feature map is
     2x2 -- almost no spatial resolution left for attributes that ARE spatial (translate_x/y,
     rotation, scale). At 128 it is 4x4. This is the same size Causal3DIdent used.

  3. Per-attribute target normalisation to [-1, 1]. The old trainer regressed RAW targets, so
     intensity (range ~166) and bg_phase (range ~6.3) were fed to the same loss and the same
     learning rate. A single lr cannot suit both. Targets are now scaled by train-split min/max
     and the prediction is mapped back before any metric is computed, so reported MAE stays in
     original units and is directly comparable to the old numbers.

Augmentation stays OFF for the geometric attributes: random affine would destroy exactly the
signal translate/rotation/scale predictors need. Only mild noise/blur is used, as on C3DI.

    python experiments/hdae/scripts/train_morpho_predictors_v2.py --attr digit
    python experiments/hdae/scripts/train_morpho_predictors_v2.py --all
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# kind: "categorical" (n classes) or "scalar". digit and hue are the two the causal graph
# declares categorical; everything else is regressed, including the ordinal grid factors
# (rotation, translate_x/y) -- a graded error is more useful than a 20-way class label.
ATTRS = {
    "digit":             ("categorical", 10),
    "hue":               ("categorical", 10),
    "thickness":         ("scalar", 1),
    "intensity":         ("scalar", 1),
    "rotation":          ("scalar", 1),
    "scale":             ("scalar", 1),
    "translate_x":       ("scalar", 1),
    "translate_y":       ("scalar", 1),
    "bg_freq":           ("scalar", 1),
    "bg_phase":          ("scalar", 1),
    "bg_amplitude":      ("scalar", 1),
    "texture_amplitude": ("scalar", 1),
}


class H5Attrs(Dataset):
    def __init__(self, path, idx, col, kind, lo=None, hi=None, n_classes=None, preload=True):
        self.path, self.idx, self.col, self.kind = path, idx, col, kind
        self.lo, self.hi, self.n_classes = lo, hi, n_classes
        self._h5 = None
        with h5py.File(path, "r") as f:
            self.attrs = f["attrs"][:].astype(np.float32)
            # The whole image array is 70000*64*64*3 = 860 MB as uint8 -- small enough to hold in
            # RAM, which takes the random HDF5 read out of the per-sample hot path entirely.
            # Measured before this change: 6 concurrent jobs held the A100 at ~61% mean
            # utilisation and 179 W of a ~400 W TDP, dipping to 26% while waiting on data.
            # DataLoader workers are forked after this array exists, so it is shared
            # copy-on-write and does NOT cost 860 MB per worker.
            self.images = f["images"][:] if preload else None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        if self.images is not None:
            raw = self.images[j]
        else:
            if self._h5 is None:
                self._h5 = h5py.File(self.path, "r")
            raw = self._h5["images"][j]
        img = torch.from_numpy(raw.astype(np.float32) / 255.).permute(2, 0, 1)
        v = float(self.attrs[j, self.col])
        if self.kind == "categorical":
            # hue is stored as its bin CENTRE (0.05, 0.15, ...), not a class index
            t = torch.tensor([round((v - self.lo) / (self.hi - self.lo) * self.n_classes - 0.5)
                              if self.n_classes and self.hi > 1.5 else v], dtype=torch.float32)
            t = t.clamp(0, (self.n_classes or 1) - 1)
        else:
            t = torch.tensor([2 * (v - self.lo) / (self.hi - self.lo) - 1], dtype=torch.float32)
        return img, t


class Net(nn.Module):
    def __init__(self, out_dim, dropout=0.2, pretrained=True):
        super().__init__()
        self.net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        f = self.net.classifier[2].in_features
        self.net.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(f, out_dim))

    def forward(self, x):
        return self.net(x)


def prep(x, size, noise=0.0):
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    if noise:
        x = x + torch.randn_like(x) * noise
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


@torch.no_grad()
def evaluate(model, loader, kind, lo, hi, size, dev):
    model.eval(); P, T = [], []
    for img, t in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(prep(img.to(dev), size))
        P.append(p.float().cpu()); T.append(t)
    P, T = torch.cat(P), torch.cat(T)
    if kind == "categorical":
        acc = (P.argmax(1) == T[:, 0].long()).float().mean().item()
        top2 = (P.topk(2, 1).indices == T[:, 0:1].long()).any(1).float().mean().item()
        return {"accuracy": acc, "top2_accuracy": top2}
    # back to original units before measuring
    pu = (P.clamp(-1, 1) + 1) / 2 * (hi - lo) + lo
    tu = (T + 1) / 2 * (hi - lo) + lo
    return {"mae": (pu - tu).abs().mean().item(),
            "r2": 1 - ((pu - tu) ** 2).sum().item() / max(((tu - tu.mean()) ** 2).sum().item(), 1e-9)}


def train_one(attr, args):
    kind, out_dim = ATTRS[attr]
    dev = torch.device(args.device)
    with h5py.File(args.packed, "r") as f:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in f["attribute_names"][:]]
        A = f["attrs"][:].astype(np.float64)
        part = f["partitions"][:]
    col = names.index(attr)
    tr_all = np.where(part == 0)[0]
    rng = np.random.RandomState(args.seed); rng.shuffle(tr_all)
    nv = int(len(tr_all) * args.val_frac)
    va_idx, tr_idx = tr_all[:nv], tr_all[nv:]
    te_idx = np.where(part == 1)[0]
    lo, hi = float(A[tr_idx, col].min()), float(A[tr_idx, col].max())
    n_cls = out_dim if kind == "categorical" else None

    mk = lambda idx, sh: DataLoader(H5Attrs(args.packed, idx, col, kind, lo, hi, n_cls),
                                    batch_size=args.batch_size, shuffle=sh,
                                    num_workers=args.workers, pin_memory=True, drop_last=sh,
                                    persistent_workers=args.workers > 0)
    tr, va, te = mk(tr_idx, True), mk(va_idx, False), mk(te_idx, False)
    model = Net(out_dim, args.dropout, not args.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * len(tr), pct_start=args.pct_start)
    scaler = torch.cuda.amp.GradScaler()
    key = "accuracy" if kind == "categorical" else "mae"
    best = -1e9 if kind == "categorical" else 1e9
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for img, t in tr:
            img, t = img.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                p = model(prep(img, args.img_size, args.noise_std))
                loss = F.cross_entropy(p, t[:, 0].long()) if kind == "categorical" else F.smooth_l1_loss(p, t)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * img.shape[0]; n += img.shape[0]
        m = evaluate(model, va, kind, lo, hi, args.img_size, dev)
        cur = m[key]
        better = cur > best if kind == "categorical" else cur < best
        if better:
            best = cur; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{attr}] ep {ep}  loss {tot/max(n,1):.5f}  val_{key} {cur:.5f}"
              f"{'  *' if better else ''}", flush=True)
    model.load_state_dict(best_state)
    test = evaluate(model, te, kind, lo, hi, args.img_size, dev)
    spread = float(np.abs(A[te_idx, col] - A[te_idx, col].mean()).mean())
    test["spread"] = spread
    if kind != "categorical":
        test["mae_pct_of_spread"] = test["mae"] / spread * 100
    os.makedirs(args.outdir, exist_ok=True)
    torch.save({"state_dict": best_state, "out_dim": out_dim, "kind": kind,
                "lo": lo, "hi": hi, "attr": attr, "img_size": args.img_size,
                "test_metrics": test}, os.path.join(args.outdir, f"{attr}.pt"))
    json.dump({"attr": attr, "kind": kind, "lo": lo, "hi": hi, "test": test,
               "epochs": args.epochs, "minutes": (time.time() - t0) / 60},
              open(os.path.join(args.outdir, f"{attr}_metrics.json"), "w"), indent=2)
    s = (f"acc {test['accuracy']*100:.2f}%  top2 {test['top2_accuracy']*100:.2f}%"
         if kind == "categorical" else
         f"MAE {test['mae']:.4f}  ({test['mae_pct_of_spread']:.1f}% of spread)  R2 {test['r2']:.4f}")
    print(f"[{attr}] TEST  {s}   [{(time.time()-t0)/60:.1f} min]", flush=True)
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default=None, choices=list(ATTRS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--packed", default=os.path.join(REPO, "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5"))
    ap.add_argument("--outdir", default=os.path.join(REPO, "experiments/hdae/outputs/attr_predictors_morpho_v2"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pct-start", type=float, default=0.25,
                    help="OneCycle warmup fraction; lower = longer anneal, steadier late epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    todo = list(ATTRS) if args.all else [args.attr]
    if todo == [None]:
        raise SystemExit("pass --attr <name> or --all")
    out = {}
    for a in todo:
        out[a] = train_one(a, args)
    print("\n=== summary")
    for a, m in out.items():
        print(f"  {a:18s} " + (f"acc {m['accuracy']*100:.2f}%" if "accuracy" in m
                               else f"MAE {m['mae']:.4f}  {m['mae_pct_of_spread']:.1f}% of spread"))


if __name__ == "__main__":
    main()
