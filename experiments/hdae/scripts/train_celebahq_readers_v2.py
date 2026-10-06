"""Stronger attribute readers, hardened with degradation augmentation.

WHY: the readers are trained on real CelebA-HQ but every number they produce is read off a
model-generated image. Measured on 2048 held-out reconstructions, 21 of 39 readers lose more
than 5 balanced-accuracy points, and the target attribute Bald loses 12 (0.966 -> 0.847).

The fix here is robustness, not domain training: nothing generated is ever put in the training
set. Instead the augmentation simulates what DDIM resynthesis does to an image -- it smooths
fine texture, shifts colour slightly, and softens edges -- so the reader stops depending on
high-frequency detail that survives in a photograph but not in a sample.

  resample      downsample and upsample again. The closest single proxy for generative
                smoothing, and the one that matters most.
  blur          Gaussian, random sigma.
  noise         mild additive Gaussian.
  jpeg-ish      quantisation of pixel values, standing in for compression artefacts.
  colour        brightness/contrast/saturation/hue jitter.
  geometry      RandomResizedCrop + horizontal flip + small rotation.

Also changed from v1: ConvNeXt-Small instead of Tiny, native 256 input instead of a 224
downsample (the fine-geometry attributes were losing detail there), cosine schedule, longer
training, and an EMA copy of the weights.

Fit on a 90% slice of partition 0 and select on the remaining 10%, so partitions 1 and 2 stay
completely clean for evaluation.
"""
import argparse, json, os, time, copy
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
import sys
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.data.celeba_hq import CelebAHQPacked

OBSERVED = ["Male", "Young", "Beard", "Bald"]
REDUNDANT = ["No_Beard"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def degrade(x):
    """Simulate generative resynthesis on a [0,1] tensor batch, per-sample."""
    B = x.shape[0]
    out = x
    # resample: the dominant effect of DDIM reconstruction
    m = torch.rand(B, device=x.device) < 0.5
    if m.any():
        s = int(np.random.choice([96, 128, 160, 192]))
        d = F.interpolate(out[m], size=(s, s), mode="bilinear", align_corners=False)
        out = out.clone()
        out[m] = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
    # blur
    m = torch.rand(B, device=x.device) < 0.35
    if m.any():
        k = int(np.random.choice([3, 5, 7]))
        sig = float(np.random.uniform(0.3, 1.8))
        out = out.clone()
        out[m] = v2.functional.gaussian_blur(out[m], [k, k], [sig, sig])
    # additive noise
    m = (torch.rand(B, device=x.device) < 0.3).view(B, 1, 1, 1)
    out = torch.where(m, (out + torch.randn_like(out) * np.random.uniform(0.01, 0.05)), out)
    # coarse quantisation, standing in for compression
    m = (torch.rand(B, device=x.device) < 0.25).view(B, 1, 1, 1)
    lv = float(np.random.choice([16, 24, 32, 48]))
    out = torch.where(m, (out * lv).round() / lv, out)
    return out.clamp(0, 1)


class Split(Dataset):
    def __init__(self, ds, idx, cols, train, size):
        self.ds, self.idx, self.cols, self.train = ds, idx, cols, train
        self.geo = v2.Compose([
            v2.RandomResizedCrop(size, scale=(0.72, 1.0), ratio=(0.9, 1.11), antialias=True),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomApply([v2.RandomRotation(8)], p=0.3),
        ]) if train else v2.Resize((size, size), antialias=True)
        self.col = v2.RandomApply(
            [v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03)], p=0.5)

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = (self.ds[j]["img"] + 1) / 2          # [0,1]
        x = self.geo(x)
        if self.train:
            x = self.col(x)
        y = torch.from_numpy((self.ds.attrs[j, self.cols] == 1).astype("float32"))
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_small")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-degrade", action="store_true")
    ap.add_argument("--attrs", nargs="+", default=None)
    ap.add_argument("--out", default="experiments/hdae/outputs/attr_predictors_celebahq/_readers_v2.pt")
    a = ap.parse_args()
    dev = "cuda"

    ds = CelebAHQPacked("experiments/hdae/data/packed/celebahq_256.lmdb",
                        "experiments/hdae/data/packed/celebahq_256_attrs.npz")
    names = list(ds.attribute_names)
    if a.attrs is None:
        a.attrs = [n for n in names if n not in set(REDUNDANT)]     # all 40, observed included
    cols = [names.index(x) for x in a.attrs]
    tr_all = np.where(ds.partitions == 0)[0]
    rng = np.random.default_rng(0); rng.shuffle(tr_all)
    cut = int(0.9 * len(tr_all))
    tr_i, va_i = np.sort(tr_all[:cut]), np.sort(tr_all[cut:])
    print(f"{len(a.attrs)} heads | fit {len(tr_i)} | select {len(va_i)} "
          f"| partitions 1,2 untouched | arch {a.arch} @ {a.size}px | degrade={not a.no_degrade}")

    pos = [(ds.attrs[tr_i, c] == 1).mean() for c in cols]
    tl = DataLoader(Split(ds, tr_i, cols, True, a.size), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(Split(ds, va_i, cols, False, a.size), batch_size=a.bs,
                    num_workers=a.workers, pin_memory=True, persistent_workers=True)

    m = getattr(models, a.arch)(weights="IMAGENET1K_V1")
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(cols))
    m = m.to(dev)
    ema = copy.deepcopy(m).eval()
    for p in ema.parameters(): p.requires_grad_(False)

    pw = torch.tensor([(1 - p) / max(p, 1e-6) for p in pos], device=dev, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(tl),
                                                pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler()
    MEANd, STDd = MEAN.to(dev), STD.to(dev)
    best = -1.0

    for ep in range(a.epochs):
        m.train(); t0 = time.time()
        for x, y in tl:
            x = x.to(dev, non_blocking=True)
            if not a.no_degrade:
                x = degrade(x)
            x = (x - MEANd) / STDd
            with torch.autocast("cuda", dtype=torch.float16):
                l = lossf(m(x), y.to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward(); scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                d = 0.999
                for pe, pm in zip(ema.parameters(), m.parameters()): pe.mul_(d).add_(pm, alpha=1-d)
                for be, bm in zip(ema.buffers(), m.buffers()): be.copy_(bm)
        # select on clean images -- the reader must stay good on real photos too
        ema.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in vl:
                x = ((x.to(dev, non_blocking=True)) - MEANd) / STDd
                with torch.autocast("cuda", dtype=torch.float16):
                    P.append((ema(x) > 0).float().cpu())
                Y.append(y)
        P, Y = torch.cat(P), torch.cat(Y)
        bal = []
        for k in range(len(cols)):
            p, yv = P[:, k], Y[:, k]
            tpr = float(p[yv == 1].mean()) if (yv == 1).any() else np.nan
            tnr = float((1 - p[yv == 0]).mean()) if (yv == 0).any() else np.nan
            bal.append((tpr + tnr) / 2)
        mb = float(np.nanmean(bal))
        worst = sorted([(b, n) for b, n in zip(bal, a.attrs) if not np.isnan(b)])[:3]
        print(f"  epoch {ep:2d}  mean balanced {mb:.4f}  ({time.time()-t0:.0f}s)  weakest: "
              + ", ".join(f"{n}={b:.3f}" for b, n in worst), flush=True)
        if mb > best:
            best = mb
            # fp16 weights: inference already runs under autocast fp16, so this costs
            # nothing and halves the 198 MB each ConvNeXt-Small otherwise occupies. 40
            # per-attribute models at fp32 is 7.9 GB, which does not fit on this disk.
            _sd = {k: (v.half() if v.is_floating_point() else v)
                   for k, v in ema.state_dict().items()}
            torch.save({"state_dict": _sd, "fp16": True, "attrs": a.attrs, "img_size": a.size,
                        "arch": a.arch, "balanced": {n: float(b) for n, b in zip(a.attrs, bal)},
                        "mean_balanced": mb, "epoch": ep, "degrade": not a.no_degrade}, a.out)
            print(f"    saved -> {a.out}", flush=True)
    print("READERS_V2 DONE", flush=True)


if __name__ == "__main__":
    main()
