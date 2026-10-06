"""One multi-head ConvNeXt-Tiny reading EVERY unobserved CelebA attribute, for FC_unobs.

FC_unobs asks whether do(Bald) disturbs anything the model was never conditioned on. Sampling a
handful of attributes would answer a narrower question than the metric claims, so all 36 get a
head: 41 attributes in the pack, minus the four the model conditions on (Male, Young, Beard,
Bald), minus No_Beard, which is the raw CelebA column that Beard is derived from and would
otherwise score the target-adjacent attribute a second time with opposite sign.

One shared backbone with 36 heads costs one training run instead of 36.

Hair and facial-hair attributes are INCLUDED rather than dropped. Bald plausibly causes
Receding_Hairline, Gray_Hair, Bangs, Black_Hair and the rest, so their movement is not
necessarily collateral damage -- but hiding them would be worse than reporting them, so they are
measured and tagged, and the caller can read the hair subset separately from the rest.

Several attributes are weakly predictable at best (Blurry, Attractive, Oval_Face). Per-attribute
balanced accuracy is saved alongside the weights so a noisy reader can be identified rather than
silently diluting the mean.

Same split convention as train_celebahq_predictors.py: fit on partition 0, select on partition 1.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import sys
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.data.celeba_hq import CelebAHQPacked

OBSERVED = ["Male", "Young", "Beard", "Bald"]
REDUNDANT = ["No_Beard"]          # == -Beard
# Tagged, not excluded: plausibly caused by Bald, so reported as its own subset.
HAIR = ["Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Receding_Hairline",
        "Straight_Hair", "Wavy_Hair", "Wearing_Hat", "5_o_Clock_Shadow", "Goatee", "Mustache",
        "Sideburns"]
UNOBS = None                       # resolved from the pack at runtime
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Split(Dataset):
    def __init__(self, ds, idx, cols, train):
        self.ds, self.idx, self.cols, self.train = ds, idx, cols, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i]); r = self.ds[j]
        x = r["img"]
        if self.train and np.random.rand() < 0.5:
            x = torch.flip(x, dims=[-1])
        y = torch.from_numpy((self.ds.attrs[j, self.cols] == 1).astype("float32"))
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+", default=None)
    ap.add_argument("--lmdb", default="experiments/hdae/data/packed/celebahq_256.lmdb")
    ap.add_argument("--npz", default="experiments/hdae/data/packed/celebahq_256_attrs.npz")
    ap.add_argument("--out", default="experiments/hdae/outputs/attr_predictors_celebahq/_unobs.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    dev = "cuda"

    ds = CelebAHQPacked(a.lmdb, a.npz)
    names = list(ds.attribute_names)
    if a.attrs is None:
        excl = set(OBSERVED) | set(REDUNDANT)
        a.attrs = [n for n in names if n not in excl]
    cols = [names.index(x) for x in a.attrs]
    print(f"{len(a.attrs)} unobserved heads "
          f"({sum(n in HAIR for n in a.attrs)} hair/facial-hair, tagged not excluded)")
    tr_i = np.where(ds.partitions == 0)[0]
    va_i = np.where(ds.partitions == 1)[0]
    pos = [(ds.attrs[tr_i, c] == 1).mean() for c in cols]
    for n_, p in zip(a.attrs, pos):
        print(f"  {n_:22} {100*p:5.1f}% positive in train")

    tl = DataLoader(Split(ds, tr_i, cols, True), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(Split(ds, va_i, cols, False), batch_size=a.bs, num_workers=a.workers,
                    pin_memory=True, persistent_workers=True)

    m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(cols))
    m = m.to(dev)
    pw = torch.tensor([(1 - p) / max(p, 1e-6) for p in pos], device=dev, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(tl))
    scaler = torch.cuda.amp.GradScaler()
    MEANd, STDd = MEAN.to(dev), STD.to(dev)

    def prep(x):
        x = (x.to(dev, non_blocking=True) + 1) / 2
        if x.shape[-1] != a.size:
            x = torch.nn.functional.interpolate(x, size=(a.size,)*2, mode="bilinear", align_corners=False)
        return (x - MEANd) / STDd

    best = -1.0
    for ep in range(a.epochs):
        m.train(); t0 = time.time()
        for x, y in tl:
            with torch.autocast("cuda", dtype=torch.float16):
                l = lossf(m(prep(x)), y.to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward(); scaler.step(opt); scaler.update(); sched.step()
        m.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in vl:
                with torch.autocast("cuda", dtype=torch.float16):
                    P.append((m(prep(x)) > 0).float().cpu())
                Y.append(y)
        P = torch.cat(P); Y = torch.cat(Y)
        bal = []
        for k in range(len(cols)):
            p, yv = P[:, k], Y[:, k]
            tpr = float(p[yv == 1].mean()) if (yv == 1).any() else float("nan")
            tnr = float((1 - p[yv == 0]).mean()) if (yv == 0).any() else float("nan")
            bal.append((tpr + tnr) / 2)
        mb = float(np.nanmean(bal))
        worst = sorted(zip(bal, a.attrs))[:3]
        print(f"  epoch {ep}  mean balanced {mb:.4f}  ({time.time()-t0:.0f}s)  "
              f"weakest: " + ", ".join(f"{n_}={b:.3f}" for b, n_ in worst), flush=True)
        if mb > best:
            best = mb
            torch.save({"state_dict": m.state_dict(), "attrs": a.attrs, "img_size": a.size,
                        "balanced": {n_: float(b) for n_, b in zip(a.attrs, bal)},
                        "hair": [n for n in a.attrs if n in HAIR],
                        "mean_balanced": mb, "epoch": ep}, a.out)
            print(f"    saved -> {a.out}", flush=True)
    print("UNOBS DONE", flush=True)


if __name__ == "__main__":
    main()
