"""Binary attribute predictors for CelebA-HQ 256 -- the blocker for every counterfactual metric.

CC / FC / CF1 all compare a predictor's reading of an edited image against the requested target.
Without predictors there is no way to choose a guidance strength on anything but eyeballing, and
no way to compare depths numerically. This trains one ConvNeXt-Tiny head per attribute.

Two choices carried from the MorphoMNIST predictor work, both of which were bugs there first:

  * Accuracy alone is NOT a sufficient check. A hue predictor once reported 100% while always
    answering class 0, because the labels were truncated the same way the predictions were. Here
    `Bald` is 2.4% positive, so a model that always answers "not bald" scores 97.6%. Balanced
    accuracy and the prediction histogram are printed, and a degenerate head is called out.
  * Class imbalance is handled with pos_weight rather than ignored, for the same reason.

Images are read at 256 and downscaled to 224 for the ImageNet-pretrained stem.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.data.celeba_hq import CelebAHQPacked

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Split(Dataset):
    def __init__(self, ds, idx, col, train):
        self.ds, self.idx, self.col, self.train = ds, idx, col, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.ds[int(self.idx[i])]
        img = r["img"]
        if self.train and torch.rand(()) < 0.5:
            img = img.flip(-1)
        # CelebA codes {-1,+1}; BCEWithLogits wants {0,1}
        return img, torch.tensor(float(r["attr"][self.col] > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="+", default=["Male", "Young", "Beard", "Bald"])
    ap.add_argument("--lmdb", default="experiments/hdae/data/packed/celebahq_256.lmdb")
    ap.add_argument("--npz", default="experiments/hdae/data/packed/celebahq_256_attrs.npz")
    ap.add_argument("--outdir", default="experiments/hdae/outputs/attr_predictors_celebahq")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = "cuda"

    ds = CelebAHQPacked(a.lmdb, a.npz)
    names = ds.attribute_names
    tr_i = np.where(ds.partitions == 0)[0]
    va_i = np.where(ds.partitions == 1)[0]
    summary = {}

    for attr in a.attrs:
        col = names.index(attr)
        pos = float((ds.attrs[tr_i, col] == 1).mean())
        print(f"\n=== {attr}  ({100*pos:.1f}% positive in train, "
              f"{int((ds.attrs[tr_i,col]==1).sum())} positives) ===", flush=True)
        tl = DataLoader(Split(ds, tr_i, col, True), batch_size=a.bs, shuffle=True,
                        num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
        vl = DataLoader(Split(ds, va_i, col, False), batch_size=a.bs, num_workers=a.workers,
                        pin_memory=True, persistent_workers=True)
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        m = m.to(dev)
        # pos_weight so a 2.4%-positive attribute cannot be solved by always predicting "no"
        pw = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(tl))
        scaler = torch.cuda.amp.GradScaler()
        best = None

        def prep(x):
            x = (x.to(dev, non_blocking=True) + 1) / 2
            if x.shape[-1] != a.size:
                x = torch.nn.functional.interpolate(x, size=(a.size,)*2, mode="bilinear", align_corners=False)
            return (x - MEAN.to(dev)) / STD.to(dev)

        for ep in range(a.epochs):
            m.train(); t0 = time.time()
            for x, y in tl:
                with torch.autocast("cuda", dtype=torch.float16):
                    l = lossf(m(prep(x)).squeeze(1), y.to(dev))
                opt.zero_grad(set_to_none=True)
                scaler.scale(l).backward(); scaler.step(opt); scaler.update(); sched.step()
            m.eval(); P, Y = [], []
            with torch.no_grad():
                for x, y in vl:
                    with torch.autocast("cuda", dtype=torch.float16):
                        P.append((m(prep(x)).squeeze(1) > 0).float().cpu())
                    Y.append(y)
            P, Y = torch.cat(P), torch.cat(Y)
            acc = float((P == Y).float().mean())
            tpr = float(P[Y == 1].mean()) if (Y == 1).any() else float("nan")
            tnr = float((1 - P[Y == 0]).mean()) if (Y == 0).any() else float("nan")
            bal = (tpr + tnr) / 2
            frac_pos = float(P.mean())
            print(f"  ep{ep+1} acc {acc:.4f}  balanced {bal:.4f}  TPR {tpr:.4f} TNR {tnr:.4f}  "
                  f"predicted-positive {100*frac_pos:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
            if best is None or bal > best["balanced"]:
                best = dict(epoch=ep+1, acc=acc, balanced=bal, tpr=tpr, tnr=tnr, frac_pos=frac_pos)
                torch.save({"state_dict": m.state_dict(), "attr": attr, "img_size": a.size,
                            "pos_rate": pos}, f"{a.outdir}/{attr}.pt")
        # a head that never fires (or always fires) is useless no matter what accuracy says
        degenerate = best["frac_pos"] < 0.01 or best["frac_pos"] > 0.99 or best["balanced"] < 0.6
        best["degenerate"] = bool(degenerate)
        print(f"  BEST balanced {best['balanced']:.4f} (acc {best['acc']:.4f})"
              + ("   *** DEGENERATE -- unusable for CC/FC ***" if degenerate else "   OK"), flush=True)
        summary[attr] = best
        del m, opt, tl, vl; torch.cuda.empty_cache()
    json.dump(summary, open(f"{a.outdir}/summary.json", "w"), indent=2)
    print(f"\nwrote {a.outdir}/summary.json")


if __name__ == "__main__":
    main()
