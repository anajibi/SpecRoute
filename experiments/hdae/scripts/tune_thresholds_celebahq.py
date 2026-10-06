"""Tune one decision threshold per attribute on VAL, then score on TEST.

The readers are trained with pos_weight, which deliberately shifts the logits to stop a rare
attribute being solved by always answering "no". They are then read at a fixed `logit > 0`, as
if that shift had not happened. The operating point is therefore arbitrary, and the asymmetric
TPR/TNR pairs in the test evaluation are that mistake showing.

Logits are cached once and the sweep runs on CPU, so this costs one inference pass, not training.
Threshold is chosen on partition 1 (val) and applied unchanged to partition 2 (test) -- picking
it on test would be scoring the tuning set.
"""
import sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/exouser/SpecRoute")
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny
from experiments.hdae.data.celeba_hq import CelebAHQPacked

MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda()
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
P = "experiments/hdae/outputs/attr_predictors_celebahq"
ds = CelebAHQPacked("experiments/hdae/data/packed/celebahq_256.lmdb",
                    "experiments/hdae/data/packed/celebahq_256_attrs.npz")
names = list(ds.attribute_names)

class S(Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i): return ds[int(s.idx[i])]["img"]

def logits(m, idx, size, nout):
    dl = DataLoader(S(idx), batch_size=64, num_workers=6, pin_memory=True)
    out = []
    with torch.no_grad():
        for x in dl:
            x = (x.cuda(non_blocking=True) + 1) / 2
            if x.shape[-1] != size:
                x = torch.nn.functional.interpolate(x, size=(size,)*2, mode="bilinear", align_corners=False)
            with torch.autocast("cuda", dtype=torch.float16):
                o = m((x - MEAN) / STD).float()
            out.append((o if nout > 1 else o.squeeze(1)).cpu())
    return torch.cat(out)

def bal(pred, y):
    tpr = pred[y == 1].mean().item() if (y == 1).any() else np.nan
    tnr = (1 - pred[y == 0]).mean().item() if (y == 0).any() else np.nan
    return (tpr + tnr) / 2, tpr, tnr

va = np.where(ds.partitions == 1)[0]; te = np.where(ds.partitions == 2)[0]
res = {}

def handle(attrs, L_va, L_te, cols):
    for k, a in enumerate(attrs):
        lv = L_va[:, k] if L_va.dim() > 1 else L_va
        lt = L_te[:, k] if L_te.dim() > 1 else L_te
        yv = torch.from_numpy((ds.attrs[va, cols[k]] == 1).astype("float32"))
        yt = torch.from_numpy((ds.attrs[te, cols[k]] == 1).astype("float32"))
        b0, _, _ = bal((lt > 0).float(), yt)
        cand = torch.quantile(lv, torch.linspace(0.005, 0.995, 199))
        best_t, best_b = 0.0, -1
        for t in cand.tolist():
            b, _, _ = bal((lv > t).float(), yv)
            if b > best_b: best_b, best_t = b, t
        b1, tpr, tnr = bal((lt > best_t).float(), yt)
        res[a] = dict(thr=best_t, bal_before=b0, bal_after=b1, gain=b1 - b0, tpr=tpr, tnr=tnr)
        print(f"  {a:22} thr {best_t:+7.3f}   bal {b0:.3f} -> {b1:.3f}  ({b1-b0:+.3f})")

for a in ["Male", "Young", "Beard", "Bald"]:
    b = torch.load(f"{P}/{a}.pt", map_location="cpu")
    m = convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    m.load_state_dict(b["state_dict"]); m = m.cuda().eval()
    c = names.index(a)
    handle([a], logits(m, va, b["img_size"], 1), logits(m, te, b["img_size"], 1), [c])
    del m; torch.cuda.empty_cache()

b = torch.load(f"{P}/_unobs.pt", map_location="cpu")
m = convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(b["attrs"]))
m.load_state_dict(b["state_dict"]); m = m.cuda().eval()
cols = [names.index(a) for a in b["attrs"]]
handle(b["attrs"], logits(m, va, b["img_size"], len(cols)), logits(m, te, b["img_size"], len(cols)), cols)

g = [v["gain"] for v in res.values()]
print(f"\nmean balanced {np.mean([v['bal_before'] for v in res.values()]):.4f} -> "
      f"{np.mean([v['bal_after'] for v in res.values()]):.4f}   (mean gain {np.mean(g):+.4f})")
print(f"heads improved: {sum(x>0.001 for x in g)}/{len(g)}   worsened: {sum(x<-0.001 for x in g)}")
json.dump(res, open(f"{P}/thresholds.json","w"), indent=2)
print("wrote thresholds.json")
