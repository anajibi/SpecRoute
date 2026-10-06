"""Held-out evaluation of every attribute predictor, on the TEST split (partition 2).

The saved summaries report validation numbers, and validation is what epoch selection used, so
those are optimistic by construction. Partition 2 was never touched by predictor training or
selection.

Reported per attribute: base rate, TPR, TNR, balanced accuracy, and the count of positives the
estimate rests on. Balanced accuracy is the figure that matters -- raw accuracy on a 5%-positive
attribute is meaningless.
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
te = np.where(ds.partitions == 2)[0]

class S(Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i): return ds[int(s.idx[i])]["img"]

dl = DataLoader(S(te), batch_size=64, num_workers=6, pin_memory=True)

def prep(x, size):
    x = (x.cuda(non_blocking=True) + 1) / 2
    if x.shape[-1] != size:
        x = torch.nn.functional.interpolate(x, size=(size,)*2, mode="bilinear", align_corners=False)
    return (x - MEAN) / STD

def stats(pred, y):
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float((1 - pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return tpr, tnr, (tpr + tnr) / 2, float((pred == y).float().mean())

out = {}
rows = []

# --- the four observed, single-head ---------------------------------------------------------
for a in ["Male", "Young", "Beard", "Bald"]:
    b = torch.load(f"{P}/{a}.pt", map_location="cpu")
    m = convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    m.load_state_dict(b["state_dict"]); m = m.cuda().eval()
    col = names.index(a); y = torch.from_numpy((ds.attrs[te, col] == 1).astype("float32"))
    pr = []
    with torch.no_grad():
        for x in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                pr.append((m(prep(x, b["img_size"])).squeeze(1) > 0).float().cpu())
    pr = torch.cat(pr)
    tpr, tnr, bal, acc = stats(pr, y)
    rows.append(("OBS", a, float(y.mean()), int(y.sum()), tpr, tnr, bal, acc))
    out[a] = dict(group="observed", base=float(y.mean()), n_pos=int(y.sum()),
                  tpr=tpr, tnr=tnr, balanced=bal, acc=acc)
    del m; torch.cuda.empty_cache()

# --- the 36 unobserved, multi-head ----------------------------------------------------------
b = torch.load(f"{P}/_unobs.pt", map_location="cpu")
attrs = b["attrs"]; hair = set(b.get("hair", []))
m = convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(attrs))
m.load_state_dict(b["state_dict"]); m = m.cuda().eval()
cols = [names.index(a) for a in attrs]
Y = torch.from_numpy((ds.attrs[np.ix_(te, cols)] == 1).astype("float32"))
pr = []
with torch.no_grad():
    for x in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            pr.append((m(prep(x, b["img_size"])) > 0).float().cpu())
pr = torch.cat(pr)
for k, a in enumerate(attrs):
    tpr, tnr, bal, acc = stats(pr[:, k], Y[:, k])
    grp = "UNOBS-hair" if a in hair else "UNOBS"
    rows.append((grp, a, float(Y[:, k].mean()), int(Y[:, k].sum()), tpr, tnr, bal, acc))
    out[a] = dict(group=grp.lower(), base=float(Y[:, k].mean()), n_pos=int(Y[:, k].sum()),
                  tpr=tpr, tnr=tnr, balanced=bal, acc=acc)

rows.sort(key=lambda r: r[6])
print(f"{'group':11} {'attribute':22} {'base':>6} {'n_pos':>6} {'TPR':>6} {'TNR':>6} {'BAL':>6} {'acc':>6}  flag")
for g, a, base, npos, tpr, tnr, bal, acc in rows:
    flag = ""
    if bal < 0.70: flag = "WEAK"
    elif npos < 60: flag = "few-pos"
    print(f"{g:11} {a:22} {base:6.3f} {npos:6d} {tpr:6.3f} {tnr:6.3f} {bal:6.3f} {acc:6.3f}  {flag}")

un = [r for r in rows if r[0].startswith("UNOBS")]
rel = [r for r in un if r[6] >= 0.75 and r[3] >= 60]
print(f"\nunobserved heads: {len(un)}   mean balanced {np.mean([r[6] for r in un]):.4f}")
print(f"reliable subset (BAL>=0.75 and n_pos>=60): {len(rel)}   "
      f"mean balanced {np.mean([r[6] for r in rel]):.4f}")
print("excluded from reliable subset: " + ", ".join(r[1] for r in un if r not in rel))
json.dump(out, open("experiments/hdae/outputs/attr_predictors_celebahq/test_eval.json","w"), indent=2)
print("\nwrote test_eval.json")
