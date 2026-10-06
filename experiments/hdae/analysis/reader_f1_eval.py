"""Precision / recall / F1 per attribute at the TUNED thresholds, on the test split.

Balanced accuracy can look acceptable while F1 is poor: on a rare attribute a reader can catch
most positives (high TPR) and still be swamped by false positives (low precision). F1 is the
figure that exposes that, and a reader with low F1 flips its answer on noise, which shows up in
FC as drift the generative model never caused.
"""
import sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/exouser/SpecRoute")
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny
from experiments.hdae.data.celeba_hq import CelebAHQPacked

MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda()
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
P = "experiments/hdae/outputs/attr_predictors_celebahq"
THR = json.load(open(f"{P}/thresholds.json")); EV = json.load(open(f"{P}/test_eval.json"))
ds = CelebAHQPacked("experiments/hdae/data/packed/celebahq_256.lmdb",
                    "experiments/hdae/data/packed/celebahq_256_attrs.npz")
names = list(ds.attribute_names); te = np.where(ds.partitions == 2)[0]

class S(Dataset):
    def __init__(s, i): s.i = i
    def __len__(s): return len(s.i)
    def __getitem__(s, k): return ds[int(s.i[k])]["img"]
dl = DataLoader(S(te), batch_size=64, num_workers=6, pin_memory=True)

def run(m, size, nout):
    o = []
    with torch.no_grad():
        for x in dl:
            x = (x.cuda(non_blocking=True) + 1) / 2
            if x.shape[-1] != size:
                x = torch.nn.functional.interpolate(x, size=(size,)*2, mode="bilinear", align_corners=False)
            with torch.autocast("cuda", dtype=torch.float16):
                r = m((x - MEAN) / STD).float()
            o.append((r if nout > 1 else r.squeeze(1)).cpu())
    return torch.cat(o)

out = {}
def score(a, lg, col):
    y = torch.from_numpy((ds.attrs[te, col] == 1).astype("float32"))
    p = (lg > THR[a]["thr"]).float()
    tp = float(((p == 1) & (y == 1)).sum()); fp = float(((p == 1) & (y == 0)).sum())
    fn = float(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out[a] = dict(prec=prec, rec=rec, f1=f1, bal=THR[a]["bal_after"],
                  n_pos=EV[a]["n_pos"], base=EV[a]["base"], group=EV[a]["group"])

for a in ["Male", "Young", "Beard", "Bald"]:
    b = torch.load(f"{P}/{a}.pt", map_location="cpu")
    m = convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    m.load_state_dict(b["state_dict"]); m = m.cuda().eval()
    score(a, run(m, b["img_size"], 1), names.index(a)); del m; torch.cuda.empty_cache()

b = torch.load(f"{P}/_unobs.pt", map_location="cpu")
m = convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(b["attrs"]))
m.load_state_dict(b["state_dict"]); m = m.cuda().eval()
L = run(m, b["img_size"], len(b["attrs"]))
for k, a in enumerate(b["attrs"]): score(a, L[:, k], names.index(a))

json.dump(out, open(f"{P}/f1_eval.json", "w"), indent=2)
rows = sorted(out.items(), key=lambda x: x[1]["f1"])
print(f"{'attribute':22} {'grp':11} {'base':>6} {'n_pos':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'BAL':>6}")
for a, v in rows:
    print(f"{a:22} {v['group'][:10]:11} {v['base']:6.3f} {v['n_pos']:6d} "
          f"{v['prec']:6.3f} {v['rec']:6.3f} {v['f1']:6.3f} {v['bal']:6.3f}")
