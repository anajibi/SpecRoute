"""XRV reading of saved counterfactual pixels, in a clean interpreter.

Run standalone: importing the repo puts diffae_upstream on sys.path, whose top-level `model`
package shadows an import inside torchxrayvision and breaks it with ModuleNotFoundError.
Nothing from the repo is imported here.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
import torchxrayvision as xrv
import pandas as pd

PX, DATA = sys.argv[1], "/home/exouser/able_data/chexpert"
XRV_PATH = list(xrv.models.DenseNet(weights="densenet121-res224-all").pathologies)
EI = XRV_PATH.index("Edema")
man = json.load(open(f"{PX}/manifest.json"))
fac = np.array(man["factual"]["edema"]); tgt = np.array(man["target"]["edema"])

def youden(y, p):
    o = np.argsort(-p); y = np.asarray(y)[o]; p = np.asarray(p)[o]
    P, N = y.sum(), len(y) - y.sum()
    j = np.cumsum(y)/max(P,1) - np.cumsum(1-y)/max(N,1)
    return float(p[int(j.argmax())])

valid = pd.read_csv(f"{DATA}/source_csv/valid.csv").set_index("Path")
rmeta = pd.read_csv(f"{DATA}/radiologist_val_meta.csv")
vxp = np.load(f"{DATA}/noise_audit/valid_xrv_probs.npy")
vpos = pd.Series(np.arange(len(valid)), index=valid.index.values)
sel = vpos.loc[rmeta["path"].values].to_numpy()
yrad = pd.to_numeric(valid.loc[rmeta["path"].values, "Edema"], errors="coerce").fillna(0).to_numpy()
thr = youden(yrad, vxp[sel][:, EI])

m = xrv.models.DenseNet(weights="densenet121-res224-all").cuda().eval()
read = {}
with torch.no_grad():
    for k in man["keys"]:
        a = np.load(f"{PX}/{k}.npy").astype(np.float32)
        t = torch.from_numpy(xrv.datasets.normalize(a, 255))[:, None]
        t = F.interpolate(t, size=(224,224), mode="bicubic", align_corners=False).cuda()
        read[k] = (m(t).cpu().numpy()[:, EI] >= thr).astype(int)

print(f"\nXRV densenet121-res224-all   step {man['global_step']}   thr {thr:.4f} (Youden, radiologist_val)")
hdr = f"{'g':>4s} | {'recon ok':>8s} | {'pos>neg':>9s} {'neg>pos':>9s} | {'real':>5s} {'free':>5s} {'wrong':>6s} {'hit':>6s}"
print(hdr); print('-'*len(hdr))
rec = read["recon"]; pos, neg = fac==1, fac==0
out={"thr":thr,"global_step":man["global_step"],"by_g":{}}
for k in [x for x in man["keys"] if x.startswith("g")]:
    cf = read[k]; ch = cf!=rec
    t = dict(recon_correct=int((rec==fac).sum()),
             pos_to_neg_ok=int((pos&ch&(cf==0)).sum()), neg_to_pos_ok=int((neg&ch&(cf==1)).sum()),
             real=int((ch&(cf==tgt)).sum()), free=int(((~ch)&(cf==tgt)).sum()),
             wrong=int((ch&(cf!=tgt)).sum()), hit=int((cf==tgt).sum()))
    out["by_g"][k]=t
    print(f"{k:>4s} | {t['recon_correct']:5d}/16 | {t['pos_to_neg_ok']:4d}/{int(pos.sum()):<4d} "
          f"{t['neg_to_pos_ok']:4d}/{int(neg.sum()):<4d} | {t['real']:5d} {t['free']:5d} "
          f"{t['wrong']:6d} {t['hit']:3d}/16")
json.dump(out, open(f"{PX}/xrv_score.json","w"), indent=2)
