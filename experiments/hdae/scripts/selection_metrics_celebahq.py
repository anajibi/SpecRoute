"""Attribute-selection metrics on the partition-0 held-out slice -- NOT the cohort.

The 2,400-image slice is the 10% of partition 0 reserved for epoch selection. It is the only
data that is neither used for fitting the readers' weights nor part of the 2,048-subject
evaluation cohort (which draws from partitions 1 and 2). Choosing which attributes to drop from
cohort numbers would be selecting readers on the evaluation set.

Threshold is the untuned logit > 0 throughout: every tuned threshold in this project was fitted
on partition 1 or 2, both of which feed the cohort.
"""
import glob, json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0,"/home/exouser/SpecRoute")
from torchvision import models
from experiments.hdae.data.celeba_hq import CelebAHQPacked
V2="experiments/hdae/outputs/attr_predictors_celebahq/single"
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda(); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
ds=CelebAHQPacked("experiments/hdae/data/packed/celebahq_256.lmdb","experiments/hdae/data/packed/celebahq_256_attrs.npz")
names=list(ds.attribute_names)
tr_all=np.where(ds.partitions==0)[0]
rng=np.random.default_rng(0); rng.shuffle(tr_all)          # identical shuffle to the trainer
sel=np.sort(tr_all[int(0.9*len(tr_all)):])                  # the 10% selection slice
print(f"selection slice: {len(sel)} images from partition 0 (never in the cohort)")
IM=torch.stack([ds[int(i)]["img"] for i in sel])
def lg(path,imgs):
    b=torch.load(path,map_location="cpu")
    m=getattr(models,b.get("arch","convnext_small"))(); m.classifier[2]=nn.Linear(m.classifier[2].in_features,1)
    m.load_state_dict({k:v.float() for k,v in b["state_dict"].items()}); m=m.cuda().eval()
    o=[]
    with torch.no_grad():
        for lo in range(0,len(imgs),64):
            x=imgs[lo:lo+64].cuda().float(); x=(x+1)/2
            x=torch.nn.functional.interpolate(x,size=(b["img_size"],)*2,mode="bilinear",align_corners=False)
            with torch.autocast("cuda",dtype=torch.float16): o.append(m((x-MEAN)/STD).squeeze(1).float().cpu())
    del m; torch.cuda.empty_cache(); return torch.cat(o)
def met(l,y):
    p=(l>0).float()
    tp=float(((p==1)&(y==1)).sum()); fp=float(((p==1)&(y==0)).sum())
    fn=float(((p==0)&(y==1)).sum()); tn=float(((p==0)&(y==0)).sum())
    pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
    tnr=tn/(tn+fp) if tn+fp else 0.0
    return dict(acc=(tp+tn)/len(y), bal=(rc+tnr)/2, prec=pr, rec=rc,
                f1=2*pr*rc/(pr+rc) if pr+rc else 0.0,
                pred_rate=float(p.mean()), base=float(y.mean()), n_pos=int(y.sum()))
out={}
for f in sorted(glob.glob(f"{V2}/*.pt")):
    a=os.path.basename(f)[:-3]; c=names.index(a)
    y=torch.from_numpy((ds.attrs[sel,c]==1).astype("float32"))
    out[a]=met(lg(f,IM),y); print(".",end="",flush=True)
print()
json.dump(out,open("experiments/hdae/outputs/attr_predictors_celebahq/selection_metrics.json","w"),indent=2)
print("wrote selection_metrics.json")
