"""All 40 readers x {real test, reconstructions} x {untuned thr=0, tuned on test}.

Logits are computed once per model and all four cells derived from them, so the only difference
between cells is the decision threshold and the image distribution.

The tuned threshold is chosen on the REAL TEST split to maximise balanced accuracy and then
applied unchanged to the reconstructions -- never selected on the reconstructions.
"""
import sys, json, glob, os, numpy as np, torch, torch.nn as nn
sys.path.insert(0,"/home/exouser/SpecRoute")
from torchvision import models
from experiments.hdae.data.celeba_hq import CelebAHQPacked
SP="/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
P="experiments/hdae/outputs/attr_predictors_celebahq"
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda(); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
ds=CelebAHQPacked("experiments/hdae/data/packed/celebahq_256.lmdb","experiments/hdae/data/packed/celebahq_256_attrs.npz")
names=list(ds.attribute_names); te=np.where(ds.partitions==2)[0]
cm=names.index("Male"); pool=np.where((ds.partitions>0)&(ds.attrs[:,cm]==1))[0]
cidx=np.sort(np.random.default_rng(7).choice(pool,2048,replace=False))
REC=torch.load(f"{SP}/xt_cache_2048.pt")["rec"]
real=torch.stack([ds[int(i)]["img"] for i in te])
OBS={"Male","Young","Beard","Bald"}

def infer(m,size,src,n):
    o=[]
    with torch.no_grad():
        for lo in range(0,n,64):
            x=src(lo,min(lo+64,n)).cuda().float(); x=(x+1)/2
            if x.shape[-1]!=size: x=torch.nn.functional.interpolate(x,size=(size,)*2,mode="bilinear",align_corners=False)
            with torch.autocast("cuda",dtype=torch.float16): o.append(m((x-MEAN)/STD).float().cpu())
    return torch.cat(o).squeeze(1)

def st(L,t,y):
    if (y==1).sum()<10 or (y==0).sum()<10: return None
    p=(L>t).float()
    tpr=float(p[y==1].mean()); tnr=float((1-p[y==0]).mean())
    tp=float(((p==1)&(y==1)).sum()); fp=float(((p==1)&(y==0)).sum()); fn=float(((p==0)&(y==1)).sum())
    pr=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.
    return dict(bal=(tpr+tnr)/2, f1=(2*pr*rc/(pr+rc) if pr+rc else 0.),
                prec=pr, rec=rc, pred_rate=float(p.mean()), base=float(y.mean()))

out={}
for f in sorted(glob.glob(f"{P}/single/*.pt")):
    a=os.path.basename(f)[:-3]
    b=torch.load(f,map_location="cpu")
    m=getattr(models,b["arch"])(); m.classifier[2]=nn.Linear(m.classifier[2].in_features,1)
    m.load_state_dict({k:v.float() for k,v in b["state_dict"].items()}); m=m.cuda().eval()
    c=names.index(a)
    yr=torch.from_numpy((ds.attrs[te,c]==1).astype("float32"))
    yc=torch.from_numpy((ds.attrs[cidx,c]==1).astype("float32"))
    Lr=infer(m,b["img_size"],lambda i,j: real[i:j],len(te))
    Lc=infer(m,b["img_size"],lambda i,j: REC[i:j],2048)
    q=torch.quantile(Lr,torch.linspace(0.005,0.995,199)).tolist()
    best=max(q,key=lambda t:(st(Lr,t,yr) or {"bal":-1})["bal"])
    out[a]=dict(thr=float(best), observed=a in OBS, n_pos_recon=int(yc.sum()), n_pos_test=int(yr.sum()),
                test_untuned=st(Lr,0.0,yr), test_tuned=st(Lr,best,yr),
                recon_untuned=st(Lc,0.0,yc), recon_tuned=st(Lc,best,yc))
    del m; torch.cuda.empty_cache(); print(".",end="",flush=True)
json.dump(out,open(f"{P}/full_eval.json","w"),indent=2)
print(f"\nFULL_EVAL DONE ({len(out)})")
