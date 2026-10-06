"""Quantify the 6 configurations: does do(Bald) drag Beard and Young along?

For each {conditional, unconditional} x_T x {null 1, 2, 4 attrs} and each g, the four
attribute predictors read the output. Reported per config:
  CC_Bald   fraction reading BALD        -- did the requested edit land
  dBeard    change in fraction reading BEARD vs the reconstruction at g=1
  dYoung    change in fraction reading YOUNG
  dMale     change in fraction reading MALE
Drift is measured against each variant's own reconstruction, not the source photo, so
inversion error is not charged to the edit.
"""
import sys, glob, json, torch, numpy as np, torch.nn as nn
sys.path.insert(0,"/home/exouser/SpecRoute/diffae_upstream"); sys.path.insert(0,"/home/exouser/SpecRoute")
from torchvision.models import convnext_tiny
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values

SP="/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
ATTRS=["Male","Young","Beard","Bald"]; TARGET=3; N=32; T=50; GS=[1,2,3,5]
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda()
STD =torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()

class Reader:
    def __init__(self,a):
        b=torch.load(f"experiments/hdae/outputs/attr_predictors_celebahq/{a}.pt",map_location="cpu")
        m=convnext_tiny(); m.classifier[2]=nn.Linear(m.classifier[2].in_features,1)
        m.load_state_dict(b["state_dict"]); self.net=m.cuda().eval(); self.size=b["img_size"]
    @torch.no_grad()
    def __call__(self,img,bs=32):
        o=[]
        for i in range(0,len(img),bs):
            x=(img[i:i+bs].cuda()+1)/2
            if x.shape[-1]!=self.size:
                x=torch.nn.functional.interpolate(x,size=(self.size,)*2,mode="bilinear",align_corners=False)
            x=(x-MEAN)/STD
            with torch.autocast("cuda",dtype=torch.float16):
                o.append((self.net(x).squeeze(1)>0).float().cpu())
        return torch.cat(o)

class MultiNullCFG(torch.nn.Module):
    def __init__(self,base,g,idx):
        super().__init__(); self.base=base; self.g=float(g); self.idx=list(idx)
    def forward(self,x,t,cond,**kw):
        co=self.base.forward(x=x,t=t,cond=cond,**kw)
        if self.g==1.0: return co
        nm=torch.zeros_like(cond["y_idx"],dtype=torch.bool); nm[:,self.idx]=True
        uo=self.base.forward(x=x,t=t,cond={"zs":cond["zs"],"y_idx":cond["y_idx"],"null_mask":nm},**kw)
        return co.__class__(pred=uo.pred+self.g*(co.pred-uo.pred),cond=cond)

cfg=load_hdae_config("experiments/hdae/configs/celebahq256_k11_cd015r.yaml")
ck=sorted(glob.glob("experiments/hdae/outputs/celebahq256_k11_cd015r/checkpoints/*.ckpt"))
net=HDAELitModule.load_from_checkpoint(ck[-1],conf=cfg.train_conf,map_location="cpu").ema_model.cuda().eval()
specs=cfg.hdae_conf.encoder.cond_specs
R={a:Reader(a) for a in ATTRS}

D="experiments/hdae/data/packed"
ds=CelebAHQPacked(f"{D}/celebahq_256.lmdb",f"{D}/celebahq_256_attrs.npz")
nm_=ds.attribute_names; cols=[nm_.index(a) for a in ATTRS]
test=np.where(ds.partitions==2)[0]
pool=test[(ds.attrs[test][:,cols[0]]==1)&(ds.attrs[test][:,cols[3]]==-1)]   # men, not already bald
men=np.random.default_rng(11).choice(pool,N,replace=False)
x=torch.stack([ds[int(i)]["img"] for i in men]).cuda()
y=to_cond_values(torch.from_numpy(ds.attrs[men][:,cols].astype("float32")),specs).cuda()
print(f"{N} men, none already bald. source: "+", ".join(
      f"{a} {float((torch.from_numpy(ds.attrs[men][:,cols[i]]).float()>0).float().mean()):.2f}"
      for i,a in enumerate(ATTRS)),flush=True)

s=cfg.train_conf._make_diffusion_conf(T).make_sampler()
res={}
with torch.no_grad():
    zs=[z.clone() for z in net.encode(x)]
    c=net.make_cond(zs,y)
    nmall=torch.ones_like(c["y_idx"],dtype=torch.bool)
    XT={"condXT": s.ddim_reverse_sample_loop(net,x,model_kwargs={"cond":c})["sample"],
        "uncondXT": s.ddim_reverse_sample_loop(net,x,model_kwargs={
            "cond":{"zs":c["zs"],"y_idx":c["y_idx"],"null_mask":nmall}})["sample"]}
    y2=y.clone(); y2[:,TARGET]=-y2[:,TARGET]; c2=net.make_cond(zs,y2)
    SETS={"null1_Bald":[3],"null2_BaldBeard":[2,3],"null4_all":[0,1,2,3]}
    print(f"\n{'config':34} {'g':>4} {'CC_Bald':>8} {'dBeard':>8} {'dYoung':>8} {'dMale':>7}",flush=True)
    for xk,xv in XT.items():
        rec=s.sample(model=net,noise=xv,model_kwargs={"cond":c}).cpu()
        base={a:float(R[a](rec).mean()) for a in ATTRS}
        for nk,idx in SETS.items():
            for g in GS:
                m=net if g==1 else MultiNullCFG(net,g,idx).cuda().eval()
                o=s.sample(model=m,noise=xv,model_kwargs={"cond":c2}).cpu()
                v={a:float(R[a](o).mean()) for a in ATTRS}
                res[f"{xk}|{nk}|{g}"]=dict(cc=v["Bald"],
                    dBeard=v["Beard"]-base["Beard"],dYoung=v["Young"]-base["Young"],
                    dMale=v["Male"]-base["Male"],raw=v,base=base)
                print(f"{xk+' / '+nk:34} {g:>4} {v['Bald']:8.3f} "
                      f"{v['Beard']-base['Beard']:+8.3f} {v['Young']-base['Young']:+8.3f} "
                      f"{v['Male']-base['Male']:+7.3f}",flush=True)
json.dump(res,open(f"{SP}/nullquant.json","w"),indent=2)
print("\nNULLQUANT DONE",flush=True)
