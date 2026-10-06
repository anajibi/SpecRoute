"""do(Bald) under 6 configurations: {conditional, unconditional} x_T  x  {null 1, 2, 4 attrs}.

Motivating observation: at high g the Bald edit also grows a beard. Two suspects are tested
separately here.

  NULL SET -- what the unconditional branch removes decides what guidance amplifies.
    null {Bald}        guidance direction is Bald alone; Beard is identical in both branches
                       and therefore contributes nothing to the guidance term.
    null {Bald,Beard}  guidance also carries Beard, pushed toward its ORIGINAL (unflipped)
                       value -- so it should actively hold the beard where it was.
    null {all 4}       standard CFG. Guidance carries every attribute at once, which is the
                       configuration that produced the beard.

  x_T -- the inverted latent. Conditional inversion runs DDIM reverse with the true attributes,
    unconditional with everything nulled. The sampler is then guided, so a conditionally
    inverted x_T is matched to a different field than the one it is sampled under; the
    guided-inversion analysis measured that asymmetry at 78% of the high-frequency excess.

Same 4 men, same seed, same checkpoint (cd015r) throughout. Rows: original, reconstruction,
then g = 1, 1.5, 2, 3, 5.
"""
import sys, glob, torch, numpy as np
sys.path.insert(0,"/home/exouser/SpecRoute/diffae_upstream"); sys.path.insert(0,"/home/exouser/SpecRoute")
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values
import torchvision.utils as vu

SP="/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
ATTRS=["Male","Young","Beard","Bald"]; TARGET=3          # Bald
GS=[1,1.5,2,3,5]; N=4; T=50

class MultiNullCFG(torch.nn.Module):
    """CFG whose unconditional branch nulls exactly the attributes in `idx`."""
    def __init__(self, base, g, idx):
        super().__init__(); self.base=base; self.g=float(g); self.idx=list(idx)
    def forward(self, x, t, cond, **kw):
        co=self.base.forward(x=x,t=t,cond=cond,**kw)
        if self.g==1.0: return co
        nm=torch.zeros_like(cond["y_idx"],dtype=torch.bool); nm[:,self.idx]=True
        uo=self.base.forward(x=x,t=t,cond={"zs":cond["zs"],"y_idx":cond["y_idx"],"null_mask":nm},**kw)
        return co.__class__(pred=uo.pred+self.g*(co.pred-uo.pred), cond=cond)

cfg=load_hdae_config("experiments/hdae/configs/celebahq256_k11_cd015r.yaml")
ck=sorted(glob.glob("experiments/hdae/outputs/celebahq256_k11_cd015r/checkpoints/*.ckpt"))
net=HDAELitModule.load_from_checkpoint(ck[-1],conf=cfg.train_conf,map_location="cpu").ema_model.cuda().eval()
specs=cfg.hdae_conf.encoder.cond_specs

D="experiments/hdae/data/packed"
ds=CelebAHQPacked(f"{D}/celebahq_256.lmdb", f"{D}/celebahq_256_attrs.npz")
nm_=ds.attribute_names; cols=[nm_.index(a) for a in ATTRS]
test=np.where(ds.partitions==2)[0]
male=test[ds.attrs[test][:,cols[0]]==1]
men=np.random.default_rng(3).choice(male,N,replace=False)     # same subjects as qual.py
x=torch.stack([ds[int(i)]["img"] for i in men]).cuda()
y=to_cond_values(torch.from_numpy(ds.attrs[men][:,cols].astype("float32")),specs).cuda()
print("subjects:",men.tolist())
print("their attrs (Male,Young,Beard,Bald):\n", ds.attrs[men][:,cols])

s=cfg.train_conf._make_diffusion_conf(T).make_sampler()
with torch.no_grad():
    zs=[z.clone() for z in net.encode(x)]
    c_cond=net.make_cond(zs,y)
    # two inversions
    xT={}
    xT["condXT"]=s.ddim_reverse_sample_loop(net,x,model_kwargs={"cond":c_cond})["sample"]
    nm_all=torch.ones_like(c_cond["y_idx"],dtype=torch.bool)
    c_null={"zs":c_cond["zs"],"y_idx":c_cond["y_idx"],"null_mask":nm_all}
    xT["uncondXT"]=s.ddim_reverse_sample_loop(net,x,model_kwargs={"cond":c_null})["sample"]
    print("xT delta between the two inversions:",
          (xT['condXT']-xT['uncondXT']).abs().mean().item())

    y2=y.clone(); y2[:,TARGET]=-y2[:,TARGET]
    c2=net.make_cond(zs,y2)
    NULLSETS={"null1_Bald":[3], "null2_BaldBeard":[2,3], "null4_all":[0,1,2,3]}
    for xk,xv in xT.items():
        rec=s.sample(model=net,noise=xv,model_kwargs={"cond":c_cond})
        for nk,idx in NULLSETS.items():
            rows=[x.cpu(),rec.cpu()]
            for g in GS:
                m = net if g==1 else MultiNullCFG(net,g,idx).cuda().eval()
                rows.append(s.sample(model=m,noise=xv,model_kwargs={"cond":c2}).cpu())
            f=f"{SP}/six_{xk}_{nk}.png"
            vu.save_image(torch.cat(rows).add(1).div(2).clamp(0,1),f,nrow=N,padding=2)
            print("wrote",f.split("/")[-1],flush=True)
print("SIX DONE",flush=True)
