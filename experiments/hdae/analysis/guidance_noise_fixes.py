"""do(Bald) on k=11 across g, under four CFG variants.

  baseline   plain CFG, exactly what produced the noise
  interval   guidance restricted to t in [200,800]
  threshold  plain CFG + Imagen dynamic thresholding on the x0 estimate
  both       interval + threshold

Same eight male subjects, same inverted x_T for every variant and every g, so any difference
between panels is the variant or the guidance, never the sampling draw.
"""
import sys, glob, torch, numpy as np
sys.path.insert(0, "/home/exouser/SpecRoute/diffae_upstream")
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values
from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper as PlainCFG
from experiments.hdae.counterfactuals.cfg_fixes import IntervalCFG, dynamic_threshold
import torchvision.utils as vu

SPD="/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
ATTRS=["Male","Young","Beard","Bald"]; GS=[1,1.5,2,3,5]; N=8; AI=3
ck=sorted(glob.glob("experiments/hdae/outputs/celebahq256_k11/checkpoints/*.ckpt"))
ck=[c for c in ck if "last" not in c] or ck
cfg=load_hdae_config("experiments/hdae/configs/celebahq256_k11.yaml")
net=HDAELitModule.load_from_checkpoint(ck[-1], conf=cfg.train_conf, map_location="cpu").ema_model.cuda().eval()
samp=cfg.train_conf._make_diffusion_conf(50).make_sampler()
D="experiments/hdae/data/packed"
ds=CelebAHQPacked(f"{D}/celebahq_256.lmdb", f"{D}/celebahq_256_attrs.npz")
names=ds.attribute_names; cols=[names.index(a) for a in ATTRS]
test=np.where(ds.partitions==2)[0]; male=test[ds.attrs[test][:,cols[0]]==1]
idx=np.sort(np.random.default_rng(3).choice(male,N,replace=False))
x=torch.stack([ds[int(i)]["img"] for i in idx]).cuda()
y=to_cond_values(torch.from_numpy(ds.attrs[idx][:,cols].astype("float32")), cfg.hdae_conf.encoder.cond_specs).cuda()
with torch.no_grad():
    zs=[z.clone() for z in net.encode(x)]
    c=net.make_cond(zs,y)
    xT=samp.ddim_reverse_sample_loop(net,x,model_kwargs={"cond":c})["sample"]
    rec=samp.sample(model=net,noise=xT,model_kwargs={"cond":c})
y2=y.clone(); y2[:,AI]=-y2[:,AI]
c2=net.make_cond(zs,y2)

def hf(v):
    F=torch.fft.rfft2(v.float()); P=F.real**2+F.imag**2
    fy=torch.fft.fftfreq(v.shape[-2],device=v.device).abs().view(-1,1)
    fx=torch.fft.rfftfreq(v.shape[-1],device=v.device).abs().view(1,-1)
    return float(P[..., torch.sqrt(fy**2+fx**2)>0.25].sum()/P.sum())

VAR={"baseline":(False,False),"interval":(True,False),"threshold":(False,True),"both":(True,True)}
print(f"{'variant':10s} {'g':>4s} {'HF energy':>10s} {'clipped%':>9s} {'edit delta':>11s}")
print(f"{'source':10s} {'':4s} {hf(x):10.4f}")
print(f"{'recon':10s} {'':4s} {hf(rec):10.4f}")
out={}
for vname,(use_int,use_thr) in VAR.items():
    rows=[x.cpu(), rec.cpu()]
    for g in GS:
        if g==1: m=net
        elif use_int: m=IntervalCFG(net,float(g),200,800).cuda().eval()
        else: m=PlainCFG(net,float(g)).cuda().eval()
        dfn=dynamic_threshold(0.995) if use_thr else None
        with torch.no_grad():
            ed=samp.ddim_sample_loop(model=m, noise=xT, clip_denoised=True,
                                     denoised_fn=dfn, model_kwargs={"cond":c2})
        rows.append(ed.cpu())
        print(f"{vname:10s} {g:4g} {hf(ed):10.4f} {100*float((ed.abs()>0.98).float().mean()):8.2f}% "
              f"{float((ed-rec).abs().mean()):11.4f}", flush=True)
    vu.save_image(torch.cat(rows).add(1).div(2).clamp(0,1), f"{SPD}/f_{vname}.png", nrow=N, padding=2)
    print(f"  -> f_{vname}.png", flush=True)
print("BALD FIX DONE", flush=True)
