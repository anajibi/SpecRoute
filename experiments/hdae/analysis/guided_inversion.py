"""Does the inversion/sampling asymmetry cause the noise?

The pipeline inverts to x_T with the UNGUIDED model and then samples back with the GUIDED one:

    x_T   = invert(x, eps_theta)                 <- vector field A, g=1
    x_hat = sample(x_T, eps_theta + g*delta)     <- vector field B, g>1

At g=1 those are the same field and the trajectory closes. At g=5 they are different fields, so
x_T is not the right starting point for the reverse path and the mismatch accumulates over every
step. This is exactly why null-text inversion exists in the editing literature.

Test: invert with the SAME guidance scale applied to the SOURCE attributes, so both directions
use a matched field, then sample toward the TARGET attributes.
"""
import sys, glob, torch, numpy as np
sys.path.insert(0, "/home/exouser/SpecRoute/diffae_upstream")
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values
from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper as PlainCFG
from experiments.hdae.counterfactuals.cfg_fixes import IntervalCFG
import torchvision.utils as vu

SPD="/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
def hf(v):
    F=torch.fft.rfft2(v.float()); P=F.real**2+F.imag**2
    fy=torch.fft.fftfreq(v.shape[-2],device=v.device).abs().view(-1,1)
    fx=torch.fft.rfftfreq(v.shape[-1],device=v.device).abs().view(1,-1)
    return float(P[..., torch.sqrt(fy**2+fx**2)>0.25].sum()/P.sum())

ATTRS=["Male","Young","Beard","Bald"]; AI=3; GS=[2,3,5]; N=8
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
    c_src=net.make_cond(zs,y)
    xT0=samp.ddim_reverse_sample_loop(net,x,model_kwargs={"cond":c_src})["sample"]
    rec=samp.sample(model=net,noise=xT0,model_kwargs={"cond":c_src})
y2=y.clone(); y2[:,AI]=-y2[:,AI]
c_tgt=net.make_cond(zs,y2)

print(f"reference: source HF {hf(x):.4f}   recon HF {hf(rec):.4f}")
print(f"\n{'variant':22s} {'g':>3s} {'HF energy':>10s} {'edit':>8s} {'HF/edit':>9s}")
rows={}
for vname, guided_inv, interval in [("plain inv + plain CFG",False,False),
                                    ("GUIDED inv + plain CFG",True,False),
                                    ("GUIDED inv + interval",True,True)]:
    panel=[x.cpu(), rec.cpu()]
    for g in GS:
        with torch.no_grad():
            if guided_inv:
                mi = IntervalCFG(net,float(g),200,800).cuda().eval() if interval else PlainCFG(net,float(g)).cuda().eval()
                xT = samp.ddim_reverse_sample_loop(mi,x,model_kwargs={"cond":c_src})["sample"]
            else:
                xT = xT0
            ms = IntervalCFG(net,float(g),200,800).cuda().eval() if interval else PlainCFG(net,float(g)).cuda().eval()
            ed = samp.sample(model=ms, noise=xT, model_kwargs={"cond":c_tgt})
        h=hf(ed); e=float((ed-rec).abs().mean())
        print(f"{vname:22s} {g:3g} {h:10.4f} {e:8.4f} {h/e:9.4f}", flush=True)
        panel.append(ed.cpu())
    key=vname.split()[0].lower()+("_int" if interval else "")
    vu.save_image(torch.cat(panel).add(1).div(2).clamp(0,1), f"{SPD}/gi_{key}.png", nrow=N, padding=2)
print("GUIDED INV DONE", flush=True)
