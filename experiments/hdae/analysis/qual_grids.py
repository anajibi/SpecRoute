"""Qualitative panels: reconstruction across T, and do(attr) across guidance, for both depths.

Same four held-out subjects and the same inverted x_T everywhere, so a difference between two
panels is the variable named in the caption and nothing else.
"""
import sys, glob, torch, numpy as np
sys.path.insert(0, "/home/exouser/SpecRoute/diffae_upstream")
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values
from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper as CFG
import torchvision.utils as vu

SPD="/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
ATTRS=["Male","Young","Beard","Bald"]; N=4
D="experiments/hdae/data/packed"
ds=CelebAHQPacked(f"{D}/celebahq_256.lmdb", f"{D}/celebahq_256_attrs.npz")
names=ds.attribute_names; cols=[names.index(a) for a in ATTRS]
test=np.where(ds.partitions==2)[0]
male=test[ds.attrs[test][:,cols[0]]==1]; fem=test[ds.attrs[test][:,cols[0]]==-1]
rng=np.random.default_rng(3)
mix=np.concatenate([rng.choice(fem,2,replace=False), rng.choice(male,2,replace=False)])
men=rng.choice(male,N,replace=False)

def load(tag):
    ck=sorted(glob.glob(f"experiments/hdae/outputs/celebahq256_{tag}/checkpoints/*.ckpt"))
    ck=[c for c in ck if "last" not in c] or ck
    cfg=load_hdae_config(f"experiments/hdae/configs/celebahq256_{tag}.yaml")
    lit=HDAELitModule.load_from_checkpoint(ck[-1], conf=cfg.train_conf, map_location="cpu")
    return cfg, lit.ema_model.cuda().eval()

def prep(idx, net, cfg):
    x=torch.stack([ds[int(i)]["img"] for i in idx]).cuda()
    y=to_cond_values(torch.from_numpy(ds.attrs[idx][:,cols].astype("float32")),
                     cfg.hdae_conf.encoder.cond_specs).cuda()
    with torch.no_grad(): zs=[z.clone() for z in net.encode(x)]
    return x,y,zs

for tag in ["k11","k1"]:
    cfg, net = load(tag)
    print(f"--- {tag} ---", flush=True)

    # A. reconstruction across T
    x,y,zs = prep(mix, net, cfg)
    rows=[x.cpu()]
    for T in [20,50,100]:
        s=cfg.train_conf._make_diffusion_conf(T).make_sampler()
        with torch.no_grad():
            c=net.make_cond(zs,y)
            xT=s.ddim_reverse_sample_loop(net,x,model_kwargs={"cond":c})["sample"]
            rows.append(s.sample(model=net,noise=xT,model_kwargs={"cond":c}).cpu())
    vu.save_image(torch.cat(rows).add(1).div(2).clamp(0,1), f"{SPD}/q_{tag}_T.png", nrow=N, padding=2)
    print(f"  wrote q_{tag}_T.png", flush=True)

    # B. guidance ladder per attribute, T=50
    s=cfg.train_conf._make_diffusion_conf(50).make_sampler()
    for ai,attr in enumerate(ATTRS):
        idx = men if attr in ("Beard","Bald") else mix
        x,y,zs = prep(idx, net, cfg)
        with torch.no_grad():
            c=net.make_cond(zs,y)
            xT=s.ddim_reverse_sample_loop(net,x,model_kwargs={"cond":c})["sample"]
            rec=s.sample(model=net,noise=xT,model_kwargs={"cond":c})
            y2=y.clone(); y2[:,ai]=-y2[:,ai]
            c2=net.make_cond(zs,y2)
            rows=[x.cpu(), rec.cpu()]
            for g in [1,1.5,2,3,5]:
                m = net if g==1 else CFG(net,float(g)).cuda().eval()
                rows.append(s.sample(model=m,noise=xT,model_kwargs={"cond":c2}).cpu())
        vu.save_image(torch.cat(rows).add(1).div(2).clamp(0,1), f"{SPD}/q_{tag}_{attr}.png", nrow=N, padding=2)
        print(f"  wrote q_{tag}_{attr}.png", flush=True)
    del net; torch.cuda.empty_cache()
print("QUAL DONE", flush=True)
