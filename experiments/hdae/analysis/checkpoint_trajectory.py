"""Does the model still improve with steps, holding dropout constant?

All five checkpoints come from AFTER the dropout change, so cfg_drop_prob/attr_dropout_prob are
fixed and the only variable is training steps. This is the comparison the parent-vs-cd015r
number cannot make, because that one moved dropout and steps together.

Same 32 held-out subjects, same seed, same T=50 protocol as t_sweep.py.
"""
import sys, os, json, subprocess, torch, numpy as np
sys.path.insert(0, "/home/exouser/SpecRoute/diffae_upstream")
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values

SP = "/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
BK = "najibi-research-7f2a"; PFX = "hdae-handoff/celebahq256/k11_cd015r"
CKPTS = ["epoch=91-step=34375.ckpt","epoch=99-step=37500.ckpt","epoch=108-step=40625.ckpt",
         "epoch=116-step=43750.ckpt","epoch=124-step=46875.ckpt"]
TMP = "/home/exouser/SpecRoute/experiments/hdae/outputs/_traj"
os.makedirs(TMP, exist_ok=True)

cfg = load_hdae_config("experiments/hdae/configs/celebahq256_k11_cd015r.yaml")
specs = cfg.hdae_conf.encoder.cond_specs
D = "experiments/hdae/data/packed"
ds = CelebAHQPacked(f"{D}/celebahq_256.lmdb", f"{D}/celebahq_256_attrs.npz")
names = ds.attribute_names; cols = [names.index(a) for a in ["Male","Young","Beard","Bald"]]
test = np.where(ds.partitions == 2)[0]
idx = np.sort(np.random.default_rng(0).choice(test, 32, replace=False))
x = torch.stack([ds[int(i)]["img"] for i in idx]).cuda()
y = to_cond_values(torch.from_numpy(ds.attrs[idx][:, cols].astype("float32")), specs).cuda()

out = {}
print(f"{'step':>7} {'MAE':>8} {'PSNR':>9} {'do(Male) delta':>15}", flush=True)
for name in CKPTS:
    step = int(name.split("step=")[1].split(".")[0])
    loc = f"{TMP}/{name}"
    if not os.path.exists(loc):
        subprocess.run(["aws","s3","cp",f"s3://{BK}/{PFX}/{name}",loc,"--only-show-errors"],check=True)
    lit = HDAELitModule.load_from_checkpoint(loc, conf=cfg.train_conf, map_location="cpu")
    net = lit.ema_model.cuda().eval()
    s = cfg.train_conf._make_diffusion_conf(50).make_sampler()
    recs, eds = [], []
    with torch.no_grad():
        for lo in range(0, len(x), 8):
            xb, yb = x[lo:lo+8], y[lo:lo+8]
            zs = net.encode(xb)
            c = net.make_cond(zs, yb)
            xT = s.ddim_reverse_sample_loop(net, xb, model_kwargs={"cond": c})["sample"]
            recs.append(s.sample(model=net, noise=xT, model_kwargs={"cond": c}))
            y2 = yb.clone(); y2[:, 0] = -y2[:, 0]
            ed = s.sample(model=net, noise=xT, model_kwargs={"cond": net.make_cond(zs, y2)})
            eds.append((ed - recs[-1]).abs().mean(dim=(1,2,3)))
    rec = torch.cat(recs)
    mse = ((rec - x) ** 2).mean(dim=(1,2,3))
    psnr = float((10*torch.log10(4.0/mse)).mean())
    mae = float((rec - x).abs().mean())
    delta = float(torch.cat(eds).mean())
    out[step] = dict(mae=mae, psnr=psnr, edit_delta=delta)
    print(f"{step:>7} {mae:8.4f} {psnr:8.2f}dB {delta:15.4f}", flush=True)
    del lit, net; torch.cuda.empty_cache(); os.remove(loc)
json.dump(out, open(f"{SP}/traj.json","w"), indent=2)
print("\nTRAJ DONE", flush=True)
