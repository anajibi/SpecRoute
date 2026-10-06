"""k=11 generative stats: is T=100 worth 2x the sampling cost over T=50?

Two things are measured, because they can disagree and only one of them was checked before:
  RECONSTRUCTION  -- DDIM invert then sample, no edit. More steps should always help here.
  EDIT QUALITY    -- do(Male) from the same x_T. This is what the counterfactual metrics use,
                     and it is a harder ask: inversion error and edit fidelity interact, so a
                     T that reconstructs better does not automatically edit better.

The pre-training FFHQ baseline on the same held-out split was 0.0398/31.29 dB at T=20,
0.0224/36.35 dB at T=50, 0.0158/39.45 dB at T=100 -- those are the numbers to beat/compare.
"""
import sys, glob, json, time, torch, numpy as np
sys.path.insert(0, "/home/exouser/SpecRoute/diffae_upstream")
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values
import torchvision.utils as vu

SPD = "/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
ATTRS = ["Male", "Young", "Beard", "Bald"]
_all = sorted(glob.glob("experiments/hdae/outputs/celebahq256_k11/checkpoints/*.ckpt"))
# Prefer a numbered checkpoint, but fall back to last.ckpt: the janitor and the chain both
# prune epoch=*.ckpt once s3 has them, and filtering "last" out left this glob empty.
ck = [c for c in _all if "last" not in c] or _all
if not ck: raise SystemExit("no k=11 checkpoint on disk; restore one from s3 first")
print("checkpoint:", ck[-1], flush=True)
cfg = load_hdae_config("experiments/hdae/configs/celebahq256_k11.yaml")
lit = HDAELitModule.load_from_checkpoint(ck[-1], conf=cfg.train_conf, map_location="cpu")
net = lit.ema_model.cuda().eval()
specs = cfg.hdae_conf.encoder.cond_specs

D = "experiments/hdae/data/packed"
ds = CelebAHQPacked(f"{D}/celebahq_256.lmdb", f"{D}/celebahq_256_attrs.npz")
names = ds.attribute_names; cols = [names.index(a) for a in ATTRS]
test = np.where(ds.partitions == 2)[0]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(test, 32, replace=False))
x = torch.stack([ds[int(i)]["img"] for i in idx]).cuda()
y = to_cond_values(torch.from_numpy(ds.attrs[idx][:, cols].astype("float32")), specs).cuda()

out = {}
print(f"\n{'T':>4s} {'recon MAE':>10s} {'recon PSNR':>11s} {'do(Male) delta':>15s} {'sec/img':>9s}")
for T in [20, 50, 100]:
    samp = cfg.train_conf._make_diffusion_conf(T).make_sampler()
    t0 = time.time(); R, E = [], []
    with torch.no_grad():
        for lo in range(0, len(x), 8):
            xb, yb = x[lo:lo+8], y[lo:lo+8]
            zs = [z.clone() for z in net.encode(xb)]
            c = net.make_cond(zs, yb)
            xT = samp.ddim_reverse_sample_loop(net, xb, model_kwargs={"cond": c})["sample"]
            r = samp.sample(model=net, noise=xT, model_kwargs={"cond": c})
            y2 = yb.clone(); y2[:, 0] = -y2[:, 0]
            e = samp.sample(model=net, noise=xT, model_kwargs={"cond": net.make_cond(zs, y2)})
            R.append(r); E.append(e)
    rec = torch.cat(R); ed = torch.cat(E); dt = time.time() - t0
    mae = (rec - x).abs().mean().item()
    psnr = float(10*torch.log10(4.0/((rec-x)**2).mean()))
    delta = (ed - rec).abs().mean().item()
    out[T] = dict(mae=mae, psnr=psnr, edit_delta=delta, sec_per_image=dt/len(x)/2)
    print(f"{T:4d} {mae:10.4f} {psnr:10.2f}dB {delta:15.4f} {dt/len(x)/2:9.2f}", flush=True)
    if T in (50, 100):
        vu.save_image(torch.cat([x[:6], rec[:6], ed[:6]]).add(1).div(2).clamp(0,1),
                      f"{SPD}/t{T}_grid.png", nrow=6)
json.dump(out, open("/home/exouser/SpecRoute/experiments/hdae/outputs/k11_t_sweep.json","w"), indent=2)
print("\nwrote experiments/hdae/outputs/k11_t_sweep.json")
